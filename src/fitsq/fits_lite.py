"""Minimal FITS reader: 80-char card scan, data offsets, TFORM -> numpy dtype.

Only what the indexer needs: primary header, HDU1 BINTABLE geometry, and the
structured dtype of one row. No astropy.io.fits dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import prod
from typing import Protocol

import numpy as np

BLOCK = 2880
CARD = 80
CARDS_PER_BLOCK = BLOCK // CARD
#: header blocks fetched per Range-GET while scanning for END
HEADER_CHUNK_BLOCKS = 4

# TFORM codes are uppercase per the FITS standard; P/Q may carry a trailing spec.
_TFORM_RE = re.compile(r"^\s*(\d*)([LXBIJKAEDCMPQ])(.*)$")

#: FITS TFORM code -> numpy big-endian dtype string
_TFORM_DTYPE = {
    "L": "S1",
    "B": "u1",
    "I": ">i2",
    "J": ">i4",
    "K": ">i8",
    "E": ">f4",
    "D": ">f8",
    "C": ">c8",
    "M": ">c16",
}


class FitsError(Exception):
    """Malformed or unsupported FITS structure."""


class Fetch(Protocol):
    """Byte-range reader: ``fetch(offset, length) -> bytes``."""

    def __call__(self, offset: int, length: int) -> bytes: ...


@dataclass(frozen=True)
class Card:
    key: str
    value: str
    comment: str


@dataclass(frozen=True)
class Header:
    cards: tuple[Card, ...]
    nbytes: int  #: header size on disk, multiple of BLOCK

    def get(self, key: str) -> str | None:
        for card in self.cards:
            if card.key == key:
                return card.value
        return None

    def int(self, key: str, default: int | None = None) -> int:
        raw = self.get(key)
        if raw is None:
            if default is None:
                raise FitsError(f"missing required keyword {key}")
            return default
        try:
            return int(float(raw))
        except ValueError as exc:
            raise FitsError(f"keyword {key} is not numeric: {raw!r}") from exc


@dataclass(frozen=True)
class BinTableInfo:
    """Everything needed to Range-GET and decode rows of an HDU1 BINTABLE."""

    nrows: int
    row_bytes: int
    data_offset: int
    dtype: np.dtype[np.void]
    ra_col: str
    dec_col: str
    primary: Header
    table: Header
    cards: tuple[tuple[int, Card], ...] = field(default=())


def _split_value_comment(field_text: str) -> tuple[str, str]:
    """Split a card's value field from its trailing comment, honouring quotes."""
    i = 0
    n = len(field_text)
    while i < n and field_text[i] == " ":
        i += 1
    if i < n and field_text[i] == "'":
        i += 1
        chars: list[str] = []
        while i < n:
            if field_text[i] == "'":
                if i + 1 < n and field_text[i + 1] == "'":  # '' escapes a quote
                    chars.append("'")
                    i += 2
                    continue
                i += 1
                break
            chars.append(field_text[i])
            i += 1
        value = "".join(chars).rstrip()
        rest = field_text[i:]
    else:
        slash = field_text.find("/", i)
        if slash == -1:
            return field_text.strip(), ""
        value = field_text[i:slash].strip()
        rest = field_text[slash:]
    slash = rest.find("/")
    comment = rest[slash + 1 :].strip() if slash != -1 else ""
    return value, comment


def parse_card(raw: str) -> Card | None:
    """Parse one 80-char card. Returns None for blank cards."""
    key = raw[:8].strip()
    rest = raw[8:]
    if not key:
        return None
    if rest.startswith("= "):
        value, comment = _split_value_comment(rest[2:])
        return Card(key, value, comment)
    # COMMENT / HISTORY / END and other valueless cards
    return Card(key, "", rest.strip())


def parse_header_blocks(data: bytes) -> tuple[tuple[Card, ...], bool]:
    """Parse whole 2880-byte blocks. Second element is True once END is seen."""
    cards: list[Card] = []
    nblocks = len(data) // BLOCK
    for b in range(nblocks):
        block = data[b * BLOCK : (b + 1) * BLOCK].decode("ascii", errors="replace")
        for c in range(CARDS_PER_BLOCK):
            raw = block[c * CARD : (c + 1) * CARD]
            if raw[:8].strip() == "END":
                return tuple(cards), True
            card = parse_card(raw)
            if card is not None:
                cards.append(card)
    return tuple(cards), False


def read_header(fetch: Fetch, offset: int) -> Header:
    """Read a header starting at ``offset``, growing the read until END is found."""
    buf = b""
    while True:
        chunk = fetch(offset + len(buf), HEADER_CHUNK_BLOCKS * BLOCK)
        if not chunk:
            raise FitsError(f"truncated header at offset {offset}")
        buf += chunk
        usable = len(buf) - len(buf) % BLOCK
        cards, done = parse_header_blocks(buf[:usable])
        if done:
            # END lives in the block that completed the header; round up to it.
            nbytes = _header_nbytes(buf[:usable])
            return Header(cards, nbytes)
        if len(chunk) < HEADER_CHUNK_BLOCKS * BLOCK:
            raise FitsError(f"header at offset {offset} has no END card")


def _header_nbytes(data: bytes) -> int:
    """Byte length of the header, i.e. blocks up to and including the END card."""
    for b in range(len(data) // BLOCK):
        block = data[b * BLOCK : (b + 1) * BLOCK].decode("ascii", errors="replace")
        for c in range(CARDS_PER_BLOCK):
            if block[c * CARD : c * CARD + 8].strip() == "END":
                return (b + 1) * BLOCK
    raise FitsError("no END card")


def padded(nbytes: int) -> int:
    """Round up to a whole number of 2880-byte blocks."""
    return -(-nbytes // BLOCK) * BLOCK


def data_nbytes(header: Header) -> int:
    """Size of an HDU's data section, padding excluded."""
    naxis = header.int("NAXIS")
    if naxis == 0:
        return 0
    dims = [header.int(f"NAXIS{i}") for i in range(1, naxis + 1)]
    bitpix = header.int("BITPIX")
    gcount = header.int("GCOUNT", 1)
    pcount = header.int("PCOUNT", 0)
    return (prod(dims) + pcount) * gcount * (abs(bitpix) // 8)


def tform_to_dtype(tform: str) -> tuple[str, int]:
    """Map a TFORM value to a numpy dtype string and its byte width."""
    match = _TFORM_RE.match(tform)
    if match is None:
        raise FitsError(f"unsupported TFORM {tform!r}")
    repeat = int(match.group(1) or 1)
    code = match.group(2)
    if code in ("P", "Q"):
        raise FitsError(f"variable-length column not supported: {tform!r}")
    if match.group(3).strip():
        raise FitsError(f"unsupported TFORM {tform!r}")
    if code == "A":
        return f"S{repeat}", repeat
    if code == "X":
        nbytes = -(-repeat // 8)
        return f"S{nbytes}", nbytes
    base = _TFORM_DTYPE.get(code)
    if base is None:
        raise FitsError(f"unsupported TFORM code {code!r}")
    width = np.dtype(base).itemsize
    if repeat == 1:
        return base, width
    return f"{repeat}{base}", repeat * width


def row_dtype(header: Header) -> np.dtype[np.void]:
    """Build the structured dtype of one BINTABLE row."""
    tfields = header.int("TFIELDS")
    names: list[str] = []
    formats: list[str] = []
    total = 0
    for i in range(1, tfields + 1):
        tform = header.get(f"TFORM{i}")
        if tform is None:
            raise FitsError(f"missing TFORM{i}")
        spec, width = tform_to_dtype(tform)
        name = (header.get(f"TTYPE{i}") or "").strip() or f"col{i}"
        while name in names:
            name = f"{name}_{i}"
        names.append(name)
        formats.append(spec)
        total += width
    row_bytes = header.int("NAXIS1")
    if total != row_bytes:
        raise FitsError(f"column widths sum to {total}, NAXIS1 is {row_bytes}")
    dtype = np.dtype({"names": names, "formats": formats})
    if dtype.itemsize != row_bytes:
        raise FitsError(f"dtype itemsize {dtype.itemsize} != NAXIS1 {row_bytes}")
    return dtype


def find_coord_columns(dtype: np.dtype[np.void]) -> tuple[str, str]:
    """Locate the RA/Dec columns, case-insensitively."""
    names = dtype.names or ()
    lower = {name.lower(): name for name in names}
    ra = next((lower[c] for c in ("ra", "raj2000", "ra_deg", "alpha") if c in lower), None)
    dec = next((lower[c] for c in ("dec", "decj2000", "dec_deg", "delta") if c in lower), None)
    if ra is None or dec is None:
        raise FitsError(f"no ra/dec columns among {list(names)}")
    return ra, dec


def read_bintable_info(fetch: Fetch) -> BinTableInfo:
    """Read the primary + HDU1 headers and derive the row layout of HDU1."""
    primary = read_header(fetch, 0)
    if primary.get("SIMPLE") is None:
        raise FitsError("not a FITS file: no SIMPLE keyword")
    table_offset = primary.nbytes + padded(data_nbytes(primary))
    table = read_header(fetch, table_offset)
    xtension = (table.get("XTENSION") or "").upper()
    if xtension not in ("BINTABLE", "A3DTABLE"):
        raise FitsError(f"HDU1 is {xtension or 'missing'}, expected BINTABLE")
    dtype = row_dtype(table)
    ra_col, dec_col = find_coord_columns(dtype)
    cards = tuple([(0, card) for card in primary.cards] + [(1, card) for card in table.cards])
    return BinTableInfo(
        nrows=table.int("NAXIS2"),
        row_bytes=table.int("NAXIS1"),
        data_offset=table_offset + table.nbytes,
        dtype=dtype,
        ra_col=ra_col,
        dec_col=dec_col,
        primary=primary,
        table=table,
        cards=cards,
    )


def decode_rows(buf: bytes, dtype: np.dtype[np.void]) -> np.ndarray:
    """Zero-copy view of a row buffer, ignoring a trailing partial row."""
    usable = len(buf) - len(buf) % dtype.itemsize
    return np.frombuffer(buf[:usable], dtype=dtype)


def sample_windows(nrows: int, samples: int, sample_rows: int) -> list[tuple[int, int]]:
    """Evenly spaced ``(first_row, nrows)`` windows: head, tail, and middles.

    Falls back to a single whole-table window when the table is small enough
    that sampling would read most of it anyway (Open Question 3).
    """
    if nrows <= 0:
        return []
    samples = max(1, samples)
    if nrows <= samples * sample_rows:
        return [(0, nrows)]
    if samples == 1:
        return [(0, sample_rows)]
    span = nrows - sample_rows
    starts = sorted({round(k * span / (samples - 1)) for k in range(samples)})
    return [(start, sample_rows) for start in starts]
