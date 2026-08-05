"""Synthetic FITS catalogs, byte-identical in layout to the Roman input catalogs.

12 columns, NAXIS1 = 91: source_id (K), ra/dec + 8 fluxes (D), type (3A).
"""

from __future__ import annotations

import numpy as np

from fitsq.fits_lite import BLOCK, CARD

FLUX_NAMES = tuple(f"flux_{band}" for band in "FHJKRYZW")

ROW_DTYPE = np.dtype(
    {
        "names": ["source_id", "ra", "dec", *FLUX_NAMES, "type"],
        "formats": [">i8", ">f8", ">f8", *([">f8"] * 8), "S3"],
    }
)


def card(key: str, value: object = None, comment: str = "") -> str:
    if value is None:
        text = f"{key:<8}{comment:<72}"
        return text[:CARD]
    if isinstance(value, bool):
        rendered = "T" if value else "F"
    elif isinstance(value, str):
        rendered = f"'{value}'"
    else:
        rendered = str(value)
    body = f"{rendered:>20}"
    if comment:
        body = f"{body} / {comment}"
    return f"{key:<8}= {body:<70}"[:CARD]


def header_block(cards: list[str]) -> bytes:
    text = "".join(f"{c:<{CARD}}" for c in [*cards, "END"])
    padding = (-len(text)) % BLOCK
    return (text + " " * padding).encode("ascii")


def primary_header(extra_cards: list[str] | None = None) -> bytes:
    cards = [
        card("SIMPLE", True, "conforms to FITS standard"),
        card("BITPIX", 8),
        card("NAXIS", 0),
        card("EXTEND", True),
    ]
    cards.extend(extra_cards or [])
    return header_block(cards)


def bintable_header(nrows: int, extra_cards: list[str] | None = None) -> bytes:
    cards = [
        card("XTENSION", "BINTABLE"),
        card("BITPIX", 8),
        card("NAXIS", 2),
        card("NAXIS1", ROW_DTYPE.itemsize),
        card("NAXIS2", nrows),
        card("PCOUNT", 0),
        card("GCOUNT", 1),
        card("TFIELDS", 12),
    ]
    columns = [
        ("source_id", "1K"),
        ("ra", "1D"),
        ("dec", "1D"),
        *[(name, "1D") for name in FLUX_NAMES],
        ("type", "3A"),
    ]
    for i, (name, tform) in enumerate(columns, start=1):
        cards.append(card(f"TTYPE{i}", name))
        cards.append(card(f"TFORM{i}", tform))
    cards.extend(extra_cards or [])
    return header_block(cards)


def rows_bytes(ra: np.ndarray, dec: np.ndarray) -> bytes:
    rows = np.zeros(len(ra), dtype=ROW_DTYPE)
    rows["source_id"] = np.arange(len(ra))
    rows["ra"] = ra
    rows["dec"] = dec
    for name in FLUX_NAMES:
        rows[name] = 1.0
    rows["type"] = b"STA"
    raw = rows.tobytes()
    return raw + b"\0" * ((-len(raw)) % BLOCK)


def catalog(ra: np.ndarray, dec: np.ndarray, extra_cards: list[str] | None = None) -> bytes:
    """A complete single-BINTABLE FITS file."""
    return primary_header() + bintable_header(len(ra), extra_cards) + rows_bytes(ra, dec)


def patch(
    ra_center: float, dec_center: float, n: int = 400, span: float = 0.15, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Random positions in a small tile-sized square around a centre."""
    rng = np.random.default_rng(seed)
    half = span / 2.0
    dec = dec_center + rng.uniform(-half, half, n)
    scale = max(np.cos(np.radians(dec_center)), 1e-6)
    ra = (ra_center + rng.uniform(-half, half, n) / scale) % 360.0
    return ra, dec
