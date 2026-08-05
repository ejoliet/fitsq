"""Crawl orchestration: sample rows over S3, build per-file MOCs, upsert, resume."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from mocpy import MOC

from . import fits_lite
from .s3io import S3Object, S3Reader
from .store import FileRow, Store

#: bytes per Range-GET when a whole table is read back (validate)
FULL_READ_CHUNK = 8 << 20


@dataclass(frozen=True)
class CrawlOptions:
    workers: int = 16
    sample_rows: int = 50_000
    samples: int = 3
    order: int = 9
    dilate: int = 1


@dataclass
class CrawlStats:
    total: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_read: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class FileResult:
    row: FileRow
    cards: tuple[tuple[int, str, str, str], ...]


Progress = Callable[[CrawlStats, float], None]

DEFAULT_OPTIONS = CrawlOptions()


def build_moc(ra: np.ndarray, dec: np.ndarray, order: int, dilate: int) -> MOC:
    """MOC of sampled positions, dilated by ``dilate`` cell borders at ``order``."""
    moc = MOC.from_lonlat(ra * u.deg, dec * u.deg, max_norder=order)
    for _ in range(max(0, dilate)):
        moc.add_neighbours()
    return moc


def coord_bbox(ra: np.ndarray, dec: np.ndarray) -> tuple[float, float, float, float, bool]:
    """Smallest lon/lat box around the points; ``wraps`` when it crosses RA=0.

    The RA interval is the complement of the largest angular gap between
    consecutive sorted right ascensions.
    """
    ras = np.sort(np.asarray(ra, dtype=float) % 360.0)
    dec_min = float(np.min(dec))
    dec_max = float(np.max(dec))
    if ras.size == 1:
        return float(ras[0]), float(ras[0]), dec_min, dec_max, False
    gaps = np.diff(ras)
    wrap_gap = ras[0] + 360.0 - ras[-1]
    if gaps.size == 0 or wrap_gap >= float(np.max(gaps)):
        return float(ras[0]), float(ras[-1]), dec_min, dec_max, False
    i = int(np.argmax(gaps))
    return float(ras[i + 1]), float(ras[i]), dec_min, dec_max, True


def _clean_coords(rows: np.ndarray, ra_col: str, dec_col: str) -> tuple[np.ndarray, np.ndarray]:
    ra = np.asarray(rows[ra_col], dtype=float).ravel()
    dec = np.asarray(rows[dec_col], dtype=float).ravel()
    good = np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90.0)
    return ra[good], dec[good]


def index_file(reader: S3Reader, obj: S3Object, opts: CrawlOptions) -> FileResult:
    """Sample one S3 FITS file and return its index row. Raises on bad files."""
    info = fits_lite.read_bintable_info(reader.fetcher(obj.uri))
    if info.nrows <= 0:
        raise fits_lite.FitsError("empty table (NAXIS2 = 0)")
    ras: list[np.ndarray] = []
    decs: list[np.ndarray] = []
    for first, count in fits_lite.sample_windows(info.nrows, opts.samples, opts.sample_rows):
        offset = info.data_offset + first * info.row_bytes
        buf = reader.ranged_get(obj.uri, offset, count * info.row_bytes)
        if not buf:
            continue
        ra, dec = _clean_coords(fits_lite.decode_rows(buf, info.dtype), info.ra_col, info.dec_col)
        ras.append(ra)
        decs.append(dec)
    ra_all = np.concatenate(ras) if ras else np.empty(0)
    dec_all = np.concatenate(decs) if decs else np.empty(0)
    if ra_all.size == 0:
        raise fits_lite.FitsError("no finite ra/dec values in sampled rows")
    moc = build_moc(ra_all, dec_all, opts.order, opts.dilate)
    ra_min, ra_max, dec_min, dec_max, wraps = coord_bbox(ra_all, dec_all)
    row = FileRow(
        uri=obj.uri,
        etag=obj.etag,
        size=obj.size,
        nrows=info.nrows,
        row_bytes=info.row_bytes,
        data_offset=info.data_offset,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=dec_min,
        dec_max=dec_max,
        ra_wraps=wraps,
        moc_json=moc.to_string(format="json"),
    )
    cards = tuple((hdu, c.key, c.value, c.comment) for hdu, c in info.cards)
    return FileResult(row=row, cards=cards)


def crawl(
    prefix: str,
    store: Store,
    reader: S3Reader,
    opts: CrawlOptions = DEFAULT_OPTIONS,
    on_progress: Progress | None = None,
) -> CrawlStats:
    """Index every FITS file under ``prefix``. Unchanged files (same ETag) are skipped.

    Per-file failures are logged to ``crawl_errors`` and never abort the crawl.
    """
    known = store.etags()
    objects = list(reader.list_fits(prefix))
    todo = [obj for obj in objects if known.get(obj.uri) != obj.etag]
    stats = CrawlStats(total=len(objects), skipped=len(objects) - len(todo))
    started = time.monotonic()

    def report() -> None:
        stats.bytes_read = reader.bytes_read
        if on_progress is not None:
            on_progress(stats, time.monotonic() - started)

    report()
    with ThreadPoolExecutor(max_workers=max(1, opts.workers)) as pool:
        futures = {pool.submit(index_file, reader, obj, opts): obj for obj in todo}
        for future in as_completed(futures):
            obj = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                stats.failed += 1
                message = f"{type(exc).__name__}: {exc}"
                stats.errors.append((obj.uri, message))
                store.record_error(obj.uri, message)
            else:
                store.upsert_file(result.row, result.cards)
                stats.indexed += 1
            report()
    store.set_meta("prefix", prefix)
    store.set_meta("order", str(opts.order))
    store.set_meta("dilate", str(opts.dilate))
    store.set_meta("samples", str(opts.samples))
    store.set_meta("sample_rows", str(opts.sample_rows))
    store.set_meta("last_crawl", time.strftime("%Y-%m-%dT%H:%M:%S"))
    return stats


# -- validation -----------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    uri: str
    missing_cells: int
    reason: str = "sampled MOC does not cover full-read MOC"


def uncovered(inner: MOC, outer: MOC) -> MOC:
    """The part of ``inner`` that ``outer`` does not cover.

    Not `MOC.difference`: in mocpy 0.20 that returns an *empty* MOC when the two
    operands are disjoint (`3/0-3`.difference(`3/100-103`) yields nothing instead
    of `2/0`), which would make the validate gate pass unconditionally.
    Intersecting with the complement is correct for disjoint and overlapping input.
    """
    return inner.intersection(outer.complement())


def full_moc(reader: S3Reader, row: FileRow, order: int) -> MOC:
    """Rebuild a file's true MOC by reading every row (no dilation)."""
    info = fits_lite.read_bintable_info(reader.fetcher(row.uri))
    rows_per_chunk = max(1, FULL_READ_CHUNK // info.row_bytes)
    mocs: list[MOC] = []
    for first in range(0, info.nrows, rows_per_chunk):
        count = min(rows_per_chunk, info.nrows - first)
        buf = reader.ranged_get(
            row.uri, info.data_offset + first * info.row_bytes, count * info.row_bytes
        )
        if not buf:
            break
        ra, dec = _clean_coords(fits_lite.decode_rows(buf, info.dtype), info.ra_col, info.dec_col)
        if ra.size:
            mocs.append(MOC.from_lonlat(ra * u.deg, dec * u.deg, max_norder=order))
    if not mocs:
        return MOC.new_empty(order)
    out = mocs[0]
    for moc in mocs[1:]:
        out = out.union(moc)
    return out


def validate(
    rows: Sequence[FileRow],
    reader: S3Reader,
    order: int,
    n: int = 10,
    rng: random.Random | None = None,
) -> list[Violation]:
    """Full-read ``n`` random indexed files and assert true MOC ⊆ stored MOC."""
    picker = rng or random.Random()
    sample = picker.sample(list(rows), min(n, len(rows)))
    violations: list[Violation] = []
    for row in sample:
        stored = MOC.from_string(row.moc_json, format="json")
        truth = full_moc(reader, row, order)
        missing = uncovered(truth, stored)
        if not missing.empty():
            violations.append(Violation(uri=row.uri, missing_cells=len(missing.flatten())))
    return violations
