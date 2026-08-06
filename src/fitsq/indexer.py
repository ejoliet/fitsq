"""Crawl orchestration: build per-file MOCs, upsert, resume.

Two ways to establish a file's coverage:

* **from its name** (default) — ``cat-<pix>.fits`` names a galactic HEALPix
  cell, so coverage is exact and needs no row reads at all. See
  :mod:`fitsq.naming`.
* **by sampling rows** (``--sample``) — the original approach, kept as the
  independent check that the naming convention still holds. ``validate`` is the
  standing guard: it rebuilds the footprint from every row and reports any file
  whose stored MOC does not cover it.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from mocpy import MOC

from . import fits_lite, naming
from .query import INDEX_FRAME, bounding_box, to_index_frame
from .s3io import S3Object, S3Reader
from .store import FileRow, Store

#: bytes per Range-GET when a whole table is read back (validate)
FULL_READ_CHUNK = 8 << 20

#: files buffered before an index write; bounds redone work if a crawl is killed
WRITE_BATCH = 250


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
    #: indexed, but their header did not parse (name-derived coverage still exact)
    header_errors: int = 0
    bytes_read: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class FileResult:
    row: FileRow
    cards: tuple[tuple[int, str, str, str], ...]


Progress = Callable[[CrawlStats, float], None]

DEFAULT_OPTIONS = CrawlOptions()


def build_moc(lon: np.ndarray, lat: np.ndarray, order: int, dilate: int) -> MOC:
    """MOC of positions (already in the index frame), dilated by ``dilate`` borders."""
    moc = MOC.from_lonlat(lon * u.deg, lat * u.deg, max_norder=order)
    for _ in range(max(0, dilate)):
        moc.add_neighbours()
    return moc


def coord_bbox(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float, float, float, bool]:
    """Smallest lon/lat box around the points; ``wraps`` when it crosses lon=0.

    The longitude interval is the complement of the largest angular gap between
    consecutive sorted longitudes.
    """
    lons = np.sort(np.asarray(lon, dtype=float) % 360.0)
    lat_min = float(np.min(lat))
    lat_max = float(np.max(lat))
    if lons.size == 1:
        return float(lons[0]), float(lons[0]), lat_min, lat_max, False
    gaps = np.diff(lons)
    wrap_gap = lons[0] + 360.0 - lons[-1]
    if gaps.size == 0 or wrap_gap >= float(np.max(gaps)):
        return float(lons[0]), float(lons[-1]), lat_min, lat_max, False
    i = int(np.argmax(gaps))
    return float(lons[i + 1]), float(lons[i]), lat_min, lat_max, True


def _clean_coords(rows: np.ndarray, ra_col: str, dec_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Finite ICRS ra/dec from a row buffer, converted into the index frame."""
    ra = np.asarray(rows[ra_col], dtype=float).ravel()
    dec = np.asarray(rows[dec_col], dtype=float).ravel()
    good = np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90.0)
    if not good.any():
        return ra[good], dec[good]
    return to_index_frame(ra[good], dec[good], "icrs")


def _header_metadata(
    reader: S3Reader, obj: S3Object
) -> tuple[int, int, int, tuple[tuple[int, str, str, str], ...], str | None]:
    """Read only the header blocks for nrows/row_bytes/data_offset and the cards.

    ~8.6 KB per file. A header that will not parse is reported but does not
    invalidate name-derived coverage, which needs no bytes at all — so the file
    is still indexed, with zeroed row metadata.
    """
    try:
        info = fits_lite.read_bintable_info(reader.fetcher(obj.uri))
    except Exception as exc:  # coverage does not depend on the header
        return 0, 0, 0, (), f"{type(exc).__name__}: {exc}"
    cards = tuple((hdu, c.key, c.value, c.comment) for hdu, c in info.cards)
    return info.nrows, info.row_bytes, info.data_offset, cards, None


#: a coverage builder: returns the row plus an optional non-fatal warning
Builder = Callable[[S3Reader, S3Object, "CrawlOptions"], tuple[FileResult, str | None]]


def index_file_from_name(
    reader: S3Reader, obj: S3Object, opts: CrawlOptions
) -> tuple[FileResult, str | None]:
    """Derive coverage from the filename's HEALPix pixel. No row reads.

    Exact by construction: the file's footprint *is* the named cell, so there is
    nothing to approximate and no dilation to apply.
    """
    moc = naming.moc_for_uri(obj.uri, opts.order)
    box = bounding_box(moc)
    nrows, row_bytes, data_offset, cards, header_error = _header_metadata(reader, obj)
    row = FileRow(
        uri=obj.uri,
        etag=obj.etag,
        size=obj.size,
        nrows=nrows,
        row_bytes=row_bytes,
        data_offset=data_offset,
        lon_min=box.lon_min,
        lon_max=box.lon_max,
        lat_min=box.lat_min,
        lat_max=box.lat_max,
        lon_wraps=box.wraps,
        moc_json=moc.to_string(format="json"),
    )
    return FileResult(row=row, cards=cards), header_error


def _index_file_sampled(
    reader: S3Reader, obj: S3Object, opts: CrawlOptions
) -> tuple[FileResult, str | None]:
    """:func:`index_file` in :data:`Builder` shape. Sampling has no partial success."""
    return index_file(reader, obj, opts), None


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
    lon_min, lon_max, lat_min, lat_max, wraps = coord_bbox(ra_all, dec_all)
    row = FileRow(
        uri=obj.uri,
        etag=obj.etag,
        size=obj.size,
        nrows=info.nrows,
        row_bytes=info.row_bytes,
        data_offset=info.data_offset,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_wraps=wraps,
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
    from_names: bool = True,
) -> CrawlStats:
    """Index every FITS file under ``prefix``. Unchanged files (same ETag) are skipped.

    ``from_names`` derives exact coverage from each filename's HEALPix pixel and
    reads only headers; otherwise coverage comes from sampled rows.

    Per-file failures are logged to ``crawl_errors`` and never abort the crawl. A
    filename that does not encode a pixel is a failure rather than a silent
    fallback, so a break in the naming convention surfaces instead of producing
    a wrong or missing footprint.
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

    def note_error(uri: str, message: str) -> None:
        stats.errors.append((uri, message))
        store.record_error(uri, message)

    builder: Builder = index_file_from_name if from_names else _index_file_sampled
    # Writes go out in batches: one transaction per file costs far more than the
    # indexing itself (see Store.upsert_files). A partial batch is simply redone
    # on the next run, since resume keys off the stored ETag.
    pending: list[tuple[FileRow, Sequence[tuple[int, str, str, str]]]] = []

    def flush() -> None:
        if pending:
            store.upsert_files(pending)
            pending.clear()

    report()
    with ThreadPoolExecutor(max_workers=max(1, opts.workers)) as pool:
        futures = {pool.submit(builder, reader, obj, opts): obj for obj in todo}
        for future in as_completed(futures):
            obj = futures[future]
            try:
                result, warning = future.result()
            except Exception as exc:
                stats.failed += 1
                note_error(obj.uri, f"{type(exc).__name__}: {exc}")
            else:
                if warning is not None:
                    stats.header_errors += 1
                    note_error(obj.uri, f"header unreadable (coverage kept): {warning}")
                pending.append((result.row, result.cards))
                if len(pending) >= WRITE_BATCH:
                    flush()
                stats.indexed += 1
            report()
    flush()
    store.set_meta("prefix", prefix)
    store.set_meta("order", str(opts.order))
    store.set_meta("coverage", "filename" if from_names else "sampled")
    store.set_meta("moc_frame", INDEX_FRAME)
    store.set_meta("dilate", "0" if from_names else str(opts.dilate))
    store.set_meta("samples", "" if from_names else str(opts.samples))
    store.set_meta("sample_rows", "" if from_names else str(opts.sample_rows))
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
