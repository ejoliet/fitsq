from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import astropy.units as u
import boto3
import numpy as np
import pytest
from mocpy import MOC
from moto import mock_aws

import synth
from fitsq.indexer import CrawlOptions, crawl, full_moc, index_file, uncovered, validate
from fitsq.naming import cell_moc
from fitsq.query import INDEX_FRAME, Index, cone_moc
from fitsq.s3io import S3Object, S3Reader
from fitsq.store import Store

BUCKET = "stpubdata"
PREFIX = f"s3://{BUCKET}/roman/nexus/input_catalogs/"
ORDER = 9

#: Galactic order-9 pixels, named the way the real bucket names them. 1113533 is
#: the real pixel of the README's test position (ICRS 268.77, -29.25).
PIXELS = (1113533, 1157294, 500000)
TILES = {f"cat-{pixel}.fits": pixel for pixel in PIXELS}


@pytest.fixture
def s3_reader() -> Iterator[S3Reader]:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        for name, pixel in TILES.items():
            ra, dec = synth.cell_patch(pixel, n=500, seed=pixel % 1000)
            client.put_object(
                Bucket=BUCKET,
                Key=f"roman/nexus/input_catalogs/{name}",
                Body=synth.catalog(ra, dec),
            )
        client.put_object(
            Bucket=BUCKET, Key="roman/nexus/input_catalogs/notes.txt", Body=b"ignore me"
        )
        yield S3Reader(client)


cell_centre_icrs = synth.cell_centre_icrs


def test_list_fits_filters_non_fits(s3_reader: S3Reader) -> None:
    uris = sorted(obj.uri for obj in s3_reader.list_fits(PREFIX))
    assert uris == sorted(f"{PREFIX}{name}" for name in TILES)
    assert all(obj.etag and obj.size > 0 for obj in s3_reader.list_fits(PREFIX))


def test_ranged_get_tracks_bytes(s3_reader: S3Reader) -> None:
    uri = f"{PREFIX}cat-{PIXELS[1]}.fits"
    head = s3_reader.ranged_get(uri, 0, 2880)
    assert head.startswith(b"SIMPLE")
    assert s3_reader.bytes_read == 2880
    assert s3_reader.ranged_get(uri, 0, 0) == b""


def test_index_file_geometry(s3_reader: S3Reader) -> None:
    """Sampling path: bbox is in the index frame, around the tile's cell."""
    obj = next(o for o in s3_reader.list_fits(PREFIX) if o.uri.endswith(f"cat-{PIXELS[0]}.fits"))
    result = index_file(s3_reader, obj, CrawlOptions(sample_rows=100, samples=3))
    row = result.row
    assert row.nrows == 500 and row.row_bytes == 91
    centre = cell_moc(PIXELS[0], ORDER).barycenter().spherical
    assert row.lon_min < float(centre.lon.deg) < row.lon_max
    assert row.lat_min < float(centre.lat.deg) < row.lat_max
    assert not row.lon_wraps
    assert row.moc_json.strip().startswith("{")
    keys = {key for _, key, _, _ in result.cards}
    assert {"SIMPLE", "XTENSION", "NAXIS1", "TFORM1"} <= keys


def test_crawl_indexes_and_is_resumable(tmp_path: Path, s3_reader: S3Reader) -> None:
    opts = CrawlOptions(workers=4, sample_rows=200, samples=3, order=ORDER, dilate=1)
    seen: list[tuple[int, int]] = []
    with Store(tmp_path / "i.duckdb") as store:
        stats = crawl(
            PREFIX,
            store,
            s3_reader,
            opts,
            lambda s, _e: seen.append((s.indexed, s.failed)),
            from_names=False,
        )
        assert (stats.total, stats.indexed, stats.failed, stats.skipped) == (3, 3, 0, 0)
        assert seen[-1] == (3, 0)
        assert store.get_meta("prefix") == PREFIX
        assert store.get_meta("order") == str(ORDER)
        assert store.get_meta("coverage") == "sampled"
        assert store.get_meta("moc_frame") == INDEX_FRAME
        assert store.get_meta("last_crawl") is not None
        first_bytes = s3_reader.bytes_read

        # rerun: nothing changed -> everything skipped, no S3 data reads
        again = crawl(PREFIX, store, s3_reader, opts, from_names=False)
        assert (again.indexed, again.skipped) == (0, 3)
        assert s3_reader.bytes_read == first_bytes
        assert len(store.file_rows()) == 3


def test_crawl_reindexes_changed_etag(tmp_path: Path, s3_reader: S3Reader) -> None:
    opts = CrawlOptions(workers=2, sample_rows=200)
    changed = PIXELS[1]
    with Store(tmp_path / "i.duckdb") as store:
        crawl(PREFIX, store, s3_reader, opts, from_names=False)
        ra, dec = synth.cell_patch(changed, n=400, seed=42)
        s3_reader.client.put_object(
            Bucket=BUCKET,
            Key=f"roman/nexus/input_catalogs/cat-{changed}.fits",
            Body=synth.catalog(ra, dec),
        )
        stats = crawl(PREFIX, store, s3_reader, opts, from_names=False)
        assert (stats.indexed, stats.skipped) == (1, 2)
        row = next(r for r in store.file_rows() if r.uri.endswith(f"cat-{changed}.fits"))
        assert row.nrows == 400


def test_crawl_survives_bad_files(tmp_path: Path, s3_reader: S3Reader) -> None:
    s3_reader.client.put_object(
        Bucket=BUCKET, Key="roman/nexus/input_catalogs/junk.fits", Body=b"not a fits file" * 500
    )
    image = synth.primary_header() + synth.header_block(
        [synth.card("XTENSION", "IMAGE"), synth.card("BITPIX", 8), synth.card("NAXIS", 0)]
    )
    s3_reader.client.put_object(
        Bucket=BUCKET, Key="roman/nexus/input_catalogs/image.fits", Body=image
    )
    s3_reader.client.put_object(
        Bucket=BUCKET,
        Key="roman/nexus/input_catalogs/empty.fits",
        Body=synth.primary_header() + synth.bintable_header(0),
    )
    with Store(tmp_path / "i.duckdb") as store:
        stats = crawl(
            PREFIX, store, s3_reader, CrawlOptions(workers=4, sample_rows=200), from_names=False
        )
        assert stats.indexed == 3 and stats.failed == 3
        names, rows = store.sql("SELECT uri, error FROM crawl_errors ORDER BY uri")
        assert names == ["uri", "error"] and len(rows) == 3
        assert all("FitsError" in error or "Error" in error for _, error in rows)
        assert store.status()["errors"] == 3


@pytest.mark.parametrize("from_names", [True, False], ids=["from-names", "sampled"])
def test_indexed_files_answer_cone_queries(
    tmp_path: Path, s3_reader: S3Reader, from_names: bool
) -> None:
    """Both coverage modes must answer the same cone query identically."""
    ra, dec = cell_centre_icrs(PIXELS[0])
    with Store(tmp_path / "i.duckdb") as store:
        crawl(
            PREFIX,
            store,
            s3_reader,
            CrawlOptions(workers=2, sample_rows=200, order=ORDER),
            from_names=from_names,
        )
        index = Index(store)
        hits = index.search(cone_moc(ra, dec, 30 * u.arcsec, ORDER))
        assert [r.uri for r in hits] == [f"{PREFIX}cat-{PIXELS[0]}.fits"]
        assert index.search(cone_moc(ra - 10.0, dec, 30 * u.arcsec, ORDER)) == []


def test_from_names_reads_no_row_data(tmp_path: Path, s3_reader: S3Reader) -> None:
    """Name-derived coverage costs headers only, and is exact (== the named cell)."""
    with Store(tmp_path / "i.duckdb") as store:
        stats = crawl(PREFIX, store, s3_reader, CrawlOptions(workers=2), from_names=True)
        assert (stats.indexed, stats.failed) == (3, 0)
        assert store.get_meta("coverage") == "filename"
        assert store.get_meta("dilate") == "0"
        # headers only: three tiles of 500 rows would be ~136 KB of row data
        assert stats.bytes_read < 3 * 40_000
        for row in store.file_rows():
            pixel = int(row.uri.rsplit("cat-", 1)[1].removesuffix(".fits"))
            stored = MOC.from_string(row.moc_json, format="json")
            assert stored == cell_moc(pixel, ORDER)
            assert row.nrows == 500  # header still gives the row count


def test_from_names_rejects_unparseable_name(tmp_path: Path, s3_reader: S3Reader) -> None:
    """A file that breaks the naming convention fails loudly, never silently."""
    ra, dec = synth.cell_patch(PIXELS[0], n=50, seed=7)
    s3_reader.client.put_object(
        Bucket=BUCKET,
        Key="roman/nexus/input_catalogs/mystery-tile.fits",
        Body=synth.catalog(ra, dec),
    )
    with Store(tmp_path / "i.duckdb") as store:
        stats = crawl(PREFIX, store, s3_reader, CrawlOptions(workers=2), from_names=True)
        assert (stats.indexed, stats.failed) == (3, 1)
        assert not any(r.uri.endswith("mystery-tile.fits") for r in store.file_rows())
        _, errors = store.sql("SELECT uri, error FROM crawl_errors")
        assert len(errors) == 1
        assert "mystery-tile.fits" in errors[0][0] and "NamingError" in errors[0][1]


def test_from_names_keeps_coverage_when_header_unreadable(
    tmp_path: Path, s3_reader: S3Reader
) -> None:
    """Coverage needs no bytes, so a corrupt header must not lose the file."""
    s3_reader.client.put_object(
        Bucket=BUCKET,
        Key=f"roman/nexus/input_catalogs/cat-{PIXELS[2] + 1}.fits",
        Body=b"not a fits file" * 500,
    )
    with Store(tmp_path / "i.duckdb") as store:
        stats = crawl(PREFIX, store, s3_reader, CrawlOptions(workers=2), from_names=True)
        assert (stats.indexed, stats.failed, stats.header_errors) == (4, 0, 1)
        row = next(r for r in store.file_rows() if r.uri.endswith(f"cat-{PIXELS[2] + 1}.fits"))
        assert row.nrows == 0
        assert MOC.from_string(row.moc_json, format="json") == cell_moc(PIXELS[2] + 1, ORDER)


def test_validate_passes_for_both_modes(tmp_path: Path, s3_reader: S3Reader) -> None:
    """The guard must accept name-derived and sampled coverage alike."""
    for from_names in (True, False):
        with Store(tmp_path / f"i-{from_names}.duckdb") as store:
            crawl(
                PREFIX,
                store,
                s3_reader,
                CrawlOptions(workers=2, sample_rows=50, samples=3, order=ORDER, dilate=1),
                from_names=from_names,
            )
            rows = store.file_rows()
        assert not full_moc(s3_reader, rows[0], ORDER).empty()
        assert validate(rows, s3_reader, ORDER, n=3, rng=random.Random(0)) == []


def test_validate_catches_a_broken_naming_convention(tmp_path: Path, s3_reader: S3Reader) -> None:
    """The point of keeping the sampling path: a lying filename must be caught.

    The file is named for one cell but its rows live in another, exactly what a
    future change to the producer's naming would look like.
    """
    liar = f"cat-{PIXELS[2]}.fits"
    ra, dec = synth.cell_patch(PIXELS[0], n=200, seed=99)  # rows of a *different* cell
    s3_reader.client.put_object(
        Bucket=BUCKET, Key=f"roman/nexus/input_catalogs/{liar}", Body=synth.catalog(ra, dec)
    )
    with Store(tmp_path / "i.duckdb") as store:
        crawl(PREFIX, store, s3_reader, CrawlOptions(workers=2), from_names=True)
        rows = [r for r in store.file_rows() if r.uri.endswith(liar)]
    violations = validate(rows, s3_reader, ORDER, n=1, rng=random.Random(0))
    assert len(violations) == 1
    assert violations[0].uri.endswith(liar) and violations[0].missing_cells > 0


def test_validate_detects_under_coverage(tmp_path: Path, s3_reader: S3Reader) -> None:
    """A deliberately truncated MOC must be reported as a violation."""
    with Store(tmp_path / "i.duckdb") as store:
        crawl(
            PREFIX,
            store,
            s3_reader,
            CrawlOptions(workers=2, sample_rows=200, order=ORDER),
            from_names=False,
        )
        rows = store.file_rows()
    target = rows[0]
    tiny = cone_moc(0.0, 89.999, 1 * u.arcmin, ORDER)
    broken = [
        type(target)(**{**target.__dict__, "moc_json": tiny.to_string(format="json")}),
    ]
    violations = validate(broken, s3_reader, ORDER, n=1, rng=random.Random(0))
    assert len(violations) == 1
    assert violations[0].uri == target.uri and violations[0].missing_cells > 0


def test_index_file_sampling_windows_cover_whole_small_table(s3_reader: S3Reader) -> None:
    """Small tables are read whole, so sampled == true coverage."""
    obj = next(o for o in s3_reader.list_fits(PREFIX) if o.uri.endswith(f"cat-{PIXELS[1]}.fits"))
    result = index_file(s3_reader, obj, CrawlOptions(sample_rows=50_000, samples=3, dilate=0))
    sampled = MOC.from_string(result.row.moc_json, format="json")
    assert uncovered(full_moc(s3_reader, result.row, ORDER), sampled).empty()


def test_index_file_rejects_all_nan_coords(s3_reader: S3Reader) -> None:
    nan = np.full(10, np.nan)
    s3_reader.client.put_object(
        Bucket=BUCKET,
        Key="roman/nexus/input_catalogs/nan.fits",
        Body=synth.catalog(nan, nan),
    )
    obj = S3Object(uri=f"{PREFIX}nan.fits", size=1, etag="x")
    with pytest.raises(Exception, match="finite"):
        index_file(s3_reader, obj, CrawlOptions(sample_rows=100))
