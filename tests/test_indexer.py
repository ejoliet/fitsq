from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import astropy.units as u
import boto3
import numpy as np
import pytest
from moto import mock_aws

import synth
from fitsq.indexer import CrawlOptions, crawl, full_moc, index_file, uncovered, validate
from fitsq.query import Index, cone_moc
from fitsq.s3io import S3Object, S3Reader
from fitsq.store import Store

BUCKET = "stpubdata"
PREFIX = f"s3://{BUCKET}/roman/nexus/input_catalogs/"
ORDER = 9

TILES = {
    "cat-1113533.fits": (268.77, -29.25),
    "cat-2.fits": (10.0, 10.0),
    "cat-3.fits": (359.95, 0.0),
}


@pytest.fixture
def s3_reader() -> Iterator[S3Reader]:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        for name, (ra_c, dec_c) in TILES.items():
            ra, dec = synth.patch(ra_c, dec_c, n=500, seed=abs(hash(name)) % 1000)
            client.put_object(
                Bucket=BUCKET,
                Key=f"roman/nexus/input_catalogs/{name}",
                Body=synth.catalog(ra, dec),
            )
        client.put_object(
            Bucket=BUCKET, Key="roman/nexus/input_catalogs/notes.txt", Body=b"ignore me"
        )
        yield S3Reader(client)


def test_list_fits_filters_non_fits(s3_reader: S3Reader) -> None:
    uris = sorted(obj.uri for obj in s3_reader.list_fits(PREFIX))
    assert uris == sorted(f"{PREFIX}{name}" for name in TILES)
    assert all(obj.etag and obj.size > 0 for obj in s3_reader.list_fits(PREFIX))


def test_ranged_get_tracks_bytes(s3_reader: S3Reader) -> None:
    uri = f"{PREFIX}cat-2.fits"
    head = s3_reader.ranged_get(uri, 0, 2880)
    assert head.startswith(b"SIMPLE")
    assert s3_reader.bytes_read == 2880
    assert s3_reader.ranged_get(uri, 0, 0) == b""


def test_index_file_geometry(s3_reader: S3Reader) -> None:
    obj = next(o for o in s3_reader.list_fits(PREFIX) if o.uri.endswith("cat-1113533.fits"))
    result = index_file(s3_reader, obj, CrawlOptions(sample_rows=100, samples=3))
    row = result.row
    assert row.nrows == 500 and row.row_bytes == 91
    assert 268.6 < row.ra_min < 268.95 and -29.35 < row.dec_min < -29.15
    assert not row.ra_wraps
    assert row.moc_json.strip().startswith("{")
    keys = {key for _, key, _, _ in result.cards}
    assert {"SIMPLE", "XTENSION", "NAXIS1", "TFORM1"} <= keys


def test_crawl_indexes_and_is_resumable(tmp_path: Path, s3_reader: S3Reader) -> None:
    opts = CrawlOptions(workers=4, sample_rows=200, samples=3, order=ORDER, dilate=1)
    seen: list[tuple[int, int]] = []
    with Store(tmp_path / "i.duckdb") as store:
        stats = crawl(
            PREFIX, store, s3_reader, opts, lambda s, _e: seen.append((s.indexed, s.failed))
        )
        assert (stats.total, stats.indexed, stats.failed, stats.skipped) == (3, 3, 0, 0)
        assert seen[-1] == (3, 0)
        assert store.get_meta("prefix") == PREFIX
        assert store.get_meta("order") == str(ORDER)
        assert store.get_meta("last_crawl") is not None
        first_bytes = s3_reader.bytes_read

        # rerun: nothing changed -> everything skipped, no S3 data reads
        again = crawl(PREFIX, store, s3_reader, opts)
        assert (again.indexed, again.skipped) == (0, 3)
        assert s3_reader.bytes_read == first_bytes
        assert len(store.file_rows()) == 3


def test_crawl_reindexes_changed_etag(tmp_path: Path, s3_reader: S3Reader) -> None:
    opts = CrawlOptions(workers=2, sample_rows=200)
    with Store(tmp_path / "i.duckdb") as store:
        crawl(PREFIX, store, s3_reader, opts)
        ra, dec = synth.patch(200.0, -10.0, n=500, seed=42)
        s3_reader.client.put_object(
            Bucket=BUCKET,
            Key="roman/nexus/input_catalogs/cat-2.fits",
            Body=synth.catalog(ra, dec),
        )
        stats = crawl(PREFIX, store, s3_reader, opts)
        assert (stats.indexed, stats.skipped) == (1, 2)
        row = next(r for r in store.file_rows() if r.uri.endswith("cat-2.fits"))
        assert 199.0 < row.ra_min < 201.0


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
        stats = crawl(PREFIX, store, s3_reader, CrawlOptions(workers=4, sample_rows=200))
        assert stats.indexed == 3 and stats.failed == 3
        names, rows = store.sql("SELECT uri, error FROM crawl_errors ORDER BY uri")
        assert names == ["uri", "error"] and len(rows) == 3
        assert all("FitsError" in error or "Error" in error for _, error in rows)
        assert store.status()["errors"] == 3


def test_indexed_files_answer_cone_queries(tmp_path: Path, s3_reader: S3Reader) -> None:
    with Store(tmp_path / "i.duckdb") as store:
        crawl(PREFIX, store, s3_reader, CrawlOptions(workers=2, sample_rows=200, order=ORDER))
        index = Index(store)
        hits = index.search(cone_moc(268.77, -29.25, 30 * u.arcsec, ORDER))
        assert [r.uri for r in hits] == [f"{PREFIX}cat-1113533.fits"]
        assert index.search(cone_moc(268.77 - 10.0, -29.25, 30 * u.arcsec, ORDER)) == []
        # tile straddling RA=0 is found from either side
        assert index.search(cone_moc(0.0, 0.0, 0.2 * u.deg, ORDER))


def test_full_moc_and_validate_pass(tmp_path: Path, s3_reader: S3Reader) -> None:
    """Sampling only a fraction of rows must still cover every row (dilation)."""
    with Store(tmp_path / "i.duckdb") as store:
        crawl(
            PREFIX,
            store,
            s3_reader,
            CrawlOptions(workers=2, sample_rows=50, samples=3, order=ORDER, dilate=1),
        )
        rows = store.file_rows()
    truth = full_moc(s3_reader, rows[0], ORDER)
    assert not truth.empty()
    assert validate(rows, s3_reader, ORDER, n=3, rng=random.Random(0)) == []


def test_validate_detects_under_coverage(tmp_path: Path, s3_reader: S3Reader) -> None:
    """A deliberately truncated MOC must be reported as a violation."""
    with Store(tmp_path / "i.duckdb") as store:
        crawl(PREFIX, store, s3_reader, CrawlOptions(workers=2, sample_rows=200, order=ORDER))
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
    """Small tables are read whole (Open Question 3), so sampled == true coverage."""
    obj = next(o for o in s3_reader.list_fits(PREFIX) if o.uri.endswith("cat-2.fits"))
    result = index_file(s3_reader, obj, CrawlOptions(sample_rows=50_000, samples=3, dilate=0))
    from mocpy import MOC

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
