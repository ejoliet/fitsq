"""Opt-in smoke tests against the real public bucket. Never run by default.

    FITSQ_E2E=1 uv run pytest tests/test_e2e.py -s              # light: ~14 MB read
    FITSQ_E2E=1 uv run pytest tests/test_e2e.py -s -m slow      # adds a ~535 MB full read

Indexes one real file (not the whole prefix) and checks the known cone hit from
the README. The validate gate is a separate test because it reads a whole file.
"""

from __future__ import annotations

import os
from pathlib import Path

import astropy.units as u
import pytest

from fitsq.indexer import CrawlOptions, index_file, validate
from fitsq.query import Index, cone_moc
from fitsq.s3io import S3Reader
from fitsq.store import Store

PREFIX = "s3://stpubdata/roman/nexus/soc_simulations/input_catalogs/"
KNOWN_FILE = "cat-1113533.fits"
KNOWN_CONE = (268.77, -29.25, 30 * u.arcsec)
ORDER = 9

pytestmark = pytest.mark.skipif(
    os.environ.get("FITSQ_E2E") != "1", reason="set FITSQ_E2E=1 to hit the real bucket"
)


def index_known_file(store: Store) -> S3Reader:
    reader = S3Reader(anon=True)
    objects = [obj for obj in reader.list_fits(PREFIX) if obj.uri.endswith(KNOWN_FILE)]
    assert objects, f"{KNOWN_FILE} not found under {PREFIX}"
    result = index_file(reader, objects[0], CrawlOptions(order=ORDER, dilate=1))
    store.upsert_file(result.row, result.cards)
    row = result.row
    print(f"\n{KNOWN_FILE}: nrows={row.nrows} row_bytes={row.row_bytes} size={row.size}")
    print(
        f"  bbox ra [{row.ra_min:.4f}, {row.ra_max:.4f}] "
        f"dec [{row.dec_min:.4f}, {row.dec_max:.4f}] wraps={row.ra_wraps}"
    )
    print(f"  sampled {reader.bytes_read / 1e6:.1f} MB")
    assert row.row_bytes == 91, "ground truth: NAXIS1 = 91"
    assert row.nrows > 1_000_000
    return reader


def test_real_bucket_cone_finds_known_file(tmp_path: Path) -> None:
    with Store(tmp_path / "e2e.duckdb") as store:
        index_known_file(store)
        index = Index(store)
        ra, dec, radius = KNOWN_CONE
        hits = index.search(cone_moc(ra, dec, radius, ORDER))
        assert [Path(r.uri).name for r in hits] == [KNOWN_FILE]
        # 10 degrees away in empty sky must return nothing
        assert index.search(cone_moc(ra - 10.0, dec, radius, ORDER)) == []


@pytest.mark.slow
def test_real_bucket_validate_gate(tmp_path: Path) -> None:
    """Sampling adequacy on real data: full read of one file must be covered."""
    with Store(tmp_path / "e2e.duckdb") as store:
        reader = index_known_file(store)
        assert validate(store.file_rows(), reader, ORDER, n=1) == []
        print(f"  total read incl. full scan {reader.bytes_read / 1e6:.1f} MB")
