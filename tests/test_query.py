from __future__ import annotations

import random
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from mocpy import MOC

import synth
from fitsq.indexer import build_moc, coord_bbox, uncovered
from fitsq.query import (
    BBox,
    Index,
    QueryError,
    bounding_box,
    cone_moc,
    parse_angle,
    parse_stcs,
    polygon_moc,
)
from fitsq.store import FileRow, Store

ORDER = 9


def add_file(store: Store, uri: str, ra: np.ndarray, dec: np.ndarray, dilate: int = 1) -> None:
    moc = build_moc(ra, dec, ORDER, dilate)
    ra_min, ra_max, dec_min, dec_max, wraps = coord_bbox(ra, dec)
    store.upsert_file(
        FileRow(
            uri=uri,
            etag="e",
            size=len(ra) * 91,
            nrows=len(ra),
            row_bytes=91,
            data_offset=5760,
            ra_min=ra_min,
            ra_max=ra_max,
            dec_min=dec_min,
            dec_max=dec_max,
            ra_wraps=wraps,
            moc_json=moc.to_string(format="json"),
        )
    )


def test_parse_angle_forms() -> None:
    assert parse_angle("30arcsec").to_value(u.arcsec) == pytest.approx(30)
    assert parse_angle("2arcmin").to_value(u.arcmin) == pytest.approx(2)
    assert parse_angle("0.5deg").to_value(u.deg) == pytest.approx(0.5)
    assert parse_angle("0.05", "deg").to_value(u.deg) == pytest.approx(0.05)
    assert parse_angle("1", "arcmin").to_value(u.arcmin) == pytest.approx(1)
    for bad in ("", "abc", "5parsec", "-1deg", "0deg"):
        with pytest.raises(QueryError):
            parse_angle(bad)


def test_parse_stcs_polygon_and_circle() -> None:
    poly = parse_stcs("POLYGON ICRS 268.7 -29.3 268.9 -29.3 268.9 -29.1 268.7 -29.1", ORDER)
    expected = polygon_moc([268.7, 268.9, 268.9, 268.7], [-29.3, -29.3, -29.1, -29.1], ORDER)
    assert poly == expected
    circle = parse_stcs("Circle ICRS 268.77 -29.25 0.05", ORDER)
    assert circle == cone_moc(268.77, -29.25, 0.05 * u.deg, ORDER)


def test_parse_stcs_rejects_unsupported() -> None:
    with pytest.raises(QueryError, match="v1 supports"):
        parse_stcs("UNION ICRS (Circle 1 2 3)", ORDER)
    with pytest.raises(QueryError, match="frame"):
        parse_stcs("POLYGON 1 2 3 4 5 6", ORDER)
    with pytest.raises(QueryError, match="3 numbers"):
        parse_stcs("CIRCLE ICRS 1 2", ORDER)
    with pytest.raises(QueryError, match="pairs"):
        parse_stcs("POLYGON ICRS 1 2 3 4", ORDER)
    with pytest.raises(QueryError, match="non-numeric"):
        parse_stcs("CIRCLE ICRS a b c", ORDER)
    with pytest.raises(QueryError, match="empty"):
        parse_stcs("   ", ORDER)


def test_coord_bbox_plain_and_wrapping() -> None:
    ra = np.array([10.0, 11.0, 12.0])
    dec = np.array([-1.0, 0.0, 1.0])
    assert coord_bbox(ra, dec) == (10.0, 12.0, -1.0, 1.0, False)
    ra_wrap = np.array([359.5, 0.5, 359.9])
    lo, hi, _, _, wraps = coord_bbox(ra_wrap, dec)
    assert wraps and lo == pytest.approx(359.5) and hi == pytest.approx(0.5)
    single = coord_bbox(np.array([5.0]), np.array([2.0]))
    assert single == (5.0, 5.0, 2.0, 2.0, False)


def test_bounding_box_is_superset_of_moc() -> None:
    moc = cone_moc(268.77, -29.25, 0.5 * u.deg, ORDER)
    box = bounding_box(moc)
    assert box.dec_min < -29.75 and box.dec_max > -28.75
    assert not box.ra_unbounded and not box.wraps
    assert box.ra_min < 268.77 < box.ra_max


def test_bounding_box_flags_wrap_and_poles() -> None:
    assert bounding_box(cone_moc(0.05, 0.0, 0.5 * u.deg, ORDER)).wraps
    assert bounding_box(cone_moc(10.0, 89.9, 0.5 * u.deg, ORDER)).ra_unbounded
    wide = bounding_box(cone_moc(30.0, 0.0, 60.0 * u.deg, ORDER))
    # A 60 deg cone at the equator spans RA 330..90; the box must contain that.
    assert wide.wraps and wide.ra_min < 330.0 and wide.ra_max > 90.0
    assert wide.dec_min < -60.0 and wide.dec_max > 60.0


def test_bounding_box_ra_halfwidth_never_under_covers_a_cone() -> None:
    """The prefilter box must never clip a cone: RA half-width >= exact value."""
    for dec_center in (0.0, 15.0, -30.0, 45.0, -60.0, 75.0, 88.0, -88.5):
        for radius in (0.01, 0.1, 0.5, 2.0, 10.0, 30.0):
            box = bounding_box(cone_moc(123.0, dec_center, radius * u.deg, ORDER))
            assert box.dec_min <= max(dec_center - radius, -90.0)
            assert box.dec_max >= min(dec_center + radius, 90.0)
            if box.ra_unbounded:
                continue
            half_width = ((box.ra_max - box.ra_min) % 360.0) / 2.0
            denominator = np.cos(np.radians(dec_center))
            ratio = np.sin(np.radians(radius)) / denominator
            exact = 180.0 if ratio >= 1.0 else np.degrees(np.arcsin(ratio))
            assert half_width >= exact, (dec_center, radius, half_width, exact)


def test_search_hits_and_misses(tmp_path: Path) -> None:
    with Store(tmp_path / "i.duckdb") as store:
        add_file(store, "s3://b/near.fits", *synth.patch(268.77, -29.25, seed=1))
        add_file(store, "s3://b/far.fits", *synth.patch(100.0, 10.0, seed=2))
        index = Index(store)
        hits = index.search(cone_moc(268.77, -29.25, 30 * u.arcsec, ORDER))
        assert [r.uri for r in hits] == ["s3://b/near.fits"]
        assert index.search(cone_moc(258.77, -29.25, 30 * u.arcsec, ORDER)) == []
        # MOC cache is reused across queries
        assert list(index._mocs) == ["s3://b/near.fits"]


def test_search_across_ra_zero(tmp_path: Path) -> None:
    with Store(tmp_path / "i.duckdb") as store:
        add_file(store, "s3://b/west.fits", *synth.patch(359.95, 0.0, seed=3))
        add_file(store, "s3://b/east.fits", *synth.patch(0.05, 0.0, seed=4))
        add_file(store, "s3://b/other.fits", *synth.patch(180.0, 0.0, seed=5))
        index = Index(store)
        hits = index.search(cone_moc(0.0, 0.0, 0.3 * u.deg, ORDER))
        assert [r.uri for r in hits] == ["s3://b/east.fits", "s3://b/west.fits"]


def test_search_at_pole(tmp_path: Path) -> None:
    with Store(tmp_path / "i.duckdb") as store:
        add_file(store, "s3://b/pole.fits", *synth.patch(0.0, 89.95, span=0.05, seed=6))
        add_file(store, "s3://b/equator.fits", *synth.patch(0.0, 0.0, seed=7))
        index = Index(store)
        hits = index.search(cone_moc(180.0, 89.99, 0.2 * u.deg, ORDER))
        assert [r.uri for r in hits] == ["s3://b/pole.fits"]


def test_region_polygon_search(tmp_path: Path) -> None:
    with Store(tmp_path / "i.duckdb") as store:
        add_file(store, "s3://b/in.fits", *synth.patch(268.8, -29.2, seed=8))
        add_file(store, "s3://b/out.fits", *synth.patch(120.0, 40.0, seed=9))
        index = Index(store)
        moc = parse_stcs("POLYGON ICRS 268.7 -29.3 268.9 -29.3 268.9 -29.1 268.7 -29.1", ORDER)
        assert [r.uri for r in index.search(moc)] == ["s3://b/in.fits"]


def test_candidates_prefilter_narrows_but_keeps_hits(tmp_path: Path) -> None:
    with Store(tmp_path / "i.duckdb") as store:
        for i in range(5):
            add_file(store, f"s3://b/{i}.fits", *synth.patch(10.0 * i, 5.0 * i, seed=i))
        index = Index(store)
        moc = cone_moc(20.0, 10.0, 0.2 * u.deg, ORDER)
        candidates = index.candidates(moc)
        assert 0 < len(candidates) < 5
        assert [r.uri for r in index.search(moc)] == ["s3://b/2.fits"]


def test_where_clause_skips_ra_filter_when_unbounded() -> None:
    box = BBox(0.0, 360.0, -10.0, 10.0, False, True)
    from fitsq.query import _where

    where, params = _where(box)
    assert "ra_" not in where and params == [-10.0, 10.0]


def test_property_random_cones_vs_brute_force(tmp_path: Path) -> None:
    """Index must never under-return; over-return only from cell quantisation.

    Margins absorb HEALPix cell quantisation (order 9 cells are ~6.9'), so the
    thresholds are 0.85 x r for "must return" and 1.3 x r for "must not".
    """
    rng = random.Random(1234)
    catalogs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with Store(tmp_path / "i.duckdb") as store:
        for i in range(12):
            ra_c = rng.uniform(0.0, 360.0)
            dec_c = rng.uniform(-60.0, 60.0)
            ra, dec = synth.patch(ra_c, dec_c, n=300, span=1.0, seed=i)
            uri = f"s3://b/c{i}.fits"
            catalogs[uri] = (ra, dec)
            add_file(store, uri, ra, dec, dilate=0)
        index = Index(store)
        for _ in range(30):
            ra_q = rng.uniform(0.0, 360.0)
            dec_q = rng.uniform(-60.0, 60.0)
            radius = rng.uniform(1.0, 3.0)
            found = {r.uri for r in index.search(cone_moc(ra_q, dec_q, radius * u.deg, ORDER))}
            for uri, (ra, dec) in catalogs.items():
                sep = _sep_deg(ra_q, dec_q, ra, dec)
                if np.any(sep <= 0.85 * radius):
                    assert uri in found, f"under-returned {uri}"
                if np.all(sep > 1.3 * radius):
                    assert uri not in found, f"over-returned {uri}"


def _sep_deg(ra1: float, dec1: float, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    lon1, lat1 = np.radians(ra1), np.radians(dec1)
    lon2, lat2 = np.radians(ra2), np.radians(dec2)
    cos_sep = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))


def test_uncovered_handles_disjoint_mocs() -> None:
    """Regression: MOC.difference returns empty for disjoint operands in mocpy 0.20."""
    a = MOC.from_string("3/0-3")
    b = MOC.from_string("3/100-103")
    assert a.difference(b).empty()  # documents the mocpy behaviour we avoid
    assert not uncovered(a, b).empty()
    assert uncovered(a, a).empty()
    assert uncovered(MOC.from_string("3/0-3"), MOC.from_string("3/0-7")).empty()


def test_dilation_grows_coverage() -> None:
    ra, dec = synth.patch(10.0, 10.0, n=50, seed=11)
    plain = build_moc(ra, dec, ORDER, 0)
    dilated = build_moc(ra, dec, ORDER, 1)
    assert uncovered(plain, dilated).empty()
    assert not uncovered(dilated, plain).empty()
    assert build_moc(ra, dec, ORDER, 0) == MOC.from_lonlat(
        ra * u.deg, dec * u.deg, max_norder=ORDER
    )
