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
    INDEX_FRAME,
    BBox,
    Index,
    QueryError,
    bounding_box,
    cone_moc,
    parse_angle,
    parse_stcs,
    polygon_moc,
    resolve_frame,
    to_icrs,
    to_index_frame,
)
from fitsq.store import FileRow, Store

ORDER = 9


def add_file(store: Store, uri: str, ra: np.ndarray, dec: np.ndarray, dilate: int = 1) -> None:
    """Index synthetic ICRS rows the way the sampling crawl does (index frame)."""
    lon, lat = to_index_frame(ra, dec, "icrs")
    moc = build_moc(lon, lat, ORDER, dilate)
    lon_min, lon_max, lat_min, lat_max, wraps = coord_bbox(lon, lat)
    store.upsert_file(
        FileRow(
            uri=uri,
            etag="e",
            size=len(ra) * 91,
            nrows=len(ra),
            row_bytes=91,
            data_offset=5760,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_wraps=wraps,
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


def test_to_icrs_galactic_known_value() -> None:
    """l=3.45 b=-0.27 is ICRS (268.66289, -26.11349) — the Roman bulge field."""
    ra, dec = to_icrs(3.45, -0.27, "galactic")
    assert ra[0] == pytest.approx(268.66289, abs=1e-4)
    assert dec[0] == pytest.approx(-26.11349, abs=1e-4)
    # icrs is a no-op pass-through
    assert to_icrs(268.77, -29.25, "icrs") == (pytest.approx([268.77]), pytest.approx([-29.25]))


def test_resolve_frame_aliases_and_errors() -> None:
    assert resolve_frame("GALACTIC") == "galactic"
    assert resolve_frame("gal") == "galactic"
    assert resolve_frame("J2000") == "fk5"
    assert resolve_frame("b1950") == "fk4"
    with pytest.raises(QueryError, match="unknown frame"):
        resolve_frame("supergalactic")


def test_galactic_cone_finds_file_indexed_in_icrs(tmp_path: Path) -> None:
    """A galactic query must hit a file whose rows are stored as ICRS ra/dec."""
    ra_c, dec_c = 268.66289, -26.11349  # == l=3.45, b=-0.27
    with Store(tmp_path / "i.duckdb") as store:
        add_file(store, "s3://b/target.fits", *synth.patch(ra_c, dec_c, seed=21))
        add_file(store, "s3://b/elsewhere.fits", *synth.patch(268.77, -29.25, seed=22))
        index = Index(store)
        gal = index.search(cone_moc(3.45, -0.27, 30 * u.arcsec, ORDER, "galactic"))
        assert [r.uri for r in gal] == ["s3://b/target.fits"]
        # same sky position expressed in ICRS gives the identical answer
        icrs = index.search(cone_moc(ra_c, dec_c, 30 * u.arcsec, ORDER))
        assert [r.uri for r in icrs] == [r.uri for r in gal]


def test_galactic_and_icrs_cones_agree_exactly() -> None:
    """Frame change is a rotation, so the MOC must be identical either way."""
    ra, dec = to_icrs(3.45, -0.27, "galactic")
    assert cone_moc(3.45, -0.27, 2 * u.arcmin, ORDER, "galactic") == cone_moc(
        float(ra[0]), float(dec[0]), 2 * u.arcmin, ORDER
    )


def test_galactic_polygon_converts_all_vertices() -> None:
    lon = [3.4, 3.5, 3.5, 3.4]
    lat = [-0.32, -0.32, -0.22, -0.22]
    gal = polygon_moc(lon, lat, ORDER, "galactic")
    ra, dec = to_icrs(lon, lat, "galactic")
    assert gal == polygon_moc(list(ra), list(dec), ORDER)
    assert not gal.empty()


def test_stcs_honours_galactic_frame() -> None:
    """Regression: the frame token used to be parsed then ignored (treated as ICRS)."""
    gal = parse_stcs("CIRCLE GALACTIC 3.45 -0.27 0.05", ORDER)
    assert gal == cone_moc(3.45, -0.27, 0.05 * u.deg, ORDER, "galactic")
    as_icrs = parse_stcs("CIRCLE ICRS 3.45 -0.27 0.05", ORDER)
    assert gal != as_icrs
    poly = parse_stcs("POLYGON GALACTIC 3.4 -0.32 3.5 -0.32 3.5 -0.22", ORDER)
    assert poly == polygon_moc([3.4, 3.5, 3.5], [-0.32, -0.32, -0.22], ORDER, "galactic")
    with pytest.raises(QueryError, match="requires a frame"):
        parse_stcs("CIRCLE SUPERGALACTIC 1 2 3", ORDER)


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
    # built in the index frame so the input values are the MOC's own values
    moc = cone_moc(268.77, -29.25, 0.5 * u.deg, ORDER, INDEX_FRAME)
    box = bounding_box(moc)
    assert box.lat_min < -29.75 and box.lat_max > -28.75
    assert not box.lon_unbounded and not box.wraps
    assert box.lon_min < 268.77 < box.lon_max


def test_bounding_box_flags_wrap_and_poles() -> None:
    assert bounding_box(cone_moc(0.05, 0.0, 0.5 * u.deg, ORDER, INDEX_FRAME)).wraps
    assert bounding_box(cone_moc(10.0, 89.9, 0.5 * u.deg, ORDER, INDEX_FRAME)).lon_unbounded
    wide = bounding_box(cone_moc(30.0, 0.0, 60.0 * u.deg, ORDER, INDEX_FRAME))
    # A 60 deg cone at the equator spans lon 330..90; the box must contain that.
    assert wide.wraps and wide.lon_min < 330.0 and wide.lon_max > 90.0
    assert wide.lat_min < -60.0 and wide.lat_max > 60.0


def test_bounding_box_lon_halfwidth_never_under_covers_a_cone() -> None:
    """The prefilter box must never clip a cone: lon half-width >= exact value."""
    for lat_center in (0.0, 15.0, -30.0, 45.0, -60.0, 75.0, 88.0, -88.5):
        for radius in (0.01, 0.1, 0.5, 2.0, 10.0, 30.0):
            box = bounding_box(cone_moc(123.0, lat_center, radius * u.deg, ORDER, INDEX_FRAME))
            assert box.lat_min <= max(lat_center - radius, -90.0)
            assert box.lat_max >= min(lat_center + radius, 90.0)
            if box.lon_unbounded:
                continue
            half_width = ((box.lon_max - box.lon_min) % 360.0) / 2.0
            denominator = np.cos(np.radians(lat_center))
            ratio = np.sin(np.radians(radius)) / denominator
            exact = 180.0 if ratio >= 1.0 else np.degrees(np.arcsin(ratio))
            assert half_width >= exact, (lat_center, radius, half_width, exact)


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
