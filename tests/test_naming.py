from __future__ import annotations

import astropy.units as u
import pytest

import synth
from fitsq.naming import (
    TILE_ORDER,
    NamingError,
    cell_moc,
    moc_for_uri,
    npix,
    parse_pixel,
    pixel_for_lonlat,
)
from fitsq.query import to_index_frame

PREFIX = "s3://stpubdata/roman/nexus/soc_simulations/input_catalogs/"


def test_npix_matches_nside_512() -> None:
    assert TILE_ORDER == 9
    assert npix() == 12 * 512**2 == 3_145_728


def test_parse_pixel() -> None:
    assert parse_pixel(f"{PREFIX}cat-1113599.fits") == 1113599
    assert parse_pixel("cat-0.fits") == 0
    # the 's3' in the scheme must not be mistaken for the pixel
    assert parse_pixel("s3://bucket/3/cat-2921486.fits") == 2921486
    assert parse_pixel("CAT-1113533.FITS") == 1113533


def test_parse_pixel_rejects_bad_names() -> None:
    for bad in ("s3://b/tile-A7.fits", "s3://b/cat-.fits", "s3://b/catalog.fits", "s3://b/cat-1"):
        with pytest.raises(NamingError, match="does not match"):
            parse_pixel(bad)
    with pytest.raises(NamingError, match="out of range"):
        parse_pixel(f"s3://b/cat-{npix()}.fits")


def test_cell_moc_is_a_single_exact_cell() -> None:
    moc = cell_moc(1113599)
    assert moc.max_order == TILE_ORDER
    assert len(moc.flatten()) == 1
    assert moc == moc_for_uri(f"{PREFIX}cat-1113599.fits")
    with pytest.raises(NamingError):
        cell_moc(-1)


def test_pixel_round_trip_through_cell_centre() -> None:
    for pixel in (0, 1113533, 1113599, 1157294, npix() - 1):
        lon, lat = synth.cell_centre_galactic(pixel)
        assert pixel_for_lonlat(lon, lat) == pixel


def test_known_real_file_pixel_matches_measured_position() -> None:
    """cat-1113599 was measured on the bucket at ICRS (268.4683, -28.3813)."""
    lon, lat = to_index_frame(268.4683, -28.3813, "icrs")
    assert pixel_for_lonlat(float(lon[0]), float(lat[0])) == 1113599


def test_cell_contains_its_own_rows_but_not_a_neighbour_cell() -> None:
    ra, dec = synth.cell_patch(1113599, n=200, seed=1)
    lon, lat = to_index_frame(ra, dec, "icrs")
    cell = cell_moc(1113599)
    assert cell.contains_lonlat(lon * u.deg, lat * u.deg).all()
    assert not cell_moc(1157294).contains_lonlat(lon * u.deg, lat * u.deg).any()
