"""Filename -> sky coverage.

Catalog files are named ``cat-<pix>.fits`` where ``<pix>`` is the HEALPix
**NESTED** pixel index at order 9 (nside 512) **in galactic coordinates**.
Verified against the whole bucket: the suffix equals the galactic order-9
nested pixel of the tile for 2495/2495 files (0/2495 in ICRS), and every
sampled row falls inside its own named cell.

That makes per-file coverage derivable from the listing alone, exactly and
with no data reads. ``fitsq validate`` remains the guard: it rebuilds the true
footprint from the rows and fails if a file ever breaks the convention.
"""

from __future__ import annotations

import re

import astropy.units as u
from mocpy import MOC

#: HEALPix order of the tiling encoded in filenames (nside 512).
TILE_ORDER = 9

_NAME_RE = re.compile(r"cat-(\d+)\.fits$", re.IGNORECASE)


class NamingError(Exception):
    """Filename does not encode a usable HEALPix pixel."""


def npix(order: int = TILE_ORDER) -> int:
    """Number of HEALPix cells at ``order``."""
    return int(12 * 4**order)  # int() because mypy types int**int as Any


def parse_pixel(uri: str, order: int = TILE_ORDER) -> int:
    """Extract the galactic HEALPix pixel index from a ``cat-<pix>.fits`` URI."""
    match = _NAME_RE.search(uri.rsplit("/", 1)[-1])
    if match is None:
        raise NamingError(f"filename does not match cat-<pix>.fits: {uri!r}")
    pixel = int(match.group(1))
    limit = npix(order)
    if not 0 <= pixel < limit:
        raise NamingError(f"pixel {pixel} out of range for order {order} (0..{limit - 1})")
    return pixel


def cell_moc(pixel: int, order: int = TILE_ORDER) -> MOC:
    """The named cell as a MOC, in galactic coordinates.

    Exact: a single HEALPix cell needs no approximation and no dilation, unlike
    a sampled footprint. mocpy does plain spherical maths on lon/lat, so the
    galactic values are carried as-is (see :data:`fitsq.query.INDEX_FRAME`).
    """
    if not 0 <= pixel < npix(order):
        raise NamingError(f"pixel {pixel} out of range for order {order}")
    return MOC.from_json({str(order): [pixel]})


def moc_for_uri(uri: str, order: int = TILE_ORDER) -> MOC:
    """Coverage of a catalog file, derived from its name alone."""
    return cell_moc(parse_pixel(uri, order), order)


def pixel_for_lonlat(lon_deg: float, lat_deg: float, order: int = TILE_ORDER) -> int:
    """Galactic ``l``/``b`` -> HEALPix pixel, i.e. the expected filename suffix."""
    flat = MOC.from_lonlat(lon_deg * u.deg, lat_deg * u.deg, max_norder=order).flatten()
    return int(flat[0])
