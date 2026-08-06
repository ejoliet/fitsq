"""Query side: cone / region -> MOC -> bbox prefilter -> MOC intersection -> URIs."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from mocpy import MOC

from .store import FileRow, Store

DEFAULT_ORDER = 9

#: Frame the index is stored in. Galactic, because catalog filenames encode a
#: galactic HEALPix cell (see :mod:`fitsq.naming`): keeping the index in that
#: frame makes name-derived coverage *exact* — one cell, no rotation, no
#: dilation. mocpy does plain spherical maths on lon/lat and is frame-agnostic,
#: so a MOC of galactic values is as valid as one of ICRS values; only the two
#: sides of a comparison have to agree, which is what this constant enforces.
INDEX_FRAME = "galactic"

#: Accepted frame names (query input) -> astropy frame.
FRAME_ALIASES = {
    "icrs": "icrs",
    "fk5": "fk5",
    "j2000": "fk5",
    "fk4": "fk4",
    "b1950": "fk4",
    "galactic": "galactic",
    "gal": "galactic",
}

_UNIT_ALIASES = {
    "deg": u.deg,
    "degree": u.deg,
    "degrees": u.deg,
    "d": u.deg,
    "arcmin": u.arcmin,
    "amin": u.arcmin,
    "'": u.arcmin,
    "arcsec": u.arcsec,
    "asec": u.arcsec,
    '"': u.arcsec,
    "rad": u.rad,
    "radian": u.rad,
    "radians": u.rad,
}

_RADIUS_RE = re.compile(r"^\s*([+-]?[\d.eE+-]+)\s*([a-zA-Z'\"]*)\s*$")


class QueryError(Exception):
    """Bad query input (radius, STC-S string, ...)."""


def parse_angle(text: str, default_unit: str = "deg") -> u.Quantity:
    """Parse ``30arcsec`` / ``2arcmin`` / ``0.5deg`` / bare number + default unit."""
    match = _RADIUS_RE.match(text)
    if match is None:
        raise QueryError(f"cannot parse angle {text!r}")
    try:
        value = float(match.group(1))
    except ValueError as exc:
        raise QueryError(f"cannot parse angle {text!r}") from exc
    name = (match.group(2) or default_unit).lower()
    unit = _UNIT_ALIASES.get(name)
    if unit is None:
        raise QueryError(f"unknown angle unit {match.group(2)!r}")
    if value <= 0:
        raise QueryError("angle must be positive")
    return value * unit


def resolve_frame(name: str) -> str:
    """Map a user-supplied frame name to an astropy frame, or raise."""
    frame = FRAME_ALIASES.get(name.strip().lower())
    if frame is None:
        raise QueryError(f"unknown frame {name!r}; supported: {', '.join(sorted(FRAME_ALIASES))}")
    return frame


def convert(
    lon_deg: float | Sequence[float] | np.ndarray,
    lat_deg: float | Sequence[float] | np.ndarray,
    from_frame: str,
    to_frame: str = INDEX_FRAME,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert longitude/latitude degrees between frames.

    Frame changes among these systems are rotations, so a cone's radius and a
    polygon's great-circle edges are preserved: converting the centre or the
    vertices is enough, no re-tessellation needed.
    """
    src = resolve_frame(from_frame)
    dst = resolve_frame(to_frame)
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float))
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    if src == dst:
        return lon, lat
    out = getattr(SkyCoord(lon * u.deg, lat * u.deg, frame=src), dst)
    spherical = out.spherical
    return np.atleast_1d(spherical.lon.deg), np.atleast_1d(spherical.lat.deg)


def to_icrs(
    lon_deg: float | Sequence[float] | np.ndarray,
    lat_deg: float | Sequence[float] | np.ndarray,
    frame: str = "icrs",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert longitude/latitude in ``frame`` to ICRS degrees."""
    return convert(lon_deg, lat_deg, frame, "icrs")


def to_index_frame(
    lon_deg: float | Sequence[float] | np.ndarray,
    lat_deg: float | Sequence[float] | np.ndarray,
    frame: str = "icrs",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert query/row coordinates into the frame the index is stored in."""
    return convert(lon_deg, lat_deg, frame, INDEX_FRAME)


def cone_moc(
    lon_deg: float, lat_deg: float, radius: u.Quantity, order: int, frame: str = "icrs"
) -> MOC:
    """Cone MOC in the index frame. Input is read in ``frame`` (e.g. galactic l/b)."""
    lon, lat = to_index_frame(lon_deg, lat_deg, frame)
    return MOC.from_cone(
        lon=float(lon[0]) * u.deg, lat=float(lat[0]) * u.deg, radius=radius, max_depth=order
    )


def polygon_moc(lon_deg: list[float], lat_deg: list[float], order: int, frame: str = "icrs") -> MOC:
    """Polygon MOC in the index frame, from vertices given in ``frame``."""
    lon, lat = to_index_frame(lon_deg, lat_deg, frame)
    return MOC.from_polygon(lon=lon * u.deg, lat=lat * u.deg, max_depth=order)


def parse_stcs(text: str, order: int) -> MOC:
    """v1 STC-S subset: ``POLYGON <frame> lon lat ...`` and ``CIRCLE <frame> lon lat r``.

    Frame is required and honoured: coordinates are converted from it into the
    index frame rather than silently taken as already being in it.
    """
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise QueryError("empty region string")
    shape = tokens[0].upper()
    if shape not in ("POLYGON", "CIRCLE"):
        raise QueryError(
            f"unsupported STC-S construct {shape!r}; v1 supports POLYGON and CIRCLE only"
        )
    rest = tokens[1:]
    if not rest or rest[0].lower() not in FRAME_ALIASES:
        raise QueryError(
            f"{shape} requires a frame, e.g. '{shape} ICRS ...' or '{shape} GALACTIC ...'"
        )
    frame = rest[0]
    rest = rest[1:]
    try:
        numbers = [float(token) for token in rest]
    except ValueError as exc:
        raise QueryError(f"non-numeric coordinate in region: {text!r}") from exc
    if shape == "CIRCLE":
        if len(numbers) != 3:
            raise QueryError("CIRCLE needs 3 numbers: lon lat radius (deg)")
        lon, lat, radius = numbers
        return cone_moc(lon, lat, radius * u.deg, order, frame)
    if shape == "POLYGON":
        if len(numbers) < 6 or len(numbers) % 2:
            raise QueryError("POLYGON needs >= 3 lon/lat pairs")
        return polygon_moc(numbers[0::2], numbers[1::2], order, frame)
    raise QueryError(f"unsupported STC-S construct {shape!r}")  # pragma: no cover


@dataclass(frozen=True)
class BBox:
    """Conservative lon/lat bounds of a region. ``wraps`` crosses lon=0."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    wraps: bool
    lon_unbounded: bool = False


def bounding_box(moc: MOC) -> BBox:
    """Bounding box of a MOC, via its bounding cone (barycenter + max vertex distance).

    Always a superset of the MOC, so it is safe as a prefilter. Values are in
    whatever frame the MOC carries — mocpy labels coordinates ICRS regardless,
    so the numbers are read positionally rather than trusted as ra/dec.
    """
    center = moc.barycenter()
    radius_deg = float(moc.largest_distance_from_coo_to_vertices(center).to_value(u.deg))
    spherical = center.spherical
    lon = float(spherical.lon.deg)
    lat = float(spherical.lat.deg)
    lat_min = lat - radius_deg
    lat_max = lat + radius_deg
    worst_lat = max(abs(min(lat_min, 90.0)), abs(max(lat_max, -90.0)))
    if lat_min <= -90.0 or lat_max >= 90.0 or radius_deg >= 90.0 or worst_lat >= 89.0:
        return BBox(0.0, 360.0, max(lat_min, -90.0), min(lat_max, 90.0), False, True)
    dlon = radius_deg / math.cos(math.radians(worst_lat))
    if dlon >= 180.0:
        return BBox(0.0, 360.0, lat_min, lat_max, False, True)
    lo = (lon - dlon) % 360.0
    hi = (lon + dlon) % 360.0
    return BBox(lo, hi, lat_min, lat_max, lo > hi)


def _where(bbox: BBox) -> tuple[str, list[float | bool]]:
    clauses = ["lat_max >= ?", "lat_min <= ?"]
    params: list[float | bool] = [bbox.lat_min, bbox.lat_max]
    if not bbox.lon_unbounded:
        # File boxes that wrap RA=0 are kept unconditionally: cheap and conservative.
        clauses.append("(lon_wraps OR CAST(? AS BOOLEAN) OR (lon_max >= ? AND lon_min <= ?))")
        params += [bbox.wraps, bbox.lon_min, bbox.lon_max]
    return " AND ".join(clauses), params


class Index:
    """Read-only query handle. Caches parsed file MOCs for the process lifetime."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self._mocs: dict[str, MOC] = {}

    def moc_for(self, row: FileRow) -> MOC:
        moc = self._mocs.get(row.uri)
        if moc is None:
            moc = MOC.from_string(row.moc_json, format="json")
            self._mocs[row.uri] = moc
        return moc

    def candidates(self, moc: MOC) -> list[FileRow]:
        where, params = _where(bounding_box(moc))
        return self.store.file_rows(where, params)

    def search(self, moc: MOC) -> list[FileRow]:
        """Files whose sampled coverage intersects ``moc``, sorted by URI."""
        hits = [
            row for row in self.candidates(moc) if not moc.intersection(self.moc_for(row)).empty()
        ]
        return sorted(hits, key=lambda row: row.uri)
