"""Query side: cone / region -> MOC -> bbox prefilter -> MOC intersection -> URIs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from mocpy import MOC

from .store import FileRow, Store

DEFAULT_ORDER = 9

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


def cone_moc(ra_deg: float, dec_deg: float, radius: u.Quantity, order: int) -> MOC:
    return MOC.from_cone(lon=ra_deg * u.deg, lat=dec_deg * u.deg, radius=radius, max_depth=order)


def polygon_moc(lon_deg: list[float], lat_deg: list[float], order: int) -> MOC:
    return MOC.from_polygon(
        lon=np.asarray(lon_deg) * u.deg, lat=np.asarray(lat_deg) * u.deg, max_depth=order
    )


def parse_stcs(text: str, order: int) -> MOC:
    """v1 STC-S subset: ``POLYGON ICRS lon lat ...`` and ``CIRCLE ICRS lon lat r``."""
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise QueryError("empty region string")
    shape = tokens[0].upper()
    if shape not in ("POLYGON", "CIRCLE"):
        raise QueryError(
            f"unsupported STC-S construct {shape!r}; v1 supports POLYGON and CIRCLE only"
        )
    rest = tokens[1:]
    if rest and rest[0].upper() in ("ICRS", "FK5", "J2000"):
        rest = rest[1:]
    else:
        raise QueryError(f"{shape} requires a frame, e.g. '{shape} ICRS ...'")
    try:
        numbers = [float(token) for token in rest]
    except ValueError as exc:
        raise QueryError(f"non-numeric coordinate in region: {text!r}") from exc
    if shape == "CIRCLE":
        if len(numbers) != 3:
            raise QueryError("CIRCLE needs 3 numbers: lon lat radius (deg)")
        lon, lat, radius = numbers
        return cone_moc(lon, lat, radius * u.deg, order)
    if shape == "POLYGON":
        if len(numbers) < 6 or len(numbers) % 2:
            raise QueryError("POLYGON needs >= 3 lon/lat pairs")
        return polygon_moc(numbers[0::2], numbers[1::2], order)
    raise QueryError(f"unsupported STC-S construct {shape!r}")  # pragma: no cover


@dataclass(frozen=True)
class BBox:
    """Conservative lon/lat bounds of a query region. ``wraps`` crosses RA=0."""

    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float
    wraps: bool
    ra_unbounded: bool = False


def bounding_box(moc: MOC) -> BBox:
    """Bounding box of a MOC, via its bounding cone (barycenter + max vertex distance).

    Always a superset of the MOC, so it is safe as a prefilter.
    """
    center = moc.barycenter()
    radius_deg = float(moc.largest_distance_from_coo_to_vertices(center).to_value(u.deg))
    ra = float(center.icrs.ra.deg)
    dec = float(center.icrs.dec.deg)
    dec_min = dec - radius_deg
    dec_max = dec + radius_deg
    worst_dec = max(abs(min(dec_min, 90.0)), abs(max(dec_max, -90.0)))
    if dec_min <= -90.0 or dec_max >= 90.0 or radius_deg >= 90.0 or worst_dec >= 89.0:
        return BBox(0.0, 360.0, max(dec_min, -90.0), min(dec_max, 90.0), False, True)
    dra = radius_deg / math.cos(math.radians(worst_dec))
    if dra >= 180.0:
        return BBox(0.0, 360.0, dec_min, dec_max, False, True)
    lo = (ra - dra) % 360.0
    hi = (ra + dra) % 360.0
    return BBox(lo, hi, dec_min, dec_max, lo > hi)


def _where(bbox: BBox) -> tuple[str, list[float | bool]]:
    clauses = ["dec_max >= ?", "dec_min <= ?"]
    params: list[float | bool] = [bbox.dec_min, bbox.dec_max]
    if not bbox.ra_unbounded:
        # File boxes that wrap RA=0 are kept unconditionally: cheap and conservative.
        clauses.append("(ra_wraps OR CAST(? AS BOOLEAN) OR (ra_max >= ? AND ra_min <= ?))")
        params += [bbox.wraps, bbox.ra_min, bbox.ra_max]
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
