"""DuckDB index store: schema, upserts, query helpers."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

#: 2 renamed the bbox columns from ra_*/dec_* to lon_*/lat_*: MOCs and the box
#: are stored in the index frame (galactic), not ICRS. See meta 'moc_frame'.
SCHEMA_VERSION = "2"

# Note: crawl_errors.at is a DuckDB reserved word — quote it in ad-hoc SQL,
# e.g. SELECT uri, "at" FROM crawl_errors.

DEFAULT_INDEX = Path(
    os.environ.get("FITSQ_INDEX") or Path.home() / ".cache" / "fitsq" / "index.duckdb"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
  uri TEXT PRIMARY KEY, etag TEXT, size BIGINT,
  nrows BIGINT, row_bytes INT, data_offset BIGINT,
  -- bbox prefilter, in the index frame (see meta 'moc_frame'), degrees
  lon_min DOUBLE, lon_max DOUBLE, lat_min DOUBLE, lat_max DOUBLE,
  lon_wraps BOOLEAN,   -- box crosses lon=0
  moc_json TEXT,
  indexed_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS headers (
  uri TEXT, hdu INT, card_key TEXT, card_value TEXT, card_comment TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS crawl_errors (uri TEXT, error TEXT, "at" TIMESTAMP);
"""

#: DuckDB's read_csv maps an empty field to NULL even when it is quoted, which
#: would silently turn an empty FITS card value into NULL. Point nullstr at a
#: sentinel that cannot occur in an 80-char card instead.
#: strict_mode=false is required too: moc_json is pretty-printed and so contains
#: newlines inside a quoted field, which otherwise defeats the CSV sniffer.
_CSV_NULLSTR = "__fitsq_null__"

#: column specs for the CSV staging path in :meth:`Store.upsert_files`
_FILES_CSV_SPEC = (
    "header=false, strict_mode=false, quote='\"', escape='\"', "
    "nullstr=['" + _CSV_NULLSTR + "'], columns={"
    "'uri':'TEXT','etag':'TEXT','size':'BIGINT','nrows':'BIGINT','row_bytes':'INT',"
    "'data_offset':'BIGINT','lon_min':'DOUBLE','lon_max':'DOUBLE','lat_min':'DOUBLE',"
    "'lat_max':'DOUBLE','lon_wraps':'BOOLEAN','moc_json':'TEXT','indexed_at':'TIMESTAMP'}"
)
_HEADERS_CSV_SPEC = (
    "header=false, strict_mode=false, quote='\"', escape='\"', "
    "nullstr=['" + _CSV_NULLSTR + "'], columns={"
    "'uri':'TEXT','hdu':'INT','card_key':'TEXT','card_value':'TEXT','card_comment':'TEXT'}"
)

FILE_COLUMNS = (
    "uri",
    "etag",
    "size",
    "nrows",
    "row_bytes",
    "data_offset",
    "lon_min",
    "lon_max",
    "lat_min",
    "lat_max",
    "lon_wraps",
    "moc_json",
    "indexed_at",
)


class SchemaMismatch(Exception):
    """Index was written by an incompatible fitsq version."""


class IndexMissing(Exception):
    """No index file at the requested path."""


@dataclass(frozen=True)
class FileRow:
    uri: str
    etag: str
    size: int
    nrows: int
    row_bytes: int
    data_offset: int
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    lon_wraps: bool
    moc_json: str


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _write_csv(path: Path, rows: Sequence[tuple[Any, ...]]) -> Path:
    """Stage rows for read_csv. QUOTE_ALL keeps '' distinct from NULL."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, quoting=csv.QUOTE_ALL).writerows(rows)
    return path


class Store:
    """Read/write handle on the DuckDB index."""

    def __init__(self, path: Path | str, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if read_only and not self.path.exists():
            raise IndexMissing(f"no index at {self.path}")
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            self.conn.execute("BEGIN")
            for statement in filter(str.strip, SCHEMA.split(";")):
                self.conn.execute(statement)
            self.conn.execute("COMMIT")
            self.set_meta("schema_version", SCHEMA_VERSION)
        self._check_schema()

    def _check_schema(self) -> None:
        found = self.get_meta("schema_version")
        if found not in (None, SCHEMA_VERSION):
            raise SchemaMismatch(
                f"index {self.path} has schema_version {found}, expected {SCHEMA_VERSION}: "
                "rebuild the index (delete the file and re-run `fitsq index`)"
            )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- meta ---------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", [key]).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("DELETE FROM meta WHERE key = ?", [key])
        self.conn.execute("INSERT INTO meta VALUES (?, ?)", [key, value])

    # -- writes -------------------------------------------------------------

    def etags(self) -> dict[str, str]:
        """URI -> ETag for everything already indexed (resume support)."""
        rows = self.conn.execute("SELECT uri, etag FROM files").fetchall()
        return {str(uri): str(etag) for uri, etag in rows}

    def upsert_file(self, row: FileRow, cards: Iterable[tuple[int, str, str, str]] = ()) -> None:
        """Replace a file's index entry and its header cards atomically."""
        self.upsert_files([(row, tuple(cards))])

    def upsert_files(
        self, batch: Sequence[tuple[FileRow, Sequence[tuple[int, str, str, str]]]]
    ) -> None:
        """Replace a batch of file entries and their header cards, in one transaction.

        Staged through CSV rather than INSERT ... VALUES: DuckDB 1.5 executes
        row-wise parameterised inserts at ~4 ms/row, which put a 2.5k-file crawl
        (~75k header cards) at ten minutes. read_csv does the same load in well
        under a second. Quoting is QUOTE_ALL with a sentinel nullstr so that an empty
        card value round-trips as '' and not NULL.
        """
        if not batch:
            return
        file_rows = [
            (
                row.uri,
                row.etag,
                row.size,
                row.nrows,
                row.row_bytes,
                row.data_offset,
                row.lon_min,
                row.lon_max,
                row.lat_min,
                row.lat_max,
                row.lon_wraps,
                row.moc_json,
                _now().isoformat(sep=" "),
            )
            for row, _ in batch
        ]
        card_rows = [
            (row.uri, hdu, key, value, comment)
            for row, cards in batch
            for hdu, key, value, comment in cards
        ]
        self.conn.execute("BEGIN")
        try:
            with tempfile.TemporaryDirectory(prefix="fitsq-stage-") as staging:
                files_csv = _write_csv(Path(staging) / "files.csv", file_rows)
                self.conn.execute(
                    f"CREATE OR REPLACE TEMP TABLE stage_files AS "
                    f"SELECT * FROM read_csv('{files_csv}', {_FILES_CSV_SPEC})"
                )
                self.conn.execute("DELETE FROM files WHERE uri IN (SELECT uri FROM stage_files)")
                self.conn.execute("DELETE FROM headers WHERE uri IN (SELECT uri FROM stage_files)")
                self.conn.execute("INSERT INTO files SELECT * FROM stage_files")
                if card_rows:
                    cards_csv = _write_csv(Path(staging) / "headers.csv", card_rows)
                    self.conn.execute(
                        f"INSERT INTO headers "
                        f"SELECT * FROM read_csv('{cards_csv}', {_HEADERS_CSV_SPEC})"
                    )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def record_error(self, uri: str, error: str) -> None:
        self.conn.execute("INSERT INTO crawl_errors VALUES (?, ?, ?)", [uri, error, _now()])

    # -- reads --------------------------------------------------------------

    def file_rows(self, where: str = "", params: Sequence[Any] = ()) -> list[FileRow]:
        """Load file rows, optionally narrowed by a WHERE clause (bbox prefilter)."""
        columns = ", ".join(c for c in FILE_COLUMNS if c != "indexed_at")
        sql = f"SELECT {columns} FROM files"
        if where:
            sql += f" WHERE {where}"
        rows = self.conn.execute(sql, list(params)).fetchall()
        return [
            FileRow(
                uri=str(r[0]),
                etag=str(r[1]),
                size=int(r[2]),
                nrows=int(r[3]),
                row_bytes=int(r[4]),
                data_offset=int(r[5]),
                lon_min=float(r[6]),
                lon_max=float(r[7]),
                lat_min=float(r[8]),
                lat_max=float(r[9]),
                lon_wraps=bool(r[10]),
                moc_json=str(r[11]),
            )
            for r in rows
        ]

    def status(self) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT count(*), coalesce(sum(nrows), 0), coalesce(sum(size), 0), max(indexed_at) "
            "FROM files"
        ).fetchone()
        errors = self.conn.execute("SELECT count(*) FROM crawl_errors").fetchone()
        assert row is not None and errors is not None
        return {
            "index": str(self.path),
            "index_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "files": int(row[0]),
            "rows": int(row[1]),
            "catalog_bytes": int(row[2]),
            "last_indexed": str(row[3]) if row[3] is not None else None,
            "errors": int(errors[0]),
            "prefix": self.get_meta("prefix"),
            "coverage": self.get_meta("coverage"),
            "moc_frame": self.get_meta("moc_frame"),
            "order": self.get_meta("order"),
            "dilate": self.get_meta("dilate"),
            "samples": self.get_meta("samples"),
            "sample_rows": self.get_meta("sample_rows"),
        }

    def sql(self, query: str) -> tuple[list[str], list[tuple[Any, ...]]]:
        cursor = self.conn.execute(query)
        names = [d[0] for d in cursor.description or ()]
        return names, cursor.fetchall()
