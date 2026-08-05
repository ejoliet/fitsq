from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from fitsq.store import (
    SCHEMA_VERSION,
    FileRow,
    IndexMissing,
    SchemaMismatch,
    Store,
)


def row(uri: str = "s3://b/a.fits", **kw: object) -> FileRow:
    base = {
        "uri": uri,
        "etag": "e1",
        "size": 100,
        "nrows": 10,
        "row_bytes": 91,
        "data_offset": 5760,
        "ra_min": 1.0,
        "ra_max": 2.0,
        "dec_min": -1.0,
        "dec_max": 1.0,
        "ra_wraps": False,
        "moc_json": '{"9": [1, 2]}',
    }
    base.update(kw)
    return FileRow(**base)  # type: ignore[arg-type]


def test_create_and_upsert(tmp_path: Path) -> None:
    with Store(tmp_path / "i.duckdb") as store:
        store.upsert_file(row(), [(0, "SIMPLE", "T", "std"), (1, "XTENSION", "BINTABLE", "")])
        assert store.etags() == {"s3://b/a.fits": "e1"}
        rows = store.file_rows()
        assert len(rows) == 1 and rows[0].nrows == 10
        names, cards = store.sql("SELECT hdu, card_key FROM headers ORDER BY hdu")
        assert names == ["hdu", "card_key"] and len(cards) == 2
        assert store.get_meta("schema_version") == SCHEMA_VERSION


def test_upsert_replaces_row_and_cards(tmp_path: Path) -> None:
    with Store(tmp_path / "i.duckdb") as store:
        store.upsert_file(row(), [(0, "A", "1", "")])
        store.upsert_file(row(etag="e2", nrows=99), [(0, "B", "2", "")])
        rows = store.file_rows()
        assert len(rows) == 1 and rows[0].etag == "e2" and rows[0].nrows == 99
        _, cards = store.sql("SELECT card_key FROM headers")
        assert cards == [("B",)]


def test_file_rows_where_prefilter(tmp_path: Path) -> None:
    with Store(tmp_path / "i.duckdb") as store:
        store.upsert_file(row("s3://b/1.fits", dec_min=10.0, dec_max=11.0))
        store.upsert_file(row("s3://b/2.fits", dec_min=-40.0, dec_max=-39.0))
        hits = store.file_rows("dec_max >= ? AND dec_min <= ?", [9.0, 12.0])
        assert [r.uri for r in hits] == ["s3://b/1.fits"]


def test_errors_and_status(tmp_path: Path) -> None:
    path = tmp_path / "i.duckdb"
    with Store(path) as store:
        store.upsert_file(row())
        store.record_error("s3://b/bad.fits", "FitsError: boom")
        store.set_meta("prefix", "s3://b/")
        status = store.status()
        assert status["files"] == 1 and status["rows"] == 10 and status["errors"] == 1
        assert status["prefix"] == "s3://b/" and status["index_bytes"] > 0
        assert status["last_indexed"] is not None


def test_read_only_missing_index(tmp_path: Path) -> None:
    with pytest.raises(IndexMissing):
        Store(tmp_path / "nope.duckdb", read_only=True)


def test_reopen_read_only(tmp_path: Path) -> None:
    path = tmp_path / "i.duckdb"
    with Store(path) as store:
        store.upsert_file(row())
    with Store(path, read_only=True) as store:
        assert len(store.file_rows()) == 1


def test_schema_mismatch_refused(tmp_path: Path) -> None:
    path = tmp_path / "i.duckdb"
    with Store(path) as store:
        store.set_meta("schema_version", "999")
    with pytest.raises(SchemaMismatch, match="rebuild"):
        Store(path, read_only=True)


def test_status_on_empty_index(tmp_path: Path) -> None:
    with Store(tmp_path / "i.duckdb") as store:
        status = store.status()
        assert status["files"] == 0 and status["last_indexed"] is None


def test_upsert_rolls_back_on_failure(tmp_path: Path) -> None:
    with Store(tmp_path / "i.duckdb") as store:
        store.upsert_file(row())
        with pytest.raises(duckdb.Error):
            store.upsert_file(row(etag="e3"), [(0, "K", "v", object())])  # type: ignore[list-item]
        assert store.etags() == {"s3://b/a.fits": "e1"}
