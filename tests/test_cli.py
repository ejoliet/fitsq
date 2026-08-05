from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from moto import mock_aws
from typer.testing import CliRunner

import synth
from fitsq import cli
from fitsq.s3io import S3Reader

BUCKET = "stpubdata"
PREFIX = f"s3://{BUCKET}/cat/"
runner = CliRunner()

TILES = {"cat-1113533.fits": (268.77, -29.25), "cat-2.fits": (10.0, 10.0)}


@pytest.fixture
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A real index built from a mocked bucket, via the `index` command itself."""
    index_path = tmp_path / "index.duckdb"
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        for name, (ra_c, dec_c) in TILES.items():
            ra, dec = synth.patch(ra_c, dec_c, n=400, seed=len(name))
            client.put_object(Bucket=BUCKET, Key=f"cat/{name}", Body=synth.catalog(ra, dec))
        monkeypatch.setattr(cli, "S3Reader", lambda **_kw: S3Reader(client))
        result = runner.invoke(
            cli.app,
            ["index", PREFIX, "--index", str(index_path), "--sample-rows", "200", "--workers", "2"],
        )
        assert result.exit_code == 0, result.output
        assert "indexed 2" in result.output
        yield index_path


def test_cone_text_json_csv(indexed: Path) -> None:
    args = ["cone", "268.77", "-29.25", "30arcsec", "--index", str(indexed)]
    text = runner.invoke(cli.app, args)
    assert text.exit_code == 0
    assert text.output.strip() == f"{PREFIX}cat-1113533.fits"

    as_json = runner.invoke(cli.app, [*args, "--format", "json"])
    payload = json.loads(as_json.output)
    assert payload[0]["uri"].endswith("cat-1113533.fits") and payload[0]["nrows"] == 400

    as_csv = runner.invoke(cli.app, [*args, "--format", "csv"])
    rows = list(csv.reader(io.StringIO(as_csv.output)))
    assert rows[0] == ["uri", "nrows", "size"] and rows[1][1] == "400"


def test_cone_bare_radius_with_unit(indexed: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["cone", "268.77", "-29.25", "0.05", "--unit", "deg", "--index", str(indexed)],
    )
    assert result.exit_code == 0 and "cat-1113533.fits" in result.output


def test_cone_empty_sky_exits_zero_with_no_output(indexed: Path) -> None:
    result = runner.invoke(cli.app, ["cone", "150.0", "40.0", "30arcsec", "--index", str(indexed)])
    assert result.exit_code == 0 and result.output.strip() == ""


def test_cone_bad_radius_exits_2(indexed: Path) -> None:
    result = runner.invoke(cli.app, ["cone", "1", "2", "5parsec", "--index", str(indexed)])
    assert result.exit_code == 2


def test_missing_index_exits_1(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app, ["cone", "1", "2", "1deg", "--index", str(tmp_path / "nope.duckdb")]
    )
    assert result.exit_code == 1
    assert "no index" in result.output


def test_region_polygon_and_rejection(indexed: Path) -> None:
    ok = runner.invoke(
        cli.app,
        [
            "region",
            "POLYGON ICRS 268.7 -29.3 268.9 -29.3 268.9 -29.1 268.7 -29.1",
            "--index",
            str(indexed),
        ],
    )
    assert ok.exit_code == 0 and "cat-1113533.fits" in ok.output
    bad = runner.invoke(cli.app, ["region", "UNION ICRS (Circle 1 2 3)", "--index", str(indexed)])
    assert bad.exit_code == 2 and "v1 supports" in bad.output


def test_sql_and_status(indexed: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["sql", "SELECT count(*) AS n, sum(nrows) AS rows FROM files", "--index", str(indexed)],
    )
    assert result.exit_code == 0
    rows = list(csv.reader(io.StringIO(result.output)))
    assert rows[0] == ["n", "rows"] and rows[1] == ["2", "800"]

    status = runner.invoke(cli.app, ["status", "--index", str(indexed)])
    payload = json.loads(status.output)
    assert payload["files"] == 2 and payload["rows"] == 800
    assert payload["prefix"] == PREFIX and payload["order"] == "9"


def test_validate_command(indexed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        for name, (ra_c, dec_c) in TILES.items():
            ra, dec = synth.patch(ra_c, dec_c, n=400, seed=len(name))
            client.put_object(Bucket=BUCKET, Key=f"cat/{name}", Body=synth.catalog(ra, dec))
        monkeypatch.setattr(cli, "S3Reader", lambda **_kw: S3Reader(client))
        result = runner.invoke(
            cli.app, ["validate", "--index", str(indexed), "--n", "2", "--seed", "0"]
        )
        assert result.exit_code == 0, result.output
        assert "2 files validated" in result.output


def test_validate_empty_index_exits_1(tmp_path: Path) -> None:
    from fitsq.store import Store

    path = tmp_path / "empty.duckdb"
    Store(path).close()
    result = runner.invoke(cli.app, ["validate", "--index", str(path)])
    assert result.exit_code == 1 and "empty" in result.output


def test_index_reports_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        client.put_object(Bucket=BUCKET, Key="cat/junk.fits", Body=b"junk" * 1000)
        monkeypatch.setattr(cli, "S3Reader", lambda **_kw: S3Reader(client))
        result = runner.invoke(cli.app, ["index", PREFIX, "--index", str(tmp_path / "i.duckdb")])
        assert result.exit_code == 0  # per-file failures never fail the crawl
        assert "failed 1" in result.output and "error" in result.output
