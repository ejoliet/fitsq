from __future__ import annotations

import csv
import io
import json
import re
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

#: named the way the real bucket does: galactic order-9 HEALPix pixel
TARGET_PIXEL = 1113533
PIXELS = (TARGET_PIXEL, 1157294)
TILES = tuple(f"cat-{pixel}.fits" for pixel in PIXELS)
TARGET = f"{PREFIX}cat-{TARGET_PIXEL}.fits"


def _populate(client: object) -> None:
    for pixel in PIXELS:
        ra, dec = synth.cell_patch(pixel, n=400, seed=pixel % 1000)
        client.put_object(  # type: ignore[attr-defined]
            Bucket=BUCKET, Key=f"cat/cat-{pixel}.fits", Body=synth.catalog(ra, dec)
        )


@pytest.fixture
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A real index built from a mocked bucket, via the `index` command itself."""
    index_path = tmp_path / "index.duckdb"
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        _populate(client)
        monkeypatch.setattr(cli, "S3Reader", lambda **_kw: S3Reader(client))
        result = runner.invoke(
            cli.app,
            ["index", PREFIX, "--index", str(index_path), "--sample-rows", "200", "--workers", "2"],
        )
        assert result.exit_code == 0, result.output
        assert "indexed 2" in result.output
        assert "exact, from filenames" in result.output  # default mode
        yield index_path


def test_cone_text_json_csv(indexed: Path) -> None:
    ra, dec = synth.cell_centre_icrs(TARGET_PIXEL)
    args = ["cone", f"{ra:.6f}", f"{dec:.6f}", "30arcsec", "--index", str(indexed)]
    text = runner.invoke(cli.app, args)
    assert text.exit_code == 0
    assert text.output.strip() == TARGET

    as_json = runner.invoke(cli.app, [*args, "--format", "json"])
    payload = json.loads(as_json.output)
    assert payload[0]["uri"] == TARGET and payload[0]["nrows"] == 400

    as_csv = runner.invoke(cli.app, [*args, "--format", "csv"])
    rows = list(csv.reader(io.StringIO(as_csv.output)))
    assert rows[0] == ["uri", "nrows", "size"] and rows[1][1] == "400"


def test_cone_bare_radius_with_unit(indexed: Path) -> None:
    ra, dec = synth.cell_centre_icrs(TARGET_PIXEL)
    result = runner.invoke(
        cli.app,
        ["cone", f"{ra:.6f}", f"{dec:.6f}", "0.05", "--unit", "deg", "--index", str(indexed)],
    )
    assert result.exit_code == 0 and TARGET in result.output


def test_cone_galactic_frame(indexed: Path) -> None:
    """The tile is a galactic cell, so l/b input must find it directly."""
    lon, lat = synth.cell_centre_galactic(TARGET_PIXEL)
    result = runner.invoke(
        cli.app,
        [
            "cone",
            f"{lon:.6f}",
            f"{lat:.6f}",
            "1arcmin",
            "--frame",
            "galactic",
            "--index",
            str(indexed),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == TARGET


def test_region_galactic_circle(indexed: Path) -> None:
    lon, lat = synth.cell_centre_galactic(TARGET_PIXEL)
    result = runner.invoke(
        cli.app, ["region", f"CIRCLE GALACTIC {lon:.6f} {lat:.6f} 0.02", "--index", str(indexed)]
    )
    assert result.exit_code == 0, result.output
    assert TARGET in result.output


def test_cone_bad_frame_exits_2(indexed: Path) -> None:
    result = runner.invoke(
        cli.app, ["cone", "1", "2", "1deg", "--frame", "supergalactic", "--index", str(indexed)]
    )
    assert result.exit_code == 2


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
    ra, dec = synth.cell_centre_icrs(TARGET_PIXEL)
    lo_ra, hi_ra = ra - 0.1, ra + 0.1
    lo_dec, hi_dec = dec - 0.1, dec + 0.1
    box = (
        f"{lo_ra:.4f} {lo_dec:.4f} {hi_ra:.4f} {lo_dec:.4f} "
        f"{hi_ra:.4f} {hi_dec:.4f} {lo_ra:.4f} {hi_dec:.4f}"
    )
    ok = runner.invoke(cli.app, ["region", f"POLYGON ICRS {box}", "--index", str(indexed)])
    assert ok.exit_code == 0, ok.output
    assert TARGET in ok.output
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
    assert payload["coverage"] == "filename" and payload["moc_frame"] == "galactic"


def test_validate_command(indexed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """validate re-reads the rows: it is the guard on the naming convention."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        _populate(client)
        monkeypatch.setattr(cli, "S3Reader", lambda **_kw: S3Reader(client))
        result = runner.invoke(
            cli.app, ["validate", "--index", str(indexed), "--n", "2", "--seed", "0"]
        )
        assert result.exit_code == 0, result.output
        assert "2 files validated, stored MOCs cover all rows" in result.output


def test_index_sample_mode_reads_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--sample keeps the original row-sampling path available for cross-checking."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        _populate(client)
        monkeypatch.setattr(cli, "S3Reader", lambda **_kw: S3Reader(client))
        sampled = runner.invoke(
            cli.app,
            ["index", PREFIX, "--index", str(tmp_path / "s.duckdb"), "--sample", "--workers", "2"],
        )
        assert sampled.exit_code == 0, sampled.output
        assert "sampled rows" in sampled.output

        names = runner.invoke(
            cli.app, ["index", PREFIX, "--index", str(tmp_path / "n.duckdb"), "--workers", "2"]
        )
        assert "exact, from filenames" in names.output

        def mb(output: str) -> float:
            match = re.search(r"of \d+ files; ([\d.]+) MB read", output)
            assert match is not None, output
            return float(match.group(1))

        # name-derived reads headers only, so strictly less than sampling rows
        assert mb(sampled.output) > mb(names.output)


def test_index_warns_when_a_name_breaks_the_convention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        _populate(client)
        ra, dec = synth.cell_patch(TARGET_PIXEL, n=20, seed=3)
        client.put_object(Bucket=BUCKET, Key="cat/tile-A7.fits", Body=synth.catalog(ra, dec))
        monkeypatch.setattr(cli, "S3Reader", lambda **_kw: S3Reader(client))
        result = runner.invoke(
            cli.app, ["index", PREFIX, "--index", str(tmp_path / "i.duckdb"), "--workers", "2"]
        )
        assert result.exit_code == 0  # per-file failures never fail the crawl
        assert "failed 1" in result.output
        assert "naming convention may have changed" in result.output


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
