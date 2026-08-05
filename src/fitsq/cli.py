"""typer CLI: index, cone, region, sql, status, validate."""

from __future__ import annotations

import csv
import json
import random
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from . import indexer as idx
from .query import DEFAULT_ORDER, Index, QueryError, cone_moc, parse_angle, parse_stcs
from .s3io import S3Reader
from .store import DEFAULT_INDEX, FileRow, IndexMissing, SchemaMismatch, Store

app = typer.Typer(
    add_completion=False,
    help="Find which FITS catalog files in S3 cover a piece of sky, offline.",
)


class Format(StrEnum):
    text = "text"
    json = "json"
    csv = "csv"


IndexOpt = Annotated[Path, typer.Option("--index", envvar="FITSQ_INDEX", help="index location")]
AnonOpt = Annotated[bool, typer.Option("--anon", help="unsigned S3 access (public buckets)")]
FormatOpt = Annotated[Format, typer.Option("--format", help="output format")]


def _open_store(path: Path, *, read_only: bool) -> Store:
    try:
        return Store(path, read_only=read_only)
    except IndexMissing:
        typer.secho(f"no index at {path}; run `fitsq index <s3-prefix>` first", fg="red", err=True)
        raise typer.Exit(1) from None
    except SchemaMismatch as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from None


def _emit(rows: list[FileRow], fmt: Format) -> None:
    if fmt is Format.text:
        for row in rows:
            typer.echo(row.uri)
    elif fmt is Format.json:
        payload = [{"uri": r.uri, "nrows": r.nrows, "size": r.size} for r in rows]
        typer.echo(json.dumps(payload, indent=2))
    else:
        writer = csv.writer(sys.stdout)
        writer.writerow(["uri", "nrows", "size"])
        for row in rows:
            writer.writerow([row.uri, row.nrows, row.size])


def _index_order(store: Store) -> int:
    raw = store.get_meta("order")
    return int(raw) if raw is not None else DEFAULT_ORDER


def _run_query(index_path: Path, fmt: Format, build: Any) -> None:
    with _open_store(index_path, read_only=True) as store:
        index = Index(store)
        try:
            moc = build(_index_order(store))
        except QueryError as exc:
            typer.secho(str(exc), fg="red", err=True)
            raise typer.Exit(2) from None
        _emit(index.search(moc), fmt)


@app.command()
def index(
    prefix: Annotated[str, typer.Argument(help="s3://bucket/prefix/ to crawl")],
    index_path: IndexOpt = DEFAULT_INDEX,
    anon: AnonOpt = False,
    workers: Annotated[int, typer.Option("--workers", help="crawl parallelism")] = 16,
    sample_rows: Annotated[int, typer.Option("--sample-rows", help="rows per window")] = 50_000,
    samples: Annotated[int, typer.Option("--samples", help="windows per file")] = 3,
    order: Annotated[int, typer.Option("--order", help="MOC max_depth")] = DEFAULT_ORDER,
    dilate: Annotated[int, typer.Option("--dilate", help="border cells at max order")] = 1,
) -> None:
    """Crawl an S3 prefix and upsert per-file coverage into the index (resumable)."""
    opts = idx.CrawlOptions(
        workers=workers, sample_rows=sample_rows, samples=samples, order=order, dilate=dilate
    )
    reader = S3Reader(anon=anon)

    def progress(stats: idx.CrawlStats, elapsed: float) -> None:
        done = stats.indexed + stats.failed
        remaining = stats.total - stats.skipped - done
        eta = f"{remaining * elapsed / done:6.0f}s" if done else "   ...."
        typer.echo(
            f"\r{done + stats.skipped}/{stats.total} files "
            f"({stats.skipped} skipped, {stats.failed} failed) "
            f"{stats.bytes_read / 1e6:8.1f} MB read  ETA {eta}",
            nl=False,
            err=True,
        )

    with _open_store(index_path, read_only=False) as store:
        stats = idx.crawl(prefix, store, reader, opts, progress)
    typer.echo("", err=True)
    typer.echo(
        f"indexed {stats.indexed}, skipped {stats.skipped}, failed {stats.failed} "
        f"of {stats.total} files; {stats.bytes_read / 1e6:.1f} MB read"
    )
    for uri, error in stats.errors[:10]:
        typer.echo(f"  error {uri}: {error}", err=True)
    if len(stats.errors) > 10:
        typer.echo(f"  ... {len(stats.errors) - 10} more in the crawl_errors table", err=True)


# ignore_unknown_options so a negative declination ("-29.25") is not parsed as a flag.
@app.command(context_settings={"ignore_unknown_options": True})
def cone(
    ra: Annotated[float, typer.Argument(help="right ascension, degrees")],
    dec: Annotated[float, typer.Argument(help="declination, degrees")],
    radius: Annotated[str, typer.Argument(help="e.g. 30arcsec, 2arcmin, 0.5deg, or bare number")],
    index_path: IndexOpt = DEFAULT_INDEX,
    unit: Annotated[str, typer.Option("--unit", help="unit for a bare radius")] = "deg",
    fmt: FormatOpt = Format.text,
) -> None:
    """List files covering a cone."""
    _run_query(index_path, fmt, lambda order: cone_moc(ra, dec, parse_angle(radius, unit), order))


@app.command()
def region(
    stcs: Annotated[str, typer.Argument(help="STC-S: 'POLYGON ICRS ...' or 'CIRCLE ICRS ...'")],
    index_path: IndexOpt = DEFAULT_INDEX,
    fmt: FormatOpt = Format.text,
) -> None:
    """List files covering an STC-S region."""
    _run_query(index_path, fmt, lambda order: parse_stcs(stcs, order))


@app.command()
def sql(
    query: Annotated[str, typer.Argument(help="SQL against the index (read-only)")],
    index_path: IndexOpt = DEFAULT_INDEX,
) -> None:
    """Run SQL against the DuckDB index."""
    with _open_store(index_path, read_only=True) as store:
        names, rows = store.sql(query)
        writer = csv.writer(sys.stdout)
        writer.writerow(names)
        writer.writerows(rows)


@app.command()
def status(index_path: IndexOpt = DEFAULT_INDEX) -> None:
    """Show index summary: files, rows, size, last crawl."""
    with _open_store(index_path, read_only=True) as store:
        typer.echo(json.dumps(store.status(), indent=2))


@app.command()
def validate(
    index_path: IndexOpt = DEFAULT_INDEX,
    n: Annotated[int, typer.Option("--n", help="files to full-read")] = 10,
    anon: AnonOpt = False,
    seed: Annotated[int | None, typer.Option("--seed", help="deterministic file choice")] = None,
) -> None:
    """Full-read N random indexed files and assert true coverage ⊆ stored MOC."""
    with _open_store(index_path, read_only=True) as store:
        rows = store.file_rows()
        order = _index_order(store)
    if not rows:
        typer.secho("index is empty", fg="red", err=True)
        raise typer.Exit(1)
    violations = idx.validate(rows, S3Reader(anon=anon), order, n, random.Random(seed))
    checked = min(n, len(rows))
    if violations:
        for violation in violations:
            typer.secho(
                f"FAIL {violation.uri}: {violation.missing_cells} cells missing "
                f"({violation.reason})",
                fg="red",
            )
        typer.echo(f"{len(violations)}/{checked} files under-covered; raise --samples and re-index")
        raise typer.Exit(1)
    typer.secho(f"OK: {checked} files validated, sampled MOCs cover all rows", fg="green")


if __name__ == "__main__":  # pragma: no cover
    app()
