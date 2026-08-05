# fitsq

**File-discovery query layer for FITS source catalogs in S3.** Builds a local MOC-per-file spatial index by sampling rows via Range-GETs, then answers cone / region queries instantly and offline: *which files cover this piece of sky?*

> Type A (spec-only) README — an agent implements from this document. No code exists yet.

---

## Purpose

- **Problem**: Few thousand FITS bintable catalogs (~5.9M rows, 91 B/row, ~535 MB each) in `s3://stpubdata/roman/nexus/soc_simulations/input_catalogs/`. Headers carry no coverage metadata. No manifest. Finding files that cover a sky region currently means scanning data.
- **Solution**: One-time sampling crawl builds a few-MB `index.duckdb` (per-file MOC + header keywords). Queries run locally, sub-millisecond, no S3 access at query time.
- **Who benefits**: Emmanuel + colleagues. Index file is shareable — onboarding for a second user is "download one file".

### Verified ground truth (probed 2026-07)

| Fact | Value | Consequence |
|---|---|---|
| Bucket | `stpubdata`, public (`--no-sign-request`) | anonymous access default |
| Table layout | 12 cols: `source_id` (K), `ra`/`dec`/8 fluxes (D), `type` (3A); NAXIS1=91 | fixed numpy big-endian dtype |
| Primary header | empty (SIMPLE/BITPIX/NAXIS=0/EXTEND) | header index is trivial EAV |
| Row order | random within tile; head/tail 50k-row samples give identical footprint | sampling is sufficient; chunk map useless |
| Tile size | ~0.15° squares; straddles order-8 HEALPix boundaries | filenames are NOT HEALPix pixels (ruled out nest+ring, orders 8–10) |

---

## Architecture

```
INDEX (once, resumable)                    QUERY (local, offline)
────────────────────────                   ────────────────────────
list_objects_v2 (paginated)                cone/region args
  │ per file (ThreadPool, N=16)              │
  ├─ Range-GET header blocks (2880B units,   ├─ MOC.from_cone / from_polygon / from_box
  │   scan for END) → nrows, row_bytes,      ├─ bbox prefilter (SQL on files table)
  │   data_offset, column dtype              ├─ moc.intersection(file_moc) != empty
  ├─ Range-GET 3 row samples                 │   (all file MOCs loaded once, cached in proc)
  │   (head / middle / tail, 50k rows each)  └─ print s3:// URIs (text | json | csv)
  ├─ parse ra/dec via numpy '>f8'
  ├─ MOC.from_lonlat(order=9) → dilate 1 cell
  └─ upsert row into index.duckdb
```

- **Index is the product.** `index.duckdb` (~few MB) contains everything; queries never touch S3.
- **Dilation** guards against sparse sources missed by sampling: one HEALPix cell border at max order (~6.9′ at order 9 vs 9′ tiles — generous).
- **Resumable**: files already in the index (same URI + same ETag) are skipped; re-running `index` is the incremental update path.

## Recommended Stack

| Layer | Chosen | Why | Rejected |
|---|---|---|---|
| MOC / HEALPix | `mocpy` ≥ 0.17 (verified 0.20 API: `from_lonlat`, `from_cone`, `from_polygon`, `from_box`, set ops, JSON/FITS serialize) | Rust core, one dep covers build + query side | `healpy` (heavier, no MOC ops), `astropy-healpix` (no MOC) |
| S3 access | `boto3` + `ThreadPoolExecutor` | Range-GETs are IO-bound; threads suffice; `UNSIGNED` config for public bucket | `s3fs/fsspec` (extra layer, no benefit for explicit ranges), `aioboto3` (async complexity unjustified) |
| Index store | `duckdb` | Single file, SQL surface for header/metadata queries, Emmanuel-standard | SQLite (no columnar analytics), bare Parquet (no upsert) |
| FITS header parse | manual 80-char card scan of 2880 B blocks | ~30 lines; avoids pulling astropy into v1 | `astropy.io.fits` (fine, but v1 needs only header cards + dtype math) |
| Binary row parse | `numpy` structured dtype from TFORMn (`K→'>i8'`, `D→'>f8'`, `nA→f'S{n}'`) | zero-copy `frombuffer` | astropy Table (allocates full table) |
| CLI | `typer` | subcommands + help for free | argparse (boilerplate), click (typer wraps it) |
| Runner | `uv` / `uvx` | zero-friction: `uvx --from fitsq fitsq ...`; PEP 723 script fallback | pip venv (more steps) |

> 💡 Build agent: call Context7 (`resolve-library-id` → `get-library-docs`) for mocpy and duckdb before writing code — verify `add_neighbours`/dilation API name at build time.

## Repository Layout

```
fitsq/
├── README.md
├── pyproject.toml            # [project.scripts] fitsq = "fitsq.cli:app"; deps: mocpy, duckdb, boto3, numpy, typer
├── src/fitsq/
│   ├── __init__.py
│   ├── cli.py                # typer app: index, cone, region, sql, status, validate
│   ├── fits_lite.py          # header card scan, data_offset calc, TFORM→numpy dtype
│   ├── s3io.py               # unsigned/signed client, ranged_get(uri, start, end), list_fits(prefix)
│   ├── indexer.py            # crawl orchestration, sampling, MOC build, upsert, resume
│   ├── store.py              # duckdb schema, migrations, upsert, query helpers
│   └── query.py              # cone/region → MOC → intersect → URIs
└── tests/
    ├── test_fits_lite.py     # synthetic header blocks incl. multi-block headers
    ├── test_store.py
    ├── test_query.py         # synthetic MOCs, edge: cone across RA=0, poles
    └── test_indexer.py       # moto-mocked S3 with tiny synthetic FITS
```

## Quick Start (target UX)

```bash
# 1. Build the index (one-time, ~1–2 h from laptop; resumable — rerun to continue/update)
uvx --from fitsq fitsq index s3://stpubdata/roman/nexus/soc_simulations/input_catalogs/ --anon

# 2. Query — instant, offline
fitsq cone 268.77 -29.25 30arcsec
fitsq cone 268.77 -29.25 0.05 --unit deg --format json
fitsq region "POLYGON ICRS 268.7 -29.3 268.9 -29.3 268.9 -29.1 268.7 -29.1"
fitsq sql "SELECT count(*), sum(nrows) FROM files"
fitsq status          # files indexed, total rows, index size, last crawl
```

## Configuration Reference

| Env / flag | Type | Default | Required | Purpose |
|---|---|---|---|---|
| `FITSQ_INDEX` / `--index` | path | `~/.cache/fitsq/index.duckdb` | no | index location |
| `--anon` | flag | off | no | unsigned S3 (public buckets); else boto3 default chain |
| `--workers` | int | 16 | no | crawl parallelism |
| `--sample-rows` | int | 50000 | no | rows per sample window |
| `--samples` | int | 3 | no | windows per file (head/middle/tail; >3 → evenly spaced) |
| `--order` | int | 9 | no | MOC max_depth |
| `--dilate` | int | 1 | no | border cells at max order (0 disables) |
| `--format` | enum | `text` | no | `text` \| `json` \| `csv` output |

No secrets anywhere. Signed access uses the standard AWS credential chain only.

## Interface Contract

### `fitsq index <s3-prefix>`
Crawl + upsert. Skips URIs whose ETag matches the stored one. Prints progress (`n/total, MB read, ETA`). Exit 0 even with per-file failures; failures logged to `crawl_errors` table and summarized.

### `fitsq cone <ra> <dec> <radius>`
Radius accepts `30arcsec`, `2arcmin`, `0.5deg`, or bare float with `--unit` (default deg). Output: matching `s3://` URIs, one per line (text) or objects with `{uri, nrows, size}` (json/csv). Exit 1 if index missing, 0 with empty output if no match.

### `fitsq region "<STC-S string>"`
v1 supports `POLYGON ICRS lon lat ...` and `CIRCLE ICRS lon lat r`. Maps to `MOC.from_polygon` / `MOC.from_cone`. Reject other STC-S constructs with a clear error.

### `fitsq sql "<query>"`
Pass-through to the DuckDB index (read-only connection). Tables documented below.

### `fitsq validate [--n 10]`
Full-reads N random indexed files, rebuilds true MOCs, asserts `true_moc ⊆ stored_moc`. Reports any violation with the file URI. This is the sampling-adequacy gate.

### DuckDB schema

```sql
CREATE TABLE files (
  uri TEXT PRIMARY KEY, etag TEXT, size BIGINT,
  nrows BIGINT, row_bytes INT, data_offset BIGINT,
  ra_min DOUBLE, ra_max DOUBLE, dec_min DOUBLE, dec_max DOUBLE,  -- bbox prefilter; ra wrap flag
  ra_wraps BOOLEAN,
  moc_json TEXT,                 -- mocpy JSON serialization
  indexed_at TIMESTAMP
);
CREATE TABLE headers (uri TEXT, hdu INT, card_key TEXT, card_value TEXT, card_comment TEXT);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);   -- schema_version, prefix, sample params
CREATE TABLE crawl_errors (uri TEXT, error TEXT, at TIMESTAMP);
```

## Error Handling

| Condition | Behavior |
|---|---|
| S3 throttling / transient | boto3 retries (adaptive, max 5); then log to `crawl_errors`, continue |
| File smaller than expected / truncated header | log + skip; never abort crawl |
| Non-BINTABLE HDU1 or missing ra/dec columns | log + skip with reason |
| Cone crossing RA=0 or poles | must work — covered by tests (MOC handles it; bbox prefilter must respect `ra_wraps`) |
| Index schema mismatch | refuse with message: rebuild or migrate |

## Testing

- `pytest`; S3 mocked with `moto`; synthetic FITS built in-memory (correct 2880 B padding, big-endian rows).
- Property test: random cones vs brute-force point-in-circle over synthetic catalogs — file list must match exactly (index may over-return only if dilation > 0, never under-return).
- `ruff` + `mypy --strict` clean.

## Non-Goals (v1)

- Row retrieval / source extraction (v2: stream candidate files, numpy filter, Parquet out)
- Crossmatch (if needed later: HATS/LSDB conversion, not custom code)
- Any server: TAP, UWS, MCP, HTTP (v3 candidate; note IPAC outside-activity gate before anything leaves work scope)
- Parquet shadow of catalog data
- Windows support

## Open Questions

1. mocpy dilation API: `add_neighbours` availability/name in current release — verify via Context7 at build time; fallback = union of `from_cone` per bbox corner + edges.
2. MOC order 9 + 1-cell dilation is the default; if `validate` ever fails, bump `--samples` to 5 before touching order.
3. `middle` sample window position when NAXIS2 < 3 × sample-rows: read whole table (files that small are cheap anyway).

## Agent Build Instructions

> Implement end-to-end from this README. Resolve Open Question 1 first (Context7). Constraints: Python 3.11+, typed signatures, `set -euo pipefail` mindset, no secrets, tests never hit real S3.

| Phase | Deliverable | Done when |
|---|---|---|
| 0 | Scaffold, pyproject, ruff/mypy CI | `make lint` passes |
| 1 | `fits_lite` + `s3io` | unit tests pass (synthetic headers, moto) |
| 2 | `store` + `indexer` | crawl of moto bucket produces valid index; resume works |
| 3 | `query` + `cli` | cone/region/sql/status against test index correct incl. RA-wrap |
| 4 | `validate` command | subset-assertion runs against synthetic full reads |
| 5 | Smoke test vs real bucket (opt-in, `FITSQ_E2E=1`) | `fitsq cone 268.77 -29.25 30arcsec` returns `cat-1113533.fits` |

### Acceptance Criteria

- [ ] `uvx --from . fitsq index …/input_catalogs/ --anon` completes resumably on the real bucket
- [ ] Cone at (268.77, −29.25) returns `cat-1113533.fits`; cone 10° away in empty sky returns nothing
- [ ] `fitsq validate --n 10` passes on the real index
- [ ] Query latency < 50 ms with a warm process, index ≤ 20 MB for ~5k files
- [ ] Coverage ≥ 80%, `ruff` + `mypy --strict` clean

## Next Steps

1. Resolve Open Question 1 (mocpy dilation) via Context7.
2. Build Phases 0–4 against synthetic data.
3. Run the real crawl from laptop (evening job); commit nothing containing the index — it's a cache artifact.
4. Run `fitsq validate --n 10`; if clean, freeze v1.
5. Decide v2 scope (row retrieval) based on actual usage.
