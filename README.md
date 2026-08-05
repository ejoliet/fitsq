# fitsq

**File-discovery query layer for FITS source catalogs in S3.** Builds a local MOC-per-file spatial index by sampling rows via Range-GETs, then answers cone / region queries instantly and offline: *which files cover this piece of sky?*

> **Status: v1 implemented.** Phases 0–5 complete; `ruff` + `mypy --strict` clean; 61 tests passing, 95.8% coverage. Verified against the real bucket (see [Verification results](#verification-results)). Not yet run: the full ~2.5k-file production crawl.

---

## Purpose

- **Problem**: A few thousand FITS bintable catalogs in `s3://stpubdata/roman/nexus/soc_simulations/input_catalogs/`. Headers carry no coverage metadata. No manifest. Finding files that cover a sky region currently means scanning data.
- **Solution**: One-time sampling crawl builds a few-MB `index.duckdb` (per-file MOC + header keywords). Queries run locally in ~10 ms, no S3 access at query time.
- **Who benefits**: Emmanuel + colleagues. Index file is shareable — onboarding for a second user is "download one file".

### Verified ground truth (bucket probed 2026-07, full listing measured 2026-08-05)

| Fact | Value | Consequence |
|---|---|---|
| Bucket | `stpubdata`, public (`--no-sign-request`) | anonymous access default |
| Table layout | 12 cols: `source_id` (K), `ra`/`dec`/8 fluxes (D), `type` (3A); NAXIS1=91 | fixed numpy big-endian dtype |
| Primary header | empty (SIMPLE/BITPIX/NAXIS=0/EXTEND) | header index is trivial EAV |
| Row order | random within tile; head/tail 50k-row samples give identical footprint | sampling is sufficient; chunk map useless |
| Tile size | ~0.15° squares; straddles order-8 HEALPix boundaries | filenames are NOT HEALPix pixels (ruled out nest+ring, orders 8–10) |
| Corpus | **2495 files, 204.9 GB total** | — |
| File size | **bimodal: median 3.5 MB (~40k rows); 291 files 400 MB–1.2 GB; max 13.2M rows** | see below |

> ⚠️ **The corpus is bimodal, not uniform.** An earlier revision of this spec generalized `cat-1113533.fits` (5.88M rows, 535 MB) to every file. In reality **88% of files (2201) hold ≤ 150k rows**, i.e. fewer than `--samples × --sample-rows`. For those the indexer reads the **whole table** rather than sampling windows — the former "Open Question 3" path is the *common* case, and those files are indexed exactly rather than approximately. A full crawl at default settings reads **≈11.1 GB, 5.4% of the corpus**.

---

## Architecture

```
INDEX (once, resumable)                    QUERY (local, offline)
────────────────────────                   ────────────────────────
list_objects_v2 (paginated)                cone/region args
  │ per file (ThreadPool, N=16)              │
  ├─ Range-GET header blocks (2880B units,   ├─ MOC.from_cone / from_polygon
  │   scan for END) → nrows, row_bytes,      ├─ bounding-cone bbox prefilter
  │   data_offset, column dtype              │   (SQL WHERE on files table)
  ├─ Range-GET 3 row samples                 ├─ moc.intersection(file_moc) != empty
  │   (head / middle / tail, 50k rows each;  │   (file MOCs parsed once, cached per proc)
  │    whole table if it is smaller)         └─ print s3:// URIs (text | json | csv)
  ├─ parse ra/dec via numpy '>f8'
  ├─ MOC.from_lonlat(order=9) → dilate 1 cell
  └─ upsert row into index.duckdb
```

- **Index is the product.** `index.duckdb` (~2 MB for 5k files) contains everything; queries never touch S3.
- **Dilation** guards against sparse sources missed by sampling: one HEALPix cell border at max order (~6.9′ at order 9 vs 9′ tiles — generous).
- **Resumable**: files already in the index (same URI + same ETag) are skipped; re-running `index` is the incremental update path.

## Stack (as built)

| Layer | Chosen | Why |
|---|---|---|
| MOC / HEALPix | `mocpy` 0.20 | Rust core, one dep covers build + query side |
| S3 access | `boto3` + `ThreadPoolExecutor`, `UNSIGNED` for public buckets | Range-GETs are IO-bound; threads suffice |
| Index store | `duckdb` | Single file, SQL surface for header/metadata queries |
| FITS header parse | manual 80-char card scan of 2880 B blocks (`fits_lite`) | ~200 lines incl. dtype math; no astropy.io.fits |
| Binary row parse | `numpy` structured dtype from TFORMn (`K→'>i8'`, `D→'>f8'`, `nA→f'S{n}'`) | zero-copy `frombuffer` |
| CLI | `typer` | subcommands + help for free |
| Runner | `uv` / `uvx` | `uvx --from . fitsq ...` |

`astropy` is a direct dependency (units/coordinates on the query path), not just a mocpy transitive.

### mocpy 0.20 API notes (verified by introspecting the installed wheel — Context7 has no mocpy entry)

- `MOC.add_neighbours()` exists, mutates **in place**, returns self. Dilation needs no fallback.
- `MOC.from_cone` / `from_polygon` / `from_box` must be called with **keyword** arguments; a positional `radius` raises `takes 3 positional arguments but 4 were given`.
- ⚠️ **`MOC.difference()` is not set subtraction for disjoint operands.** `MOC.from_string("3/0-3").difference(MOC.from_string("3/100-103"))` returns an *empty* MOC instead of `2/0`. `indexer.uncovered()` uses `inner.intersection(outer.complement())` instead — using `difference` would have made `validate` pass unconditionally. Pinned by `test_uncovered_handles_disjoint_mocs`.

## Repository Layout

```
fitsq/
├── README.md
├── implementation-notes.md   # every non-obvious decision, dated
├── Makefile                  # make lint | test | cov | all
├── pyproject.toml            # [project.scripts] fitsq = "fitsq.cli:app"
├── uv.lock
├── src/fitsq/
│   ├── __init__.py
│   ├── cli.py                # typer app: index, cone, region, sql, status, validate
│   ├── fits_lite.py          # header card scan, data_offset calc, TFORM→numpy dtype
│   ├── s3io.py               # unsigned/signed client, ranged_get, list_fits
│   ├── indexer.py            # crawl, sampling, MOC build, upsert, resume, validate
│   ├── store.py              # duckdb schema, upsert, query helpers
│   └── query.py              # cone/region → MOC → bbox prefilter → intersect → URIs
└── tests/
    ├── synth.py              # synthetic FITS builder (real 91-byte row layout)
    ├── test_fits_lite.py     # cards, multi-block headers, TFORM, offsets
    ├── test_store.py
    ├── test_query.py         # STC-S, bbox safety scan, RA=0, poles, property test
    ├── test_indexer.py       # moto-mocked S3, resume, bad files, validate gate
    ├── test_cli.py           # all six commands end to end
    └── test_e2e.py           # opt-in, real bucket (FITSQ_E2E=1)
```

## Quick Start

```bash
uv sync

# 1. Build the index (one-time, resumable — rerun to continue/update)
uv run fitsq index s3://stpubdata/roman/nexus/soc_simulations/input_catalogs/ --anon

# 2. Query — instant, offline
uv run fitsq cone 268.77 -29.25 30arcsec
uv run fitsq cone 268.77 -29.25 0.05 --unit deg --format json
uv run fitsq region "POLYGON ICRS 268.7 -29.3 268.9 -29.3 268.9 -29.1 268.7 -29.1"
uv run fitsq sql "SELECT count(*), sum(nrows) FROM files"
uv run fitsq status
uv run fitsq validate --n 10 --anon
```

## Configuration Reference

| Env / flag | Type | Default | Commands | Purpose |
|---|---|---|---|---|
| `FITSQ_INDEX` / `--index` | path | `~/.cache/fitsq/index.duckdb` | all | index location |
| `--anon` | flag | off | `index`, `validate` | unsigned S3 (public buckets); else boto3 default chain |
| `--workers` | int | 16 | `index` | crawl parallelism |
| `--sample-rows` | int | 50000 | `index` | rows per sample window |
| `--samples` | int | 3 | `index` | windows per file (head/middle/tail; >3 → evenly spaced) |
| `--order` | int | 9 | `index` | MOC max_depth |
| `--dilate` | int | 1 | `index` | border cells at max order (0 disables) |
| `--format` | enum | `text` | `cone`, `region` | `text` \| `json` \| `csv` |
| `--unit` | str | `deg` | `cone` | unit for a bare numeric radius |
| `--n` | int | 10 | `validate` | files to full-read |
| `--seed` | int | none | `validate` | deterministic file choice |

No secrets anywhere. Signed access uses the standard AWS credential chain only.

## Interface Contract

Exit codes: **0** success (including zero matches), **1** index missing / empty / schema mismatch / validate violation, **2** bad usage (unparseable radius, unsupported region).

### `fitsq index <s3-prefix>`
Crawl + upsert. Skips URIs whose ETag matches the stored one. Prints progress (`n/total, MB read, ETA`) to stderr. Exit 0 even with per-file failures; failures logged to `crawl_errors` and summarized (first 10 shown).

### `fitsq cone <ra> <dec> <radius>`
Radius accepts `30arcsec`, `2arcmin`, `0.5deg`, or a bare float with `--unit`. Output: matching `s3://` URIs, one per line (text) or `{uri, nrows, size}` (json/csv). Negative declinations work as written — the command sets `ignore_unknown_options` so click does not read `-29.25` as a flag.

### `fitsq region "<STC-S string>"`
v1 supports `POLYGON <frame> lon lat ...` and `CIRCLE <frame> lon lat r`, where frame is `ICRS`/`FK5`/`J2000` and is **required**. Other STC-S constructs are rejected with a clear error.

### `fitsq sql "<query>"`
Pass-through to the DuckDB index (read-only connection), CSV out. Note `at` is a DuckDB reserved word: `SELECT uri, "at" FROM crawl_errors`.

### `fitsq status`
JSON summary: files, rows, catalog bytes, index bytes, last crawl, error count, and the crawl parameters the index was built with.

### `fitsq validate [--n 10]`
Full-reads N random indexed files, rebuilds true MOCs, asserts `true_moc ⊆ stored_moc`. Reports any violation with the file URI. This is the sampling-adequacy gate. Note this downloads whole files (up to 1.2 GB each).

### DuckDB schema

```sql
CREATE TABLE files (
  uri TEXT PRIMARY KEY, etag TEXT, size BIGINT,
  nrows BIGINT, row_bytes INT, data_offset BIGINT,
  ra_min DOUBLE, ra_max DOUBLE, dec_min DOUBLE, dec_max DOUBLE,  -- bbox prefilter
  ra_wraps BOOLEAN,                                              -- box crosses RA=0
  moc_json TEXT,                 -- mocpy to_string(format="json")
  indexed_at TIMESTAMP
);
CREATE TABLE headers (uri TEXT, hdu INT, card_key TEXT, card_value TEXT, card_comment TEXT);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);   -- schema_version, prefix, sample params
CREATE TABLE crawl_errors (uri TEXT, error TEXT, "at" TIMESTAMP);
```

`meta.schema_version` is `1`; a mismatch refuses to open with a rebuild message.

## Error Handling

| Condition | Behavior |
|---|---|
| S3 throttling / transient | boto3 retries (adaptive, max 5); then log to `crawl_errors`, continue |
| File smaller than expected / truncated header | log + skip; never abort crawl |
| Non-BINTABLE HDU1, missing ra/dec columns, empty table, all-NaN coords | log + skip with reason |
| Cone crossing RA=0 or poles | works — covered by tests; prefilter respects `ra_wraps` and drops the RA clause near poles |
| Index schema mismatch | refuse with message: rebuild |

### Why the prefilter uses a bounding cone

The bbox prefilter derives its box from the query MOC's **bounding cone** (`barycenter()` + `largest_distance_from_coo_to_vertices()`), not from the input geometry's vertices: on a sphere a polygon's great-circle edges bulge poleward of their endpoints, so a vertex box can under-return. RA half-width is `radius / cos(|dec| + radius)`, and the RA clause is dropped entirely near a pole or for very wide cones. `test_bounding_box_ra_halfwidth_never_under_covers_a_cone` scans a dec/radius grid asserting the box never clips the exact analytic value.

## Testing

```bash
make lint      # ruff check + ruff format --check + mypy --strict
make cov       # pytest with coverage, fails under 80%

FITSQ_E2E=1 uv run pytest tests/test_e2e.py -s          # real bucket, ~14 MB
FITSQ_E2E=1 uv run pytest tests/test_e2e.py -s -m slow  # adds a ~535 MB full read
```

- `pytest`; S3 mocked with `moto`; synthetic FITS built in-memory (correct 2880 B padding, big-endian rows, real 91-byte row layout).
- Property test: random cones vs brute-force point-in-circle over synthetic catalogs. **Deviation from the original spec**, which asked for an exact file-list match: order-9 cells are ~6.9′, so exactness at the cone boundary is quantization-flaky. The test instead asserts the two directions that matter — every file with a source inside 0.85 r **must** be returned (never under-return), and no file whose nearest source is beyond 1.3 r may be returned.
- Real-bucket tests are opt-in and never run in the default suite.

## Non-Goals (v1)

- Row retrieval / source extraction (v2: stream candidate files, numpy filter, Parquet out)
- Crossmatch (if needed later: HATS/LSDB conversion, not custom code)
- Any server: TAP, UWS, MCP, HTTP (v3 candidate; note IPAC outside-activity gate before anything leaves work scope)
- Parquet shadow of catalog data
- Windows support

## Resolved questions

1. ~~mocpy dilation API~~ → `add_neighbours()` present in 0.20.0, in-place. No fallback needed.
2. Order 9 + 1-cell dilation remains the default. If `validate` ever fails, raise `--samples` to 5 before touching order.
3. ~~`middle` window when NAXIS2 < 3 × sample-rows~~ → whole table is read. This turned out to be the common case (88% of files).

## Verification results

Measured 2026-08-05 on this implementation.

| Acceptance criterion | Result |
|---|---|
| Cone (268.77, −29.25) returns `cat-1113533.fits`; 10° away returns nothing | ✅ real bucket |
| `validate` passes (true coverage ⊆ stored MOC) | ✅ real file: 13.7 MB sampled, confirmed by a 548 MB full read, zero violations |
| Query latency < 50 ms warm | ✅ median 10.5 ms, p95 12.7 ms (5k-file index); 283-file 2° cone in 29.1 ms |
| Index ≤ 20 MB for ~5k files | ✅ 1.85 MB |
| Coverage ≥ 80%, `ruff` + `mypy --strict` clean | ✅ 95.8%, both clean, 61 passed / 1 skipped |
| Resumable crawl | ✅ rerun skips unchanged ETags with zero data reads; changed ETag re-indexes |
| `uvx --from . fitsq index …` completes on the real bucket | ⬜ **not run** — full 2495-file crawl pending |

`cat-1113533.fits` as indexed: `nrows=5875592`, `row_bytes=91`, `size=534686400`, bbox ra [268.7017, 268.8491] dec [−29.3271, −29.1756] — matches the probed ground truth.

## Next Steps

1. Run the real crawl from a laptop (evening job): ~2495 files, ≈11.1 GB read. Commit nothing containing the index — it is a cache artifact (`*.duckdb` is gitignored).
2. Run `fitsq validate --n 10` against the full index; if clean, freeze v1.
3. Decide v2 scope (row retrieval) based on actual usage.
