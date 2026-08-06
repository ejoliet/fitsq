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
| Tile size | one order-9 HEALPix cell (~0.115° side) | see below |
| **Filenames ARE HEALPix pixels** | `cat-<pix>.fits` where `<pix>` is the **nside 512 (order 9) NESTED** index of the tile, **in GALACTIC coordinates** | coverage is derivable from the filename alone — verified 2495/2495 |
| Corpus | **2495 files, 204.9 GB total** | — |
| File size | **bimodal: median 3.5 MB (~40k rows); 291 files 400 MB–1.2 GB; max 13.2M rows** | see below |

### Filename → coverage (discovered 2026-08-05)

`cat-1113599.fits` is order-9 NESTED HEALPix pixel `1113599` **in galactic coordinates**. Evidence:

- suffix == galactic order-9 NESTED pixel of the measured tile centre for **2495/2495** files (ICRS: 0/2495)
- 4 files × 3000 rows: **100%** of rows fall inside their own filename cell, no spill into neighbours
- pixel 1113599's cell centre (l=1.4063, b=−1.2684) matches the measured tile centre (l=1.4077, b=−1.2686) to 0.0014°

```python
from mocpy import MOC; import astropy.units as u
from astropy.coordinates import SkyCoord
g = SkyCoord(ra=268.4683*u.deg, dec=-28.3813*u.deg).galactic
int(MOC.from_lonlat(g.l, g.b, max_norder=9).flatten()[0])   # -> 1113599
```

**This is how the index is built.** `fitsq index` derives coverage from the filename by default (`--from-names`): exact, one cell, no dilation, and no row reads — only ~8.6 KB of header per file for the row counts. The sampling crawl remains available as `fitsq index --sample` for cross-checking.

Because the convention is reverse-engineered rather than documented by the producer, two guards are wired in:

- a filename that does not parse is a **failure**, never a silent fallback — the file is left out of the index and reported, so a change in naming surfaces immediately;
- `fitsq validate` re-reads the rows and fails if a stored MOC does not cover them, which is exactly what a file named for the wrong cell would look like.

### Sky footprint (all 2495 tiles probed 2026-08-05)

The corpus is **two disjoint fields**, and the split coincides exactly with the file-size split:

| Field | Tiles | Galactic extent | Size / rows per file |
|---|---|---|---|
| Bulge (GBTDS-like) | 294 | l [−0.794, +1.671], b [−2.014, +0.377] | 322 MB – 1.2 GB, 3.5–13.2 M rows |
| Southern | 2201 | l [−61.4, −35.1], b [−76.7, −63.8] | 0–4.3 MB, 0–35 k rows |

Nothing lies between them. Useful consequence for galactic-coordinate queries: **the bulge field stops at l ≈ +1.67**, so a cone at, say, `l=3.45, b=−0.27` is ~1.8° beyond the coverage edge and correctly returns nothing — the nearest tile centre is 1.94° away (`cat-1157294.fits`). Cross-checked via the naming rule: that position is order-9 galactic pixel `1157513`, and `cat-1157513.fits` does not exist in the bucket.

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
- **Coverage comes from the filename** by default — exact, no dilation. `--sample` falls back to reading rows, where **dilation** guards against sparse sources missed by sampling: one HEALPix cell border at max order.
- **The index is stored in galactic coordinates** (`meta.moc_frame`), because that is the frame the tiling is defined in. Queries in any supported frame are rotated into it, which is lossless. The `files` bbox columns are `lon_*`/`lat_*` for that reason.
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
#    Default: exact coverage from filenames, headers only (~21 MB, seconds).
uv run fitsq index s3://stpubdata/roman/nexus/soc_simulations/input_catalogs/ --anon

#    Cross-check by sampling rows instead (~11 GB, the original path)
uv run fitsq index s3://stpubdata/roman/nexus/soc_simulations/input_catalogs/ --anon --sample

# 2. Query — instant, offline
uv run fitsq cone 268.77 -29.25 30arcsec
uv run fitsq cone 268.77 -29.25 0.05 --unit deg --format json
uv run fitsq region "POLYGON ICRS 268.7 -29.3 268.9 -29.3 268.9 -29.1 268.7 -29.1"

# Galactic l/b input — same sky, same answer
uv run fitsq cone 1.4077 -1.2686 1arcmin --frame galactic
uv run fitsq region "CIRCLE GALACTIC 1.4077 -1.2686 0.02"
uv run fitsq sql "SELECT count(*), sum(nrows) FROM files"
uv run fitsq status
uv run fitsq validate --n 10 --anon
```

## Configuration Reference

| Env / flag | Type | Default | Commands | Purpose |
|---|---|---|---|---|
| `FITSQ_INDEX` / `--index` | path | `~/.cache/fitsq/index.duckdb` | all | index location |
| `--anon` | flag | off | `index`, `validate` | unsigned S3 (public buckets); else boto3 default chain |
| `--from-names` / `--sample` | flag | `--from-names` | `index` | exact coverage from `cat-<pix>.fits` names, or by sampling rows |
| `--workers` | int | 16 | `index` | crawl parallelism |
| `--sample-rows` | int | 50000 | `index` | rows per sample window (`--sample` only) |
| `--samples` | int | 3 | `index` | windows per file (`--sample` only) |
| `--order` | int | 9 | `index` | MOC max_depth / HEALPix order of the tiling |
| `--dilate` | int | 1 | `index` | border cells at max order (`--sample` only; name-derived coverage is exact) |
| `--format` | enum | `text` | `cone`, `region` | `text` \| `json` \| `csv` |
| `--unit` | str | `deg` | `cone` | unit for a bare numeric radius |
| `--frame` | enum | `icrs` | `cone` | `icrs` \| `galactic` \| `fk5` \| `fk4` — frame of the input longitude/latitude |
| `--n` | int | 10 | `validate` | files to full-read |
| `--seed` | int | none | `validate` | deterministic file choice |

No secrets anywhere. Signed access uses the standard AWS credential chain only.

## Interface Contract

Exit codes: **0** success (including zero matches), **1** index missing / empty / schema mismatch / validate violation, **2** bad usage (unparseable radius, unsupported region).

### `fitsq index <s3-prefix>`
Crawl + upsert. Skips URIs whose ETag matches the stored one. Prints progress (`n/total, MB read, ETA`) to stderr. Exit 0 even with per-file failures; failures logged to `crawl_errors` and summarized (first 10 shown).

Coverage comes from filenames by default and from sampled rows under `--sample`. In the default mode:

- an unparseable filename is a **failure** — the file is not indexed, and the summary warns that the naming convention may have changed;
- an unreadable *header* is not fatal: coverage is exact regardless, so the file is indexed with `nrows = 0` and the problem is reported.

### `fitsq cone <lon> <lat> <radius>`
Radius accepts `30arcsec`, `2arcmin`, `0.5deg`, or a bare float with `--unit`. Output: matching `s3://` URIs, one per line (text) or `{uri, nrows, size}` (json/csv). Negative latitudes work as written — the command sets `ignore_unknown_options` so click does not read `-29.25` as a flag.

`<lon> <lat>` are ICRS ra/dec by default, or galactic `l`/`b` with `--frame galactic` (also `fk5`, `fk4`). Coordinates are converted to ICRS before the MOC is built, because MOCs are ICRS by the Space MOC standard. Frame changes among these systems are rotations, so a cone's radius and a polygon's great-circle edges survive unchanged — converting the centre or the vertices is sufficient.

### `fitsq region "<STC-S string>"`
v1 supports `POLYGON <frame> lon lat ...` and `CIRCLE <frame> lon lat r`. The frame token is **required and honoured**: `ICRS`, `FK5`/`J2000`, `FK4`/`B1950`, `GALACTIC`/`GAL`. Other STC-S constructs are rejected with a clear error.

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
  -- bbox prefilter, in the index frame (meta 'moc_frame'), degrees
  lon_min DOUBLE, lon_max DOUBLE, lat_min DOUBLE, lat_max DOUBLE,
  lon_wraps BOOLEAN,             -- box crosses lon=0
  moc_json TEXT,                 -- mocpy to_string(format="json"), index frame
  indexed_at TIMESTAMP
);
CREATE TABLE headers (uri TEXT, hdu INT, card_key TEXT, card_value TEXT, card_comment TEXT);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE crawl_errors (uri TEXT, error TEXT, "at" TIMESTAMP);
```

`meta` carries `schema_version`, `prefix`, `moc_frame` (`galactic`), `coverage` (`filename` or `sampled`), `order`, the sampling parameters, and `last_crawl`.

`meta.schema_version` is `2`; a mismatch refuses to open with a rebuild message. Version 2 renamed the bbox columns from `ra_*`/`dec_*`, because the stored box and MOC are in the index frame, not ICRS.

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
| `validate` passes (true coverage ⊆ stored MOC) | ✅ name-derived index: `--n 3` full-read on the real bucket, zero violations. Also ✅ for the sampling path: 13.7 MB sampled, confirmed by a 548 MB full read |
| Query latency < 50 ms warm | ✅ median 10.5 ms, p95 12.7 ms (5k-file index); 283-file 2° cone in 29.1 ms |
| Index ≤ 20 MB for ~5k files | ✅ 1.85 MB |
| Coverage ≥ 80%, `ruff` + `mypy --strict` clean | ✅ 95.8%, both clean, 61 passed / 1 skipped |
| Resumable crawl | ✅ rerun skips unchanged ETags with zero data reads; changed ETag re-indexes |
| Full crawl of the real bucket | ✅ all **2495 files, 0 errors, 57.5 MB read, 1m00s** (`--from-names`); 2.0 MB index over 2.23 billion catalog rows |

`cat-1113533.fits` as indexed: `nrows=5875592`, `row_bytes=91`, `size=534686400`, bbox ra [268.7017, 268.8491] dec [−29.3271, −29.1756] — matches the probed ground truth.

## Next Steps

1. Build the index whenever you need it — it is now a one-minute operation, so it need not be scheduled: `fitsq index <prefix> --anon`. Commit nothing containing the index; it is a cache artifact (`*.duckdb` is gitignored).
2. Run `fitsq validate --n 10 --anon` periodically (it full-reads whole files, up to 1.2 GB each) to keep the naming convention under watch; if clean, freeze v1.
3. Decide v2 scope (row retrieval) based on actual usage.
