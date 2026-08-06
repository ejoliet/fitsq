# Implementation notes

One line per non-obvious decision. Grep-friendly. Newest last.

## Library API facts (verified against installed mocpy 0.20.0, not docs)

- 2026-08-05 Open Question 1 RESOLVED: `MOC.add_neighbours()` exists in mocpy 0.20.0, mutates in place and returns self. No corner-cone fallback needed. Context7 has no mocpy entry, so the API was verified by introspecting the installed wheel.
- 2026-08-05 `MOC.from_cone/from_polygon/from_box` must be called with KEYWORD args: a decorator wraps them as `(cls, lon, lat, **kwargs)`, so a positional `radius` raises "takes 3 positional arguments but 4 were given". Alternative rejected: passing positionally as the docstring examples imply.
- 2026-08-05 DEVIATION: `MOC.difference(other)` is NOT set subtraction for disjoint operands in mocpy 0.20 — `MOC.from_string("3/0-3").difference(MOC.from_string("3/100-103"))` returns EMPTY instead of `2/0`. It only agrees with subtraction when the operands overlap. `validate` therefore uses `indexer.uncovered()` = `inner.intersection(outer.complement())`, which is correct in both cases. Using `difference` would have made the sampling-adequacy gate pass unconditionally (silently useless). Regression test: `test_uncovered_handles_disjoint_mocs`.
- 2026-08-05 MOC persistence uses `to_string(format="json")` / `from_string(value, format="json")` (round-trips exactly). Chosen over FITS serialization to keep the column human-inspectable via `fitsq sql`.

## Storage

- 2026-08-05 `at` is a DuckDB reserved word, so `crawl_errors.at` is quoted in the DDL. Column name kept as the README specifies; ad-hoc SQL must write `SELECT "at" FROM crawl_errors`.
- 2026-08-05 Upsert is DELETE+INSERT inside an explicit transaction rather than `ON CONFLICT DO UPDATE`: it also has to replace the file's `headers` rows, so one transaction covers both tables. Alternative rejected: ON CONFLICT (would still need a separate headers delete).
- 2026-08-05 DuckDB writes happen only on the main thread; the ThreadPool workers just fetch and build MOCs and hand results back. Avoids relying on connection thread-safety.

## Query

- 2026-08-05 The bbox prefilter derives its box from the query MOC's BOUNDING CONE (`barycenter()` + `largest_distance_from_coo_to_vertices()`), not from the input geometry's vertices. Vertex boxes are not conservative for polygons on a sphere (great-circle edges bulge poleward of their endpoints), which would under-return. The bounding cone is always a superset, and it works uniformly for cone/polygon/box.
- 2026-08-05 RA half-width uses `radius / cos(worst_dec)` where `worst_dec = |dec| + radius`, and bails out to "no RA filter" when the box touches a pole, radius >= 90 deg, worst_dec >= 89 deg, or the half-width >= 180 deg. The naive `asin(sin r / cos dec_center)` is tighter but under-covers near the poles. `test_bounding_box_ra_halfwidth_never_under_covers_a_cone` scans a dec/radius grid asserting the box never clips the exact value.
- 2026-08-05 Files whose own bbox wraps RA=0 are kept unconditionally by the prefilter (`ra_wraps OR ...`) instead of splitting their interval. ~0.15 deg tiles make this a negligible number of extra MOC intersections.
- 2026-08-05 Per-file bbox RA interval = complement of the largest angular gap between sorted RAs. Detects wrap correctly without special-casing, e.g. {359.5, 359.9, 0.5} -> [359.5, 0.5] wraps=True.

## CLI

- 2026-08-05 `cone` sets `context_settings={"ignore_unknown_options": True}` so a negative declination (`fitsq cone 268.77 -29.25 30arcsec`, straight from the README) is not parsed as an option cluster by click. Alternative rejected: requiring `--` before the positionals (breaks the documented UX). A typo'd flag still fails on arity, so the safety loss is small.
- 2026-08-05 `region` validates the STC-S shape token BEFORE parsing numbers, so `UNION ICRS (...)` reports "unsupported STC-S construct" rather than "non-numeric coordinate".
- 2026-08-05 Query commands exit 2 on a bad radius/region (usage error), 1 on a missing or mismatched index, 0 with empty output when nothing matches — per the README's interface contract.

## Coordinate frames (added 2026-08-05)

- 2026-08-05 `cone --frame icrs|galactic|fk5|fk4` and honoured STC-S frame tokens. All input is converted to ICRS in `query.to_icrs()` before any MOC is built, because MOCs are ICRS by the Space MOC standard. Galactic l/b was previously impossible to express.
- 2026-08-05 Converting only the cone centre / polygon vertices is mathematically sufficient: frame changes among ICRS/FK5/FK4/galactic are rotations, and a rotation maps great circles to great circles, so polygon edges and cone radii are preserved. No re-tessellation. Asserted by `test_galactic_and_icrs_cones_agree_exactly` and `test_galactic_polygon_converts_all_vertices`.
- 2026-08-05 BUG FIXED: `parse_stcs` previously accepted an `FK5`/`J2000` frame token and then **ignored** it, silently treating the coordinates as ICRS. Now the token selects the frame; `GALACTIC` input is converted. FK5/J2000 still resolve to fk5 (~25 mas from ICRS, negligible against 6.9' cells) but are no longer conflated by accident. Regression test: `test_stcs_honours_galactic_frame`.
- 2026-08-05 `cone`'s positional args renamed ra/dec -> lon/lat since they are frame-dependent. Help text names both interpretations.

## Filename encoding (discovered 2026-08-05)

- 2026-08-05 DISCOVERY: `cat-<N>.fits` where N is the HEALPix **nside 512 / order 9 NESTED** pixel index of the tile — computed in **GALACTIC** coordinates. Verified: suffix == galactic order-9 nested pixel of the measured centre for 2495/2495 files; the same test in ICRS matches 0/2495. Row-level check on 4 files x 3000 rows: 100% of rows inside their own filename cell, zero spill into neighbours. The README previously asserted the opposite ("filenames are NOT HEALPix pixels, ruled out nest+ring orders 8-10") — that probe evidently only tried equatorial coordinates.
- 2026-08-05 Consequence not yet exploited: per-file coverage is derivable from the filename with ZERO data reads, exactly (one cell, dilation unnecessary), reducing the crawl from ~11.1 GB to a listing. Deliberately NOT implemented yet — it changes the architecture the spec describes, and the convention is reverse-engineered rather than documented by the producer, so it needs a decision on trust plus `validate` as the standing guard.
- 2026-08-05 Cross-check of the "uncovered coordinate" question: l=3.45, b=-0.27 is galactic order-9 pixel 1157513, and `cat-1157513.fits` is absent from the bucket — an independent confirmation of the footprint-edge finding below.

## Name-derived indexing (implemented 2026-08-05, now the default)

- 2026-08-05 `fitsq index` derives coverage from the filename by default (`--from-names`); `--sample` keeps the row-sampling crawl for cross-checking. Requested explicitly after the naming discovery was confirmed.
- 2026-08-05 DECISION: the index is stored in GALACTIC coordinates (`meta.moc_frame`), not ICRS. Rationale: the named cell is a galactic HEALPix cell, so storing galactic makes name-derived coverage *exact* — the MOC is literally `{"9":[pix]}`, no rotation, no dilation, no approximation. Storing ICRS instead would need the rotated cell's polygon, and mocpy 0.20 offers no usable way to get cell vertices: `get_boundaries()` requires networkx and is marked "not stable", and `cdshealpix` is not a mocpy dependency (mocpy bundles its own Rust healpix). The alternatives were a new dependency or hand-rolled HEALPix boundary maths, both worse than moving the frame.
- 2026-08-05 Consequence: queries convert into the index frame (`query.to_index_frame`) instead of into ICRS, and sampled rows (ICRS in the files) are converted on the way in. Correct because frame changes here are rotations: cones keep their radius, polygons keep their great-circle edges.
- 2026-08-05 Schema bumped to 2: `ra_min/ra_max/dec_min/dec_max/ra_wraps` renamed to `lon_*`/`lat_*`/`lon_wraps`, since those numbers are galactic now and leaving them named ra/dec would be a lie to anyone using `fitsq sql`. Cheap to do pre-release; the store already refuses a mismatched index with a rebuild message.
- 2026-08-05 Name-derived mode still reads each header (~8.6 KB/file, ~21 MB total) for nrows/row_bytes/data_offset and the `headers` table. Kept because `nrows` appears in query output and `status`, and 21 MB is ~500x less than the 11.1 GB sampling crawl. A header that fails to parse does NOT drop the file: coverage needs no bytes, so the file is indexed with nrows=0 and the error is recorded.
- 2026-08-05 An unparseable filename is a hard per-file failure in name mode, never a silent fallback to sampling — the whole point of the guard is that a change in the producer's naming surfaces loudly. `validate` (row-level) is the second guard; `test_validate_catches_a_broken_naming_convention` proves it catches a file named for the wrong cell.
- 2026-08-05 Synthetic test data now generates rows *inside* a named cell (`synth.cell_patch`) via `MOC.contains_lonlat` rejection sampling, so test filenames and contents agree the way the real bucket's do. Before this, the CLI fixture's arbitrary names made `validate` fail correctly — the guard caught the test's own inconsistency.

## DuckDB write throughput (found while timing the first real name-derived crawl)

- 2026-08-05 The first full name-derived crawl took 10m04s wall / 8m32s CPU for 2495 files despite reading only 57.5 MB — CPU-bound on the index writes, not the network. Cause: DuckDB 1.5.5 executes row-wise parameterised INSERTs at ~3.9 ms/row (measured, linear, identical in-memory and on-disk, ~250 rows/s). With ~107k header cards that is the entire runtime.
- 2026-08-05 Measured alternatives for 8000 rows: `executemany` 31s, one giant multi-row `INSERT ... VALUES` 10.4s, **CSV staging + `read_csv` 0.053s** (~200x). `Store.upsert_files` now writes a batch through a temporary CSV; `crawl` buffers `WRITE_BATCH = 250` files per transaction. Result: 10m04s -> 1m00s wall, 8m32s -> 16s CPU. The test suite also went from ~50s to ~15s.
- 2026-08-05 Two CSV traps, both caught by tests rather than reasoning: (1) DuckDB's `read_csv` maps an empty field to NULL *even when quoted*, which silently turned empty FITS card values into NULL — fixed with `nullstr=['__fitsq_null__']`, a sentinel no 80-char card can contain; (2) `moc_json` is pretty-printed and contains newlines inside a quoted field, which defeats the CSV sniffer — fixed with `strict_mode=false`, verified to round-trip the multi-line value byte-for-byte rather than by mutating the stored JSON.
- 2026-08-05 Batch granularity is the resume granularity: a killed crawl redoes at most 250 files, since resume keys off the stored ETag.

## Corpus footprint (measured 2026-08-05, all 2495 tiles)

- 2026-08-05 Probed every file with a 500-row sample (~60 KB each, 205 MB total, zero errors) to map tile centres. Result: the corpus is TWO disjoint fields, and the split is exactly the file-size split — 294 bulge tiles at l [-0.794, +1.671], b [-2.014, +0.377] are the 322 MB-1.2 GB files; 2201 southern tiles at l [-61.4, -35.1], b [-76.7, -63.8] are the 0-4.3 MB files. Nothing in between.
- 2026-08-05 `cat-1113599.fits` is centred at l=1.4077, b=-1.2686 (ICRS 268.4683, -28.3813), NOT at l=3.45, b=-0.27 as assumed in a query request — those differ by 2.27 deg. No tile covers l=3.45, b=-0.27: it is ~1.8 deg past the bulge field's longitude edge, nearest centre 1.94 deg away (`cat-1157294.fits`). A cone there returning empty is correct behaviour, not a lookup failure.
- 2026-08-05 Reconnaissance trick worth keeping: tiny 500-row samples are useless for MOC coverage but plenty for locating a tile centre, making a whole-corpus survey cost 205 MB instead of 11 GB.

## Tooling

- 2026-08-05 `[tool.mypy]` pins no `python_version`: numpy's bundled stubs use 3.12+ `type` statement syntax, so pinning 3.11 makes mypy fail inside numpy before checking our code. The package still declares `requires-python = ">=3.11"`.
- 2026-08-05 The e2e suite is split: the cheap cone check reads ~14 MB, and the full-read validate gate is behind `-m slow` (~535 MB). Both need `FITSQ_E2E=1`.

## Verified on the real bucket (2026-08-05)

- `cat-1113533.fits`: nrows=5875592, row_bytes=91, size=534686400, bbox ra [268.7017, 268.8491] dec [-29.3271, -29.1756] — matches the README's ground-truth table (~5.9M rows, 91 B/row, ~535 MB, ~0.15 deg tile).
- Cone (268.77, -29.25, 30arcsec) returns `cat-1113533.fits`; the same cone 10 deg away returns nothing.
- Validate gate passes: 13.7 MB of samples produced a MOC covering all 5.9M rows as confirmed by a 548 MB full read.
