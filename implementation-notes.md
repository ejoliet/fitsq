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

## Tooling

- 2026-08-05 `[tool.mypy]` pins no `python_version`: numpy's bundled stubs use 3.12+ `type` statement syntax, so pinning 3.11 makes mypy fail inside numpy before checking our code. The package still declares `requires-python = ">=3.11"`.
- 2026-08-05 The e2e suite is split: the cheap cone check reads ~14 MB, and the full-read validate gate is behind `-m slow` (~535 MB). Both need `FITSQ_E2E=1`.

## Verified on the real bucket (2026-08-05)

- `cat-1113533.fits`: nrows=5875592, row_bytes=91, size=534686400, bbox ra [268.7017, 268.8491] dec [-29.3271, -29.1756] — matches the README's ground-truth table (~5.9M rows, 91 B/row, ~535 MB, ~0.15 deg tile).
- Cone (268.77, -29.25, 30arcsec) returns `cat-1113533.fits`; the same cone 10 deg away returns nothing.
- Validate gate passes: 13.7 MB of samples produced a MOC covering all 5.9M rows as confirmed by a 548 MB full read.
