# TaylorSwift Roadmap

Goals: faster processing of long high-frequency records, a single coherent API
instead of two parallel stacks, and results that users can trust because they
are validated against reference implementations.

Phases are ordered by return on effort. Phase 0 items are bugs that affect
users today; later phases build on a fixed and benchmarked base.

---

## Phase 0 — Fix what is broken today

These are small, high-impact correctness fixes.

1. **Undeclared `statsmodels` dependency breaks the default pipelines.**
   `despike_med_mod` (`src/TaylorSwift/despike.py`) imports `statsmodels.api`,
   and both `run_irga` and `run_kh20` call it through `_despike_columns` by
   default — so the flux pipelines crash on a clean install. Either declare
   `statsmodels` (as a core or `[pipelines]` extra with a clear error message),
   or replace the RLM fit with a NumPy/SciPy robust regression so the
   dependency disappears entirely.

2. **`numpy>=1.24` lower bound is wrong.** `cospectra.py` uses `np.trapezoid`
   (added in NumPy 2.0), so `compute_spectral_correction_factor` raises
   `AttributeError` on any NumPy 1.x install that the metadata claims to
   support. Bump to `numpy>=2.0` or add a `trapezoid = getattr(np, "trapezoid",
   np.trapz)` shim. Audit the other lower bounds the same way — `polars>=0.19`
   predates APIs used in `io.py` (`str.strip_chars`, modern `read_csv`
   keywords).

3. **README quick start does not run.** `from tswift.qc import run_qc` fails
   twice over: `tswift` is an alias (not importable from), and there is no
   `qc` module — the function lives in `TaylorSwift.data_quality`. The module
   table also references `qc` and `ec_polars`, which were renamed to
   `data_quality` and `compat`. Fix the examples, and export `run_qc` from
   `__init__.py` alongside the other QC symbols.

4. **Metadata cleanup.** `pyproject.toml` points at
   `github.com/paultgriffiths/TaylorSwift`; the README says
   `inkenbrandt/TaylorSwift`. Pick one. Add Python 3.13/3.14 classifiers and CI
   runners (local dev is already on 3.14 while CI stops at 3.12), and install
   `.[dev]` into the project venv so `pytest` runs locally.

5. **Physics spot-check: `horst_analytical_correction`.** The docstring gives
   `CF = 1 / (1 + (2π f τ)^α)` but the code computes `1 / (1 - (…)^α)`, which
   diverges as the term approaches 1 and goes negative beyond it (masked by
   `max(cf, 1.0)`). Verify against Horst (1997) — the standard form is
   `CF = 1 + (2π n_m τ_eff)^α` — and add a test pinning known values.

---

## Phase 1 — Speed

Establish measurement first, then attack the known hot spots.

1. **Add a benchmark suite** (`pytest-benchmark` or `asv`) with two fixtures: a
   synthetic 30-minute 20 Hz interval (36 000 samples) and a synthetic
   multi-day file. Every optimization below should land with a before/after
   number, and CI should catch regressions.

2. **`process_file` interval slicing is quadratic.** Each interval runs
   `df.filter()` over the *entire* frame (`core.py`), so a month of 20 Hz data
   (~52 M rows, ~1 500 intervals) is scanned ~1 500 times. Replace the edge
   loop with a single pass: `group_by_dynamic` on TIMESTAMP, or
   `np.searchsorted` on the already-sorted timestamps to get slice bounds.
   This is the single largest speed win available.

3. **Vectorize the lag-search covariances.** `_compute_fluxes`
   (`pipelines.py`) calls `calc_max_covariance` for 21 velocity–scalar pairs,
   and each call loops over 21 lags with a full O(N) `calc_cov` pass
   (mask + means recomputed every time) — ~450 array traversals per interval.
   Compute all lags at once via FFT cross-correlation, or precompute the
   finite mask and running sums so each lag is O(1) after one pass.

4. **Vectorize `spike_detection`** (`despike.py`): it is a Python `for` loop
   over every sample with `np.mean`/`np.std` per window. Replace with
   `np.lib.stride_tricks.sliding_window_view` or cumulative-sum rolling
   statistics — this is a 100–1000× win on realistic inputs.

5. **Parallelize across intervals and files.** Intervals are embarrassingly
   parallel: add `n_jobs` to `process_file` (via `concurrent.futures`).
   `compile_toa5` reads files serially; read with `pl.scan_csv` and let Polars
   parallelize, or thread the per-file reads.

6. **Cheaper despiking defaults.** `despike_med_mod` fits an RLM
   (`maxiter=300`) over the whole series per column. Benchmark it against the
   rolling-quantile and MAD filters already in the library; make the fast one
   the pipeline default and keep RLM opt-in.

7. **Minor FFT wins.** Swap `np.fft.rfft` for `scipy.fft.rfft(..., workers=-1)`
   in `process_interval`; the batched-6-signal design in `core.py` is already
   good.

---

## Phase 2 — Consolidation and API coherence

The library currently ships two parallel stacks: the spectral stack
(`core`/`cospectra`/`config.SiteConfig`) and the legacy CalcFlux stack
(`pipelines`/`covariance`/`rotations`/`thermo`/`compat`). Physics is duplicated
— WPL exists as both `corrections.wpl_correction` and
`corrections.webb_pearman_leuning`, Massman corrections live in both
`cospectra.py` and `pipelines.py`, and rotation is implemented twice in
`rotations.py`. Duplicated physics means fixes land in one copy and not the
other.

1. **Single source of truth per correction.** Merge the WPL, Massman, and
   rotation implementations; the pipelines and the spectral stack call the same
   functions. Keep `compat.CalcFlux` as a thin shim over the unified code.

2. **Split `cospectra.py` (866 lines).** Move `SpectralResult` into
   `results.py` (which today holds only the barely-used `FluxResult`), transfer
   functions into a `transfer_functions.py`, and `apply_spectral_corrections`
   into `corrections.py`. This also removes the current core → cospectra →
   corrections lazy-import tangle.

3. **Tabular results export.** `process_file` returns `list[SpectralResult]`,
   which users cannot save or join with met data. Add
   `results_to_dataframe(results)` producing a tidy frame of scalar stats
   (u*, L, z/L, H, covariances, QC flags, correction factors) plus an optional
   long-format spectra table, and `to_parquet`/`to_csv` convenience wrappers.

4. **Configurable column mapping in `process_file`.** Column names
   (`Ux`, `T_SONIC`, `CO2_density`, …) are hardcoded; accept a `column_map`
   the way the pipelines accept `rename_map`, so non-Campbell files work
   without manual renaming.

5. **Typing and linting.** Complete type hints, ship a `py.typed` marker, add
   `mypy` (or pyright) and a `[tool.ruff]` config to `pyproject.toml` — CI
   runs ruff today with no project configuration.

---

## Phase 3 — Functionality growth

In rough priority order for a working micrometeorologist:

1. **Vickers & Mahrt (1997) raw-data screening** integrated into
   `process_interval`: spike count, amplitude resolution, dropouts, absolute
   limits, skewness/kurtosis — recorded as per-interval `qc_flags` instead of
   the current single NaN-fraction test.
2. **Planar-fit rotation** (Wilczak et al. 2001) as an alternative to double
   rotation, fitted over a multi-day window — important for sloped or
   heterogeneous sites.
3. **Time-lag optimization in the spectral stack.** The max-covariance lag
   search exists only in the legacy pipeline; expose it in
   `process_interval` for closed-path and separated-sensor setups.
4. **Flux uncertainty** (Finkelstein & Sims 2001 random error) per interval.
5. **More input formats**: LI-COR `.ghg` archives, generic CSV with a schema
   spec, and an EddyPro `full_output` reader (which also enables the
   validation work in Phase 4).
6. **A small CLI** (`taylorswift process <dir> --config site.toml`) so field
   techs can run standard processing without writing Python.
7. **Webb terms for closed-path IRGAs** (H₂O-only WPL, or point-by-point
   mixing-ratio conversion) — currently WPL is open-path only.

---

## Phase 4 — Effectiveness, validation, and release hygiene

1. **Validate against a reference implementation.** Process a public
   benchmark dataset (e.g., an AmeriFlux "gold file" site) through both
   TaylorSwift and EddyPro/EasyFlux, and document agreement for u*, H, LE, Fc
   and the correction factors. Turn the root-level `validate_cospectra.py` and
   `validate_process_file.py` scripts into pytest golden-file regression tests
   so agreement cannot silently drift.
2. **Property-based tests for the spectral core**: Parseval's identity (the
   integral of the raw cospectrum equals the covariance), ogive convergence to
   the covariance, and transfer functions bounded in (0, 1].
3. **Replace hardcoded physics constants**: `H = 1200.0 * cov_wT` in `core.py`
   assumes ρ·Cp = 1200 J m⁻³ K⁻¹ regardless of the actual temperature and
   pressure — compute ρ·Cp from measured T and P (the thermo module already
   has the pieces).
4. **Docs.** MkDocs scaffolding and `.readthedocs.yaml` exist; wire
   `mkdocstrings` API pages for every public module, convert the two example
   notebooks into a documented gallery, and add a "which despiker/correction
   should I use?" guide — the library has 6+ despiking methods with no
   guidance.
5. **Release automation.** One version source (currently duplicated in
   `pyproject.toml` and `__init__.py`), a tag-triggered PyPI publish workflow
   with trusted publishing, a CHANGELOG, and coverage reporting with a
   ratchet in CI.
