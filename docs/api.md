# API Reference

## eccospectra.core

### `SiteConfig`

```python
@dataclass
class SiteConfig:
    z_measurement: float = 3.0       # measurement height [m]
    z_canopy: float = 0.3            # canopy height [m]
    d: float | None = None           # displacement height [m]  (default: 2/3 z_canopy)
    z0: float | None = None          # roughness length [m]     (default: 0.1 z_canopy)
    latitude: float = 0.0
    longitude: float = 0.0
    sampling_freq: float = 20.0      # Hz
    averaging_period: float = 30.0   # minutes
```

**Properties**
- `z_eff` — effective measurement height above the displacement height: `z_measurement - d`

---

### `SpectralResult`

Dataclass returned by `process_interval`. Key fields:

| Field | Description |
|---|---|
| `timestamp_start`, `timestamp_end` | Interval bounds |
| `u_mean` | Mean streamwise wind [m/s] |
| `ustar` | Friction velocity [m/s] |
| `L` | Obukhov length [m] |
| `zL` | Stability parameter z/L |
| `H` | Sensible heat flux [W/m²] |
| `cov_wT`, `cov_wu`, `cov_wCO2`, `cov_wH2O` | Raw covariances |
| `freq` | Bin-centre frequencies [Hz] |
| `freq_nd` | Dimensionless frequency f = nz/U |
| `cosp_wT`, … | n·Co(n) for each flux |
| `ncosp_wT`, … | n·Co(n) / cov(w'x') |
| `spec_u`, `spec_v`, `spec_w`, `spec_T` | n·S(n) / σ² |
| `ogive_wT`, … | Cumulative cospectra (high→low) |
| `qc_flags` | Dict populated by `qc.run_qc` and corrections |

---

### `rotate_wind(u_raw, v_raw, w_raw)`

Double rotation so that `mean(v) = mean(w) = 0`.

**Returns** `(u_rot, v_rot, w_rot, wind_dir)`

---

### `compute_cospectrum(x, y, fs)`

One-sided cospectrum (real part of CSD) with Hamming windowing.

**Returns** `(freq, cospec)` — positive frequencies only, DC excluded.

---

### `compute_spectrum(x, fs)`

One-sided power spectral density. Wraps `compute_cospectrum(x, x, fs)`.

**Returns** `(freq, psd)`

---

### `log_bin(freq, spec, bins_per_decade=20)`

Block-average spectral estimates into logarithmically spaced bins.

**Returns** `(freq_bin, spec_bin)`

---

### `process_interval(u, v, w, T_sonic, co2, h2o, config, ...)`

Full cospectral analysis for one averaging window:
quality screen → rotate → detrend → FFT → log-bin → normalize → turbulence stats.

Returns a `SpectralResult`. Intervals with >5% NaN in wind data get
`qc_flags['too_many_nans'] = True` and empty arrays.

---

### `process_file(df, config, bins_per_decade=20)`

Process all averaging intervals in a DataFrame (Polars or pandas).  
Expects columns: `TIMESTAMP`, `Ux`, `Uy`, `Uz`, `T_SONIC`, `CO2_density`, `H2O_density`.

**Returns** `list[SpectralResult]`

---

## eccospectra.io

### `read_toa5(filepath, parse_dates=True, drop_diagnostics=False, max_diag_value=0.0)`

Read a Campbell Scientific TOA5 file.

**Returns** `(pl.DataFrame, dict)` — data and metadata.

---

### `scan_toa5_directory(directory, pattern="TOA5_*.dat", recursive=False)`

Fast directory scan returning file metadata without loading data.

**Returns** `list[dict]` sorted by `first_timestamp`. Each dict has:
`path`, `filename`, `size_mb`, `first_timestamp`, `last_timestamp`, `n_records`, `metadata`.

---

### `compile_toa5(directory, pattern="TOA5_*.dat", start_date=None, end_date=None, ...)`

Compile multiple TOA5 files into one DataFrame. Deduplicates overlapping timestamps
and detects data gaps.

**Returns** `(pl.DataFrame, dict)` — combined data and compilation metadata.

---

## eccospectra.corrections

### `InstrumentConfig`

```python
@dataclass
class InstrumentConfig:
    sonic_path_length: float = 0.10        # [m]
    irga_path_length: float = 0.154        # [m] IRGASON default
    sensor_separation_lateral: float = 0.0 # [m]
    sensor_separation_longitudinal: float = 0.0
    sensor_separation_vertical: float = 0.0
    tau_sonic_T: float = 0.0               # [s]
    tau_co2: float = 0.1                   # [s]
    tau_h2o: float = 0.1                   # [s]
    irga_type: str = 'open_path'
    model: str = 'IRGASON'
```

**Properties**
- `sensor_separation_total` — Euclidean distance between sonic and IRGA volumes [m]

---

### `ukde_despike(series, prob_threshold=1e-4, max_iter=10)`

Iterative UKDE despiking (Metzger et al. 2012). Suitable for arrays up to ~10 000 samples.

---

### `polars_ukde_despike(df, col_name, prob_threshold=1e-4)`

FFT-based KDE despiking for large Polars DataFrames. Single-pass, O(n log n).
Adds a `{col_name}_cleaned` column.

---

### `despike_dataframe(df, columns, prob_threshold=1e-4, max_iter=10, verbose=False)`

Apply UKDE despiking to multiple columns of a pandas or Polars DataFrame.

---

### Transfer functions

All accept `freq: np.ndarray` and return values in [0, 1].

| Function | Description |
|---|---|
| `tf_block_average(freq, averaging_period)` | Finite-window low-freq attenuation |
| `tf_linear_detrend(freq, averaging_period)` | Linear detrend low-freq attenuation |
| `tf_first_order_response(freq, tau)` | Sensor time-constant attenuation |
| `tf_sonic_line_averaging(freq, u_mean, path_length)` | Sonic path averaging |
| `tf_scalar_path_averaging(freq, u_mean, path_length)` | IRGA optical path averaging |
| `tf_sensor_separation(freq, u_mean, separation)` | Lateral separation attenuation |

---

### `combined_transfer_function(freq, u_mean, instrument, averaging_period=30.0, flux_type='wT')`

Product of all applicable transfer functions (Massman 2000).  
`flux_type` is one of `'wT'`, `'wu'`, `'wCO2'`, `'wH2O'`.

---

### `compute_spectral_correction_factor(freq, cospec_model, transfer_function)`

Massman (2000) correction factor: ratio of uncorrected to corrected integrated cospectrum.

---

### `apply_spectral_corrections(results, config, instrument, apply_high_freq=True, apply_low_freq=True, apply_wpl=False, method='massman')`

Apply spectral corrections to a list of `SpectralResult` objects.
Writes correction factors and corrected fluxes into `qc_flags`.

---

### `wpl_correction(cov_wCO2, cov_wH2O, cov_wT, T_mean, rho_a, rho_v, rho_c)`

Webb-Pearman-Leuning (1980) density correction for open-path gas analysers.

---

## eccospectra.qc

### `fit_inertial_slope(freq_nd, ncosp, f_range=(1.0, 8.0))`

Log-log power-law fit in the inertial subrange.

**Returns** `(slope, r_squared, intercept)`

Expected slopes in area-preserving (n·Co vs f) form:
- Cospectra: −4/3 ≈ −1.33
- Power spectra: −2/3 ≈ −0.67

---

### `classify_slope(slope, is_cospectrum=True)`

Classify slope as `'good'`, `'acceptable'`, `'suspect'`, or `'bad'`.

---

### `stationarity_test(w, x, fs, n_subwindows=6)`

Foken & Wichura (1996) stationarity test.

**Returns** `(relative_diff, quality_class)` where quality class is 1–4
(1 = <15% rel. diff., 4 = >50%).

---

### `run_qc(results, f_range_cospec=(1.0, 8.0), f_range_spec=(1.0, 8.0))`

Batch QC on a list of `SpectralResult` objects. Adds to each `qc_flags`:
`slope_wT`, `slope_class_wT`, …, `ustar_filter`.

---

## eccospectra.plotting

### `plot_cospectra(results, show_kaimal=True, show_inertial=True)`

2×2 grid of normalized cospectra (w'T', w'u', w'CO₂', w'H₂O') coloured by stability class.

### `plot_spectra(results, show_inertial=True)`

2×2 grid of normalized power spectra (u, v, w, T).

### `plot_ogive(results)`

1×4 grid of ogives (cumulative cospectra, high→low frequency).

### `plot_summary_timeseries(results)`

Time series of key turbulence parameters for quick QC inspection.
