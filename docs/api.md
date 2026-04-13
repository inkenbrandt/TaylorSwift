# API Reference

## TaylorSwift.core

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

## TaylorSwift.corrections

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

## TaylorSwift.qc

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

## TaylorSwift.plotting

### `plot_cospectra(results, show_kaimal=True, show_inertial=True)`

2×2 grid of normalized cospectra (w'T', w'u', w'CO₂', w'H₂O') coloured by stability class.

### `plot_spectra(results, show_inertial=True)`

2×2 grid of normalized power spectra (u, v, w, T).

### `plot_ogive(results)`

1×4 grid of ogives (cumulative cospectra, high→low frequency).

### `plot_summary_timeseries(results)`

Time series of key turbulence parameters for quick QC inspection.

---

## TaylorSwift.constants

Physical constants, surface-type enumerations, default thresholds, and roughness helper
functions for eddy covariance calculations.

### Physical constants

| Name | Value | Units |
|---|---|---|
| `K_VON_KARMAN` | 0.40 | dimensionless |
| `G0` | 9.80665 | m s⁻² |
| `R_GAS` | 8.3144598 | J mol⁻¹ K⁻¹ |
| `SIGMA_SB` | 5.67×10⁻⁸ | W m⁻² K⁻⁴ |
| `CP_DRY_AIR` | 1004.67 | J kg⁻¹ K⁻¹ |
| `L_VAPORIZATION` | 2.501×10⁶ | J kg⁻¹ |
| `T_ZERO_C` | 273.15 | K |
| `P_REFERENCE` | 101.325 | kPa |

Dictionaries: `MOLAR_MASS`, `R_SPECIFIC`, `UNIT_CONVERSION`

---

### `SurfaceType` (IntEnum)

Surface roughness class for roughness / displacement height calculations.

Members: `CROP`, `GRASS`, `SHRUB`, `FOREST`, `BARELAND`, `WATER`, `ICE`, `URBAN`

Lookup tables keyed by `SurfaceType`: `ROUGHNESS_LENGTH`, `DISPLACEMENT_RATIO`

---

### `Hemisphere` (IntEnum)

`NORTH = 1`, `SOUTH = -1`, `EAST = 1`, `WEST = -1`

---

### `QualityThreshold`

Class-level dicts for default QC thresholds.

| Attribute | Description |
|---|---|
| `RN_THRESHOLD` | Stationarity thresholds (`high_quality`, `moderate_quality`, `low_quality`) |
| `ITC_THRESHOLD` | ITC thresholds |
| `WIND_DIRECTION` | Acceptable wind direction sectors (degrees) |
| `SIGNAL_STRENGTH` | Minimum signal strength for CO₂ and H₂O |
| `VALID_RANGE` | Physical plausibility ranges for each variable |

---

### `ProcessingConfig`

Default configuration for flux processing.

| Attribute | Description |
|---|---|
| `AVERAGING_INTERVAL` | 1800 s |
| `SUBINTERVAL` | 300 s |
| `FREQ_RESPONSE` | Low/high frequency cutoffs and number of frequency points |
| `DESPIKE` | Z-score threshold and window size for despiking |
| `ROTATION` | Max rotation angle, number of sectors for planar fit |
| `STORAGE` | Storage flux integration parameters |
| `DENSITY_CORRECTION` | WPL and density-correction flags |

---

### `ErrorCode` (IntEnum)

`SUCCESS`, `MISSING_DATA`, `RANGE_ERROR`, `QUALITY_ERROR`, `PROCESSING_ERROR`, `CONFIG_ERROR`

---

### `get_displacement_height(surface_type, canopy_height)`

Estimate displacement height *d = k_d × h* from the surface type and canopy height.

**Raises** `ValueError` if `canopy_height < 0`.

---

### `get_roughness_length(surface_type, canopy_height, custom_value=None)`

Return aerodynamic roughness length *z₀*.  For `CROP`, `GRASS`, and `SHRUB` uses
`0.15 × h`; all other types look up `ROUGHNESS_LENGTH`.  Pass `custom_value` to
bypass table/empirical estimates.

**Raises** `ValueError` if `canopy_height < 0`.

---

## TaylorSwift.data_quality

Data quality assessment following Foken *et al.* (2004, 2012).

### `QualityFlag` (IntEnum)

Nine-class Foken quality scheme: `CLASS_1` (highest) … `CLASS_9` (discard).

---

### `StabilityParameters`

Dataclass holding surface-layer stability variables for a 30-min averaging period.

| Field | Description | Units |
|---|---|---|
| `z` | Measurement height above displacement plane | m |
| `L` | Monin–Obukhov length | m |
| `u_star` | Friction velocity | m s⁻¹ |
| `sigma_w` | Std. dev. vertical wind | m s⁻¹ |
| `sigma_T` | Std. dev. air temperature | K |
| `T_star` | Temperature scale −w′T′/u★ | K |
| `latitude` | Site latitude | ° |

---

### `StationarityTest`

Dataclass of relative non-stationarity indices (Foken & Wichura 1996).

Fields: `RN_uw`, `RN_wT`, `RN_wq`, `RN_wc` (all dimensionless).

---

### `DataQuality`

```python
DataQuality(use_wind_direction=True)
```

| Method | Description |
|---|---|
| `assess_data_quality(stability, stationarity, wind_direction=None, flux_type='momentum')` | Comprehensive QC — returns dict with `overall_flag`, `stationarity_flag`, `itc_flag`, `wind_dir_flag`, `itc_measured`, `itc_modeled` |
| `get_quality_label(flag)` | Human-readable string for a numeric quality flag |

`flux_type` is one of `'momentum'`, `'heat'`, `'moisture'`, `'co2'`.

---

### `OutlierDetection`

Static methods for spike/outlier detection.

| Method | Description |
|---|---|
| `mad_outliers(data, threshold=3.5)` | Median Absolute Deviation modified Z-score; returns bool mask |
| `spike_detection(data, window_size=100, z_threshold=4.0)` | Sliding-window local z-score spike flagging; returns bool mask |

---

### `quality_filter(data, quality_flags, min_quality=3)`

Replace elements of `data` with `NaN` where `quality_flags > min_quality`.

**Returns** `np.ndarray` (float).

---

### `rolling_sigma_filter(df, value_col='Uz', time_col='TIMESTAMP', period='5s', sigma=3.0, ...)`

Rolling ±σ·std spike filter on a time-indexed Polars DataFrame.

| Parameter | Default | Description |
|---|---|---|
| `value_col` | `'Uz'` | Column to filter |
| `time_col` | `'TIMESTAMP'` | Datetime column for rolling index |
| `period` | `'5s'` | Time window (e.g. `'5s'`, `'30m'`) |
| `sigma` | `3.0` | Standard deviation threshold |
| `closed` | `'both'` | Window boundary inclusion |
| `output_col` | `None` | Output column name (default: `{value_col}_filtered`) |
| `keep_stats` | `True` | Keep or drop intermediate mean/std columns |
| `ensure_datetime` | `True` | Cast `time_col` to `pl.Datetime` |

**Returns** `pl.DataFrame` with added columns `{value_col}_roll_mean`, `{value_col}_roll_std`, and `{output_col}`.

---

## TaylorSwift.ec_polars

High-frequency eddy covariance flux processing pipeline with Polars/pandas
compatibility.  Transcribed from original Fortran/Visual Basic scripts by
Lawrence Hipps and Clayton Lewis (Utah State University).

### `CalcFlux`

```python
CalcFlux(**kwargs)
```

The primary class for computing sensible heat, latent heat, and momentum
fluxes from high-frequency sonic anemometer and gas analyser data.
All attributes have sensible defaults and may be overridden via `**kwargs`.

**Key configuration attributes**

| Attribute | Default | Description |
|---|---|---|
| `meter_type` | `'IRGASON'` | `'IRGASON'` or `'KH20'` |
| `UHeight` | 3.52 | Sonic measurement height (m) |
| `sonic_dir` | 225.0 | Sonic azimuth from true north (°) |
| `PathDist_U` | 0.0 | Sonic–hygrometer horizontal separation (m) |
| `lag` | 10 | Lag window (±samples) for covariance maximisation |
| `Rd`, `Rv`, `Cpd` | 287.05, 461.51, 1005.0 | Gas constants and heat capacities |

**Main processing methods**

| Method | Description |
|---|---|
| `runall(df)` | Full EC pipeline for KH-20 hygrometer data; returns `pd.Series` with 13 flux/diagnostic fields |
| `run_irga(df)` | Full EC pipeline for IRGASON/open-path IRGA data; returns `pd.Series` |

Both methods expect a time-indexed pandas `DataFrame` with columns
`Ux`, `Uy`, `Uz`, `Ts`, `Ta`, `Pr` (and `Ea`/`pV` as appropriate);
column names are harmonised by `renamedf`.

**Despiking methods**

| Method | Description |
|---|---|
| `despike_quart_filter(col, win=600, ...)` | Rolling inter-quantile range filter (default pipeline) |
| `despike_med_mod(col, win=800, ...)` | Median-filter + RLM robust regression spike removal |
| `despike_ewma_fb(col, span, delta)` | Forward–backward EWMA outlier removal |

**Pure utility methods**

| Method | Description |
|---|---|
| `convert_CtoK(T)` / `convert_KtoC(T)` | Temperature unit conversion |
| `calc_cov(p1, p2)` | Population covariance of two arrays |
| `calc_MSE(y)` | Mean squared deviation from mean |
| `calc_Es(T)` | Saturation vapour pressure (Pa) from temperature (K) |
| `tetens(T)` | Saturation vapour pressure (kPa) from air temperature (°C) |
| `calc_Td_dewpoint(E)` | Dew-point temperature (K) |
| `shadow_correction(Ux, Uy, Uz)` | CSAT3 transducer shadow correction |
| `coord_rotation(df)` | Double/triple wind-vector rotation |
| `calc_L(Ustr, Tsa, Uz_Ta)` | Monin–Obukhov length |
| `get_lag(x, y)` | Optimal lag between two signals |

**Polars/pandas compatibility helpers** (module-level)

| Helper | Description |
|---|---|
| `_to_pl_df(df)` | Convert pandas or Polars DataFrame to `pl.DataFrame` |
| `_to_same_type(original, pl_df)` | Return result as the same type as `original` |
| `_get_series(df, col)` | Extract a column as `pl.Series` from either DataFrame type |
| `_assign(df, **cols)` | Add/replace columns in either DataFrame type |
