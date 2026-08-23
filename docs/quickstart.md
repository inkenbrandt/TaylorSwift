# Quickstart

This page walks the full pipeline: raw Campbell Scientific files in, corrected
and quality-flagged cospectra out. Every snippet is runnable as written.

## 1. Configure your site

`SiteConfig` holds the tower geometry, instrument geometry, and sampling
parameters used throughout the pipeline.

```python
import TaylorSwift as tswift

config = tswift.SiteConfig(
    z_measurement=3.0,      # measurement height above ground [m]
    z_canopy=0.3,           # canopy height [m]
    sampling_freq=20.0,     # data acquisition rate [Hz]
    averaging_period=30.0,  # flux averaging window [minutes]
)

print(config.d)      # displacement height, defaults to (2/3) * z_canopy
print(config.z0)     # roughness length, defaults to 0.1 * z_canopy
print(config.z_eff)  # effective height z - d, used for f = n*z/U
```

Defaults describe a Campbell Scientific IRGASON. See
[Configuration](guide/configuration.md) for every field.

## 2. Load data

=== "Single TOA5 file"

    ```python
    df, meta = tswift.read_toa5("TOA5_mysite_2023_06_10.dat")

    print(df.head())
    print(meta["station_id"])
    ```

=== "Multi-file compilation"

    ```python
    from datetime import datetime

    df, meta = tswift.compile_toa5(
        "/data/raw/2023-06-10/",
        pattern="TOA5_*.dat",
        start_date=datetime(2023, 6, 10),
        end_date=datetime(2023, 6, 11),
    )

    print(f"Loaded {len(df):,} rows from {meta['n_files']} files")
    ```

Both return a `(polars.DataFrame, metadata_dict)` pair. More detail in
[Reading data](guide/io.md).

## 3. Optional: despike the raw time series

```python
df_clean = tswift.despike_dataframe(
    df,
    columns=["Ux", "Uy", "Uz", "T_SONIC", "CO2_density", "H2O_density"],
    prob_threshold=1e-4,
    verbose=True,
)
```

!!! tip "Despiking is optional"
    `process_file` already runs Vickers & Mahrt raw-data screening internally,
    which *flags* bad intervals. Despiking *repairs* individual samples. See
    [Despiking](guide/despiking.md) for when each is appropriate.

## 4. Compute spectra and cospectra

```python
results = tswift.process_file(df_clean, config)

print(f"Processed {len(results)} intervals")
```

Each element is a [`SpectralResult`](api/results.md). The most-used fields:

| Group | Fields |
| --- | --- |
| Mean quantities | `u_mean`, `wind_dir`, `T_mean`, `ustar`, `L`, `zL`, `H` |
| Raw covariances | `cov_wT`, `cov_wu`, `cov_wCO2`, `cov_wH2O` |
| Frequency | `freq` [Hz], `freq_nd` (dimensionless $f = nz/U$) |
| Cospectra | `cosp_wT`, `cosp_wu`, `cosp_wCO2`, `cosp_wH2O` — that is $n \cdot Co(n)$ |
| Normalized cospectra | `ncosp_wT`, `ncosp_wu`, `ncosp_wCO2`, `ncosp_wH2O` |
| Power spectra | `spec_u`, `spec_v`, `spec_w`, `spec_T` — that is $n \cdot S(n) / \sigma^2$ |
| Ogives | `ogive_wT`, `ogive_wu`, `ogive_wCO2`, `ogive_wH2O` |
| Diagnostics | `qc_flags` |

## 5. Spectral corrections

```python
instrument = tswift.InstrumentConfig()  # alias of SiteConfig; defaults to IRGASON

results = tswift.apply_spectral_corrections(
    results,
    config,
    instrument,
    apply_high_freq=True,
    apply_low_freq=True,
    apply_wpl=False,  # set True for open-path CO2/H2O fluxes
)
```

!!! warning "WPL needs mean densities"
    `apply_wpl=True` requires `co2_mean`, `h2o_mean`, and `P_mean` on each
    result. Populate them first with
    `tswift.enrich_results_with_means(results, df, config)`. See
    [Spectral corrections](guide/corrections.md).

## 6. Quality control

```python
results = tswift.run_qc(results)

res = results[0]
print(res.qc_flags["slope_class_wT"])  # 'good' | 'acceptable' | 'suspect' | 'bad'
print(res.qc_flags["ustar_filter"])    # True when u* is below threshold
print(res.qc_flags["vm97_hard_flag"])  # True when raw-data screening failed hard
```

[Quality control](guide/quality-control.md) documents every flag.

## 7. Export

```python
table = tswift.results_to_dataframe(results)   # one row per interval
long = tswift.spectra_to_dataframe(results)    # long-format spectra

tswift.results_to_csv(results, "fluxes.csv")
tswift.results_to_parquet(results, "fluxes.parquet")
```

## 8. Plot

All three plotting helpers return a `(fig, axes)` tuple:

```python
from TaylorSwift.plotting import plot_cospectra, plot_spectra, plot_ogive

fig_co, axes_co = plot_cospectra(results, show_model=True, show_slope=True)
fig_sp, axes_sp = plot_spectra(results)
fig_og, axes_og = plot_ogive(results)

fig_co.savefig("cospectra.pdf", dpi=150)
```

Or let them write the file for you with `save_path=`:

```python
plot_cospectra(results, save_path="cospectra.png")
```

## Working from NumPy arrays

If you already have arrays rather than a DataFrame, call `process_interval`
directly for a single averaging window:

```python
import numpy as np
import TaylorSwift as tswift

config = tswift.SiteConfig(
    z_measurement=3.0, z_canopy=0.3, sampling_freq=20.0, averaging_period=30.0
)

rng = np.random.default_rng(0)
n = int(30 * 60 * 20)  # 30 min at 20 Hz

result = tswift.process_interval(
    u_raw=5.0 + rng.normal(0, 0.5, n),
    v_raw=rng.normal(0, 0.3, n),
    w_raw=rng.normal(0, 0.15, n),
    T_sonic=20.0 + rng.normal(0, 0.5, n),
    co2=700.0 + rng.normal(0, 5.0, n),
    h2o=10.0 + rng.normal(0, 0.5, n),
    config=config,
)

print(f"u*  = {result.ustar:.3f} m/s")
print(f"H   = {result.H:.1f} W/m²")
print(f"z/L = {result.zL:.3f}")
```

## Next steps

- [Configuration](guide/configuration.md) — every `SiteConfig` field
- [Raw-data screening](guide/screening.md) — what the `vm97_*` flags mean
- [Spectral corrections](guide/corrections.md) — the transfer-function stack
- [API Reference](api/index.md)
