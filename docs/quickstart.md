# Quickstart

## Installation

```bash
pip install eccospectra
```

## 1. Configure your site

`SiteConfig` holds the tower geometry and sampling parameters needed throughout the pipeline.

```python
import eccospectra as ec

config = ec.SiteConfig(
    z_measurement=3.0,     # measurement height above ground [m]
    z_canopy=0.3,          # canopy height [m]  (d = 2/3 hc, z0 = 0.1 hc by default)
    sampling_freq=20.0,    # data acquisition rate [Hz]
    averaging_period=30.0, # flux averaging window [minutes]
)
```

## 2. Load data

### Single TOA5 file

```python
df, meta = ec.read_toa5("TOA5_mysite_2023_06_10.dat")
print(df.head())
print(meta['station_id'])
```

### Multi-file compilation (e.g. a full day)

```python
df, meta = ec.compile_toa5(
    "/data/raw/2023-06-10/",
    pattern="TOA5_*.dat",
    start_date=datetime(2023, 6, 10, 0, 0),
    end_date=datetime(2023, 6, 11, 0, 0),
)
print(f"Loaded {len(df):,} rows from {meta['n_files']} files")
```

## 3. Optional: despike the raw time series

```python
from eccospectra.corrections import despike_dataframe

df_clean = despike_dataframe(
    df,
    columns=['Ux', 'Uy', 'Uz', 'T_SONIC', 'CO2_density', 'H2O_density'],
    prob_threshold=1e-4,
    verbose=True,
)
```

## 4. Compute spectra and cospectra

```python
results = ec.process_file(df_clean, config)
print(f"Processed {len(results)} intervals")
```

Each element of `results` is a `SpectralResult` containing:
- Mean meteorological quantities: `u_mean`, `ustar`, `L`, `zL`, `H`
- Raw covariances: `cov_wT`, `cov_wu`, `cov_wCO2`, `cov_wH2O`
- Frequency array: `freq` [Hz], `freq_nd` (dimensionless f = nz/U)
- Cospectra: `cosp_wT`, `cosp_wu`, `cosp_wCO2`, `cosp_wH2O` (n·Co)
- Normalized cospectra: `ncosp_wT`, … (n·Co / cov)
- Power spectra: `spec_u`, `spec_v`, `spec_w`, `spec_T` (n·S / σ²)
- Ogives: `ogive_wT`, `ogive_wu`, `ogive_wCO2`, `ogive_wH2O`

## 5. Quality control

```python
from eccospectra.qc import run_qc

results = run_qc(results)

# Inspect QC flags on the first interval
res = results[0]
print(res.qc_flags['slope_class_wT'])   # 'good', 'acceptable', 'suspect', or 'bad'
print(res.qc_flags['ustar_filter'])     # True if u* < 0.1 m/s
```

## 6. Spectral corrections (optional)

```python
from eccospectra.corrections import InstrumentConfig, apply_spectral_corrections

instrument = InstrumentConfig()   # defaults to IRGASON
results = apply_spectral_corrections(
    results, config, instrument,
    apply_high_freq=True,
    apply_low_freq=True,
    apply_wpl=False,     # set True if using open-path CO2/H2O fluxes
)
```

## 7. Plot

```python
from eccospectra.plotting import plot_cospectra, plot_spectra, plot_ogive

fig_co  = plot_cospectra(results, show_kaimal=True)
fig_sp  = plot_spectra(results)
fig_og  = plot_ogive(results)

fig_co.savefig("cospectra.pdf", dpi=150)
```

## Minimal single-interval example

If you have NumPy arrays rather than a DataFrame:

```python
import numpy as np
import eccospectra as ec

config = ec.SiteConfig(z_measurement=3.0, z_canopy=0.3,
                       sampling_freq=20.0, averaging_period=30.0)

rng = np.random.default_rng(0)
n = int(30 * 60 * 20)   # 30 min at 20 Hz
u   = 5.0 + rng.normal(0, 0.5, n)
v   = rng.normal(0, 0.3, n)
w   = rng.normal(0, 0.15, n)
T   = 20.0 + rng.normal(0, 0.5, n)
co2 = 700.0 + rng.normal(0, 5.0, n)
h2o = 10.0  + rng.normal(0, 0.5, n)

result = ec.process_interval(u, v, w, T, co2, h2o, config)
print(f"u*  = {result.ustar:.3f} m/s")
print(f"H   = {result.H:.1f} W/m²")
print(f"z/L = {result.zL:.3f}")
```
