# TaylorSwift

The name is in honor of physicist Sir Geoffrey Ingram Taylor. FFT-based (co)spectral analysis for eddy covariance time series.

`TaylorSwift` implements the standard micrometeorological workflow for computing power spectra and cospectra from high-frequency sonic anemometer and open-path gas analyser data, following Kaimal et al. (1972) conventions.

## Features

- **Spectral computation** — double-rotation, linear detrending, Hamming-windowed FFT, logarithmic frequency binning, area-preserving normalization
- **Spectral corrections** — block-average, linear-detrend, first-order sensor response, sonic path averaging, sensor separation (Massman 2000; Horst 1997)
- **Despiking** — iterative UKDE despiking for raw time series (Metzger et al. 2012); rolling IQR, median-RLM, and EWMA methods via `CalcFlux`
- **WPL density correction** — Webb-Pearman-Leuning (1980) for open-path CO₂/H₂O fluxes
- **Quality control** — inertial-subrange slope fitting, stationarity test (Foken & Wichura 1996), Foken 9-class quality flags, ITC tests, outlier detection
- **Physical constants** — curated constants, surface-type enumerations, and roughness / displacement height helpers
- **Flux pipeline** — end-to-end `CalcFlux` processor for IRGASON and KH-20 sensor suites with Polars/pandas compatibility
- **I/O** — fast Campbell Scientific TOA5 reader and multi-file compiler (Polars backend)
- **Plotting** — publication-quality Kaimal-style spectral and cospectral figures

## Installation

```bash
pip install TaylorSwift
```

For development:

```bash
git clone https://github.com/inkenbrandt/TaylorSwift
cd TaylorSwift
pip install -e ".[dev]"
```

## Quick start

```python
import TaylorSwift as tswift

# --- Configure the site ---
config = tswift.SiteConfig(
    z_measurement=3.0,    # measurement height [m]
    z_canopy=0.3,         # canopy height [m]
    sampling_freq=20.0,   # Hz
    averaging_period=30.0 # minutes
)

# --- Load a raw TOA5 file ---
df, meta = tswift.read_toa5("path/to/TOA5_mysite.dat")

# --- Process all 30-min intervals ---
results = tswift.process_file(df, config)

# --- Run quality control ---
from tswift.qc import run_qc
results = run_qc(results)

# --- Plot ---
from tswift.plotting import plot_cospectra
fig = plot_cospectra(results)
fig.savefig("cospectra.pdf")
```

## Processing pipeline

```
TOA5 files
    │
    ▼  tswift.read_toa5() / tswift.compile_toa5()
polars.DataFrame
    │
    ▼  tswift.process_file()
    │   ├─ double rotation (mean v = w = 0)
    │   ├─ linear detrend
    │   ├─ batched FFT (6 signals)
    │   ├─ logarithmic frequency binning
    │   └─ turbulence statistics (u*, L, z/L, H)
list[SpectralResult]
    │
    ├──▶  corrections.apply_spectral_corrections()  (optional)
    ├──▶  qc.run_qc()
    └──▶  plotting.plot_cospectra() / plot_spectra() / plot_ogive()
```

## Module overview

| Module | Contents |
|---|---|
| `core` | `SiteConfig`, `SpectralResult`, `process_interval`, `process_file`, `compute_cospectrum`, `rotate_wind`, `log_bin` |
| `io` | `read_toa5`, `compile_toa5`, `scan_toa5_directory` |
| `corrections` | `InstrumentConfig`, `ukde_despike`, `despike_dataframe`, transfer functions, `apply_spectral_corrections`, `wpl_correction` |
| `qc` | `fit_inertial_slope`, `stationarity_test`, `run_qc` |
| `plotting` | `plot_cospectra`, `plot_spectra`, `plot_ogive`, `plot_summary_timeseries` |
| `constants` | `SurfaceType`, `Hemisphere`, `QualityThreshold`, `ProcessingConfig`, `ErrorCode`, `get_displacement_height`, `get_roughness_length`, physical constants |
| `data_quality` | `QualityFlag`, `StabilityParameters`, `StationarityTest`, `DataQuality`, `OutlierDetection`, `quality_filter`, `rolling_sigma_filter` |
| `ec_polars` | `CalcFlux` — full flux pipeline for IRGASON and KH-20 with Polars/pandas compat |

## Running tests

```bash
pytest
# or with coverage:
pytest --cov=TaylorSwift
```

## References

- Kaimal, J.C. et al. (1972). Spectral characteristics of surface-layer turbulence. *Quart. J. Roy. Meteor. Soc.*, 98, 563–589.
- Massman, W.J. (2000). A simple method for estimating frequency response corrections for eddy covariance systems. *Agric. For. Meteorol.*, 104, 185–198.
- Webb, E.K., Pearman, G.I. & Leuning, R. (1980). Correction of flux measurements for density effects. *Quart. J. Roy. Meteor. Soc.*, 106, 85–100.
- Foken, T. & Wichura, B. (1996). Tools for quality assessment of surface-based flux measurements. *Agric. For. Meteorol.*, 78, 83–105.
- Foken, T. et al. (2004). Post-field data quality control. In *Handbook of Micrometeorology* (pp. 181–208). Springer.
- Metzger, S. et al. (2012). Eddy-covariance flux measurements with a weight-shift microlight aircraft. *Atmos. Meas. Tech.*, 5, 1699–1717.
- Oke, T.R. (1987). *Boundary Layer Climates* (2nd ed.). Routledge.
- Stull, R.B. (1988). *An Introduction to Boundary Layer Meteorology*. Springer.
