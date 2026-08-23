# TaylorSwift

[![PyPI](https://img.shields.io/pypi/v/taylorswift-spectra.svg)](https://pypi.org/project/taylorswift-spectra/)
[![Python](https://img.shields.io/pypi/pyversions/taylorswift-spectra.svg)](https://pypi.org/project/taylorswift-spectra/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/inkenbrandt/TaylorSwift/blob/main/LICENSE)

FFT-based (co)spectral analysis for eddy covariance time series. The name
honours the physicist Sir Geoffrey Ingram Taylor, not the musician.

`TaylorSwift` implements the standard micrometeorological workflow for
computing power spectra and cospectra from high-frequency sonic anemometer and
open-path gas analyser data, following Kaimal et al. (1972) conventions.

<div class="grid cards" markdown>

-   :material-download: **[Installation](installation.md)**

    Install from PyPI, or set up a development checkout.

-   :material-rocket-launch: **[Quickstart](quickstart.md)**

    From a TOA5 file to corrected, quality-flagged cospectra in eight steps.

-   :material-book-open-variant: **[User Guide](guide/configuration.md)**

    One page per stage of the pipeline, with the physics behind each.

-   :material-api: **[API Reference](api/index.md)**

    Every public function and dataclass, generated from the source.

</div>

## Features

- **Spectral computation** — double-rotation, linear detrending,
  Hamming-windowed FFT, logarithmic frequency binning, area-preserving
  normalization
- **Spectral corrections** — block-average, linear-detrend, first-order sensor
  response, sonic path averaging, sensor separation (Massman 2000; Horst 1997)
- **Raw-data screening** — Vickers & Mahrt (1997) spike, amplitude-resolution,
  dropout, absolute-limit and higher-moment tests
- **Despiking** — iterative UKDE despiking for raw time series
  (Metzger et al. 2012); rolling IQR, median-RLM, and EWMA methods via
  `CalcFlux`
- **WPL density correction** — Webb-Pearman-Leuning (1980) for open-path
  CO₂/H₂O fluxes
- **Quality control** — inertial-subrange slope fitting, stationarity test
  (Foken & Wichura 1996), Foken 9-class quality flags, ITC tests, outlier
  detection
- **Physical constants** — curated constants, surface-type enumerations, and
  roughness / displacement height helpers
- **Flux pipeline** — end-to-end `CalcFlux` processor for IRGASON and KH-20
  sensor suites with Polars/pandas compatibility
- **I/O** — fast Campbell Scientific TOA5 reader and multi-file compiler
  (Polars backend)
- **Plotting** — publication-quality Kaimal-style spectral and cospectral
  figures

## Processing pipeline

```mermaid
graph TD
    A["TOA5 files"] -->|read_toa5 / compile_toa5| B["polars.DataFrame"]
    B -->|despike_dataframe| B2["despiked frame"]
    B2 -->|process_file| C["list[SpectralResult]"]
    C -->|apply_spectral_corrections| D["corrected fluxes"]
    D -->|run_qc| E["quality-flagged results"]
    E -->|results_to_dataframe| F["tidy table / CSV / Parquet"]
    E -->|plot_cospectra| G["figures"]
```

Inside `process_file`, each averaging interval passes through:

```mermaid
graph LR
    S["raw interval"] --> S1["Vickers &amp; Mahrt screening"]
    S1 --> S2["NaN screen &amp; gap fill"]
    S2 --> S3["double rotation"]
    S3 --> S4["linear detrend"]
    S4 --> S5["Hamming window + FFT"]
    S5 --> S6["log-frequency binning"]
    S6 --> S7["normalise + turbulence stats"]
```

## A 20-line example

```python
import numpy as np
import TaylorSwift as tswift

config = tswift.SiteConfig(
    z_measurement=3.0,
    z_canopy=0.3,
    sampling_freq=20.0,
    averaging_period=30.0,
)

rng = np.random.default_rng(0)
n = int(30 * 60 * 20)  # 30 minutes at 20 Hz

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

## Citation

If you use `TaylorSwift` in published work, please cite the underlying methods
(see [References](references.md)) alongside the software.

## References

The full, per-method bibliography lives on the
[References](references.md) page.
