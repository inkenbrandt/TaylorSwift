# API Reference

Everything in this reference is generated from the docstrings in
`src/TaylorSwift`.

## Import surface

`TaylorSwift` resolves its public names lazily, so `import TaylorSwift` is
cheap and the submodule is only imported when you touch the attribute:

```python
import TaylorSwift as tswift

tswift.SiteConfig        # -> TaylorSwift.config.SiteConfig
tswift.process_file      # -> TaylorSwift.core.process_file
tswift.__version__       # read from the installed distribution metadata
```

Every name below is available both at the top level and from its defining
module. The top-level spelling is the stable one; module paths may move.

!!! note "Distribution name vs. import name"
    The project installs as `taylorswift-spectra` but imports as
    `TaylorSwift`. See [Installation](../installation.md).

## Top-level exports by topic

### Configuration

| Name | Defined in |
| --- | --- |
| `SiteConfig` | [config](config.md) |
| `InstrumentConfig` (alias of `SiteConfig`) | [config](config.md) |
| `FluxConfig` | [config](config.md) |
| `ProcessingConfig` | [config](config.md) |
| `ScreeningConfig` | [screening](screening.md) |

### Spectral computation

| Name | Defined in |
| --- | --- |
| `process_interval` | [core](core.md) |
| `process_file` | [core](core.md) |
| `compute_cospectrum` | [cospectra](cospectra.md) |
| `compute_spectrum` | [cospectra](cospectra.md) |
| `rotate_wind` | [rotations](rotations.md) |

### Corrections

| Name | Defined in |
| --- | --- |
| `apply_spectral_corrections` | [corrections](corrections.md) |
| `compute_spectral_correction_factor` | [corrections](corrections.md) |
| `horst_analytical_correction` | [corrections](corrections.md) |
| `wpl_correction` | [corrections](corrections.md) |
| `enrich_results_with_means` | [corrections](corrections.md) |
| `combined_transfer_function` | [transfer_functions](transfer_functions.md) |
| `kaimal_cospec_model` | [transfer_functions](transfer_functions.md) |

### Quality control

| Name | Defined in |
| --- | --- |
| `vickers_mahrt_screen` | [screening](screening.md) |
| `run_qc` | [data_quality](data_quality.md) |
| `fit_inertial_slope` | [data_quality](data_quality.md) |
| `stationarity_test` | [data_quality](data_quality.md) |
| `quality_filter` | [data_quality](data_quality.md) |
| `QualityFlag` | [data_quality](data_quality.md) |
| `StabilityParameters` | [data_quality](data_quality.md) |
| `StationarityTest` | [data_quality](data_quality.md) |
| `DataQuality` | [data_quality](data_quality.md) |

### Despiking

| Name | Defined in |
| --- | --- |
| `ukde_despike` | [despike](despike.md) |
| `polars_ukde_despike` | [despike](despike.md) |
| `despike_dataframe` | [despike](despike.md) |

### Results and export

| Name | Defined in |
| --- | --- |
| `SpectralResult` | [results](results.md) |
| `FluxResult` | [results](results.md) |
| `results_to_dataframe` | [results](results.md) |
| `spectra_to_dataframe` | [results](results.md) |
| `results_to_csv` | [results](results.md) |
| `results_to_parquet` | [results](results.md) |

### I/O

| Name | Defined in |
| --- | --- |
| `read_toa5` | [io](io.md) |
| `compile_toa5` | [io](io.md) |
| `scan_toa5_directory` | [io](io.md) |

### Plotting

| Name | Defined in |
| --- | --- |
| `plot_cospectra` | [plotting](plotting.md) |
| `plot_spectra` | [plotting](plotting.md) |
| `plot_ogive` | [plotting](plotting.md) |

### Legacy flux pipelines

| Name | Defined in |
| --- | --- |
| `run_irga` | [pipelines](pipelines.md) |
| `run_kh20` | [pipelines](pipelines.md) |
| `CalcFlux` | [compat](compat.md) |

### Constants and site geometry

| Name | Defined in |
| --- | --- |
| `SurfaceType` | [constants](constants.md) |
| `Hemisphere` | [constants](constants.md) |
| `QualityThreshold` | [constants](constants.md) |
| `get_displacement_height` | [constants](constants.md) |
| `get_roughness_length` | [constants](constants.md) |

### Fitted spectral diagnostics

The [ec_spectral module](ec_spectral.md) provides the full diagnostic API.
Top-level exports include `ECSystem`, `fit_cospectrum`,
`correction_factor_integral`, `correction_factor_analytical`,
`correction_uncertainty`, `equivalent_time_constants`, and `correct_flux_table`.
