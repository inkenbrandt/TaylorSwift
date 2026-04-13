"""
TaylorSwift — Eddy Covariance Cospectral Analysis
==================================================

A Python package for computing and analyzing (co)spectra from high-frequency
eddy covariance time series, following standard micrometeorological approaches
(Kaimal et al. 1972; Moraes et al. 2008; Cheng et al. 2018).

Modules
-------
core        – FFT-based spectral / cospectral computation with detrending,
              coordinate rotation, and logarithmic frequency binning.
io          – Readers for Campbell Scientific TOA5 files and multi-file
              compilation for long time series.
corrections – Pre-processing despiking (UKDE, Metzger et al. 2012),
              spectral transfer functions (Massman 2000, Horst 1997),
              high/low frequency corrections, and WPL density correction.
plotting    – Publication-quality Kaimal-style spectral plots.
qc          – Quality-control helpers: inertial-subrange slope fitting,
              stationarity tests, and diagnostic flags.
constants   – Physical constants, surface-type enumerations, and default
              configuration for eddy covariance calculations.
data_quality – Foken et al. (2004) quality flags, stationarity and ITC
              tests, outlier detection, and rolling sigma filtering.
ec_polars   – High-frequency flux processing pipeline (CalcFlux) with
              Polars/pandas compatibility and multiple despiking methods.
"""

from .core import (
    compute_cospectrum as compute_cospectrum,
    compute_spectrum as compute_spectrum,
    rotate_wind as rotate_wind,
    process_interval as process_interval,
    process_file as process_file,
    SiteConfig as SiteConfig,
)
from .io import (
    read_toa5 as read_toa5,
    compile_toa5 as compile_toa5,
    scan_toa5_directory as scan_toa5_directory,
)
from .corrections import (
    InstrumentConfig as InstrumentConfig,
    ukde_despike as ukde_despike,
    polars_ukde_despike as polars_ukde_despike,
    despike_dataframe as despike_dataframe,
    apply_spectral_corrections as apply_spectral_corrections,
    compute_spectral_correction_factor as compute_spectral_correction_factor,
    wpl_correction as wpl_correction,
    enrich_results_with_means as enrich_results_with_means,
)
from .plotting import (
    plot_cospectra as plot_cospectra,
    plot_spectra as plot_spectra,
    plot_ogive as plot_ogive,
)
from .qc import (
    fit_inertial_slope as fit_inertial_slope,
    stationarity_test as stationarity_test,
)
from .constants import (
    SurfaceType as SurfaceType,
    Hemisphere as Hemisphere,
    QualityThreshold as QualityThreshold,
    ProcessingConfig as ProcessingConfig,
    get_displacement_height as get_displacement_height,
    get_roughness_length as get_roughness_length,
)
from .data_quality import (
    QualityFlag as QualityFlag,
    StabilityParameters as StabilityParameters,
    StationarityTest as StationarityTest,
    DataQuality as DataQuality,
    OutlierDetection as OutlierDetection,
    quality_filter as quality_filter,
    rolling_sigma_filter as rolling_sigma_filter,
)
from .ec_polars import CalcFlux as CalcFlux

__version__ = "0.2.0"
