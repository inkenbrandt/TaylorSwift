"""
eccospectra — Eddy Covariance Cospectral Analysis
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
"""

from .core import (
    compute_cospectrum,
    compute_spectrum,
    rotate_wind,
    process_interval,
    process_file,
    SiteConfig,
)
from .io import read_toa5, compile_toa5, scan_toa5_directory
from .corrections import (
    InstrumentConfig,
    ukde_despike,
    polars_ukde_despike,
    despike_dataframe,
    apply_spectral_corrections,
    compute_spectral_correction_factor,
    wpl_correction,
    enrich_results_with_means,
)
from .plotting import plot_cospectra, plot_spectra, plot_ogive
from .qc import fit_inertial_slope, stationarity_test

__version__ = "0.2.0"
