"""
TaylorSwift — Eddy Covariance Cospectral Analysis.
"""

from importlib import import_module

__version__ = "0.2.0"

_EXPORTS = {
    "compute_cospectrum": (".core", "compute_cospectrum"),
    "compute_spectrum": (".core", "compute_spectrum"),
    "rotate_wind": (".core", "rotate_wind"),
    "process_interval": (".core", "process_interval"),
    "process_file": (".core", "process_file"),
    "SiteConfig": (".core", "SiteConfig"),
    "read_toa5": (".io", "read_toa5"),
    "compile_toa5": (".io", "compile_toa5"),
    "scan_toa5_directory": (".io", "scan_toa5_directory"),
    "InstrumentConfig": (".corrections", "InstrumentConfig"),
    "wpl_correction": (".corrections", "wpl_correction"),
    "apply_spectral_corrections": (".corrections", "apply_spectral_corrections"),
    "compute_spectral_correction_factor": (
        ".corrections",
        "compute_spectral_correction_factor",
    ),
    "enrich_results_with_means": (".corrections", "enrich_results_with_means"),
    "ukde_despike": (".cleaning", "ukde_despike"),
    "polars_ukde_despike": (".cleaning", "polars_ukde_despike"),
    "despike_dataframe": (".cleaning", "despike_dataframe"),
    "plot_cospectra": (".plotting", "plot_cospectra"),
    "plot_spectra": (".plotting", "plot_spectra"),
    "plot_ogive": (".plotting", "plot_ogive"),
    "fit_inertial_slope": (".data_quality", "fit_inertial_slope"),
    "stationarity_test": (".data_quality", "stationarity_test"),
    "SurfaceType": (".constants", "SurfaceType"),
    "Hemisphere": (".constants", "Hemisphere"),
    "QualityThreshold": (".constants", "QualityThreshold"),
    "ProcessingConfig": (".constants", "ProcessingConfig"),
    "get_displacement_height": (".constants", "get_displacement_height"),
    "get_roughness_length": (".constants", "get_roughness_length"),
    "QualityFlag": (".data_quality", "QualityFlag"),
    "StabilityParameters": (".data_quality", "StabilityParameters"),
    "StationarityTest": (".data_quality", "StationarityTest"),
    "DataQuality": (".data_quality", "DataQuality"),
    "OutlierDetection": (".data_quality", "OutlierDetection"),
    "quality_filter": (".data_quality", "quality_filter"),
    "rolling_sigma_filter": (".data_quality", "rolling_sigma_filter"),
    "CalcFlux": (".compat", "CalcFlux"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(__all__))
