"""
qc.py — Quality-control diagnostics for eddy-covariance cospectra.

Provides tools to:
  1. Fit the inertial-subrange slope and compare to theoretical values
     (-7/3 ≈ -2.33 for cospectra, or -2 per Cheng et al. 2018 for stable
     conditions; -5/3 ≈ -1.67 for power spectra).
  2. Stationarity test (Foken & Wichura 1996) — compare 5-min sub-interval
     covariances against the 30-min covariance.
  3. Integral turbulence characteristics test.
  4. Flux footprint quality flag based on wind speed / u* threshold.

References
----------
Foken, T. & Wichura, B. (1996). Tools for quality assessment of
    surface-based flux measurements. Agric. For. Meteorol., 78, 83–105.
Cheng, Y. et al. (2018). On the Power-law Scaling of Turbulence Cospectra
    Part 1: Stably Stratified ABL. Boundary-Layer Meteorol.
Vickers, D. & Mahrt, L. (1997). Quality control analysis of flux data.
    J. Atmos. Ocean. Technol., 14, 512–526.
"""

import numpy as np
from .core import SpectralResult


# ---------------------------------------------------------------------------
# Inertial-subrange slope fitting
# ---------------------------------------------------------------------------
def fit_inertial_slope(
    freq_nd: np.ndarray,
    ncosp: np.ndarray,
    f_range: tuple = (1.0, 8.0),
):
    """
    Fit the power-law slope in the inertial subrange.

    In the area-preserving representation n·Co(n) vs f, the inertial
    subrange slope is:
      - −4/3 ≈ −1.33  for cospectra (classical -7/3 in Co(n) vs n)
      - −2/3 ≈ −0.67  for power spectra (classical -5/3 in S(n) vs n)
    Cheng et al. (2018) argue for -1 (i.e., -2 in Co vs n) under stable
    stratification.

    Parameters
    ----------
    freq_nd : np.ndarray
        Dimensionless frequency f = nz/U.
    ncosp : np.ndarray
        n·Co(n) or |n·Co(n)| — absolute values recommended.
    f_range : tuple
        (f_min, f_max) range over which to fit the slope.

    Returns
    -------
    slope : float
        Fitted slope in log-log space.
    r_squared : float
        Coefficient of determination of the fit.
    intercept : float
        Intercept of the log-log fit.
    """
    mask = (freq_nd >= f_range[0]) & (freq_nd <= f_range[1])
    x = np.log10(freq_nd[mask])
    y_raw = ncosp[mask]

    # Use absolute value (cospectra can be negative)
    y_raw = np.abs(y_raw)
    pos = y_raw > 0
    if pos.sum() < 3:
        return np.nan, np.nan, np.nan

    x = x[pos]
    y = np.log10(y_raw[pos])

    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs

    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return slope, r_squared, intercept


def classify_slope(slope: float, is_cospectrum: bool = True) -> str:
    """
    Classify the fitted inertial-subrange slope as a QC category.

    Returns one of: 'good', 'acceptable', 'suspect', 'bad'.
    """
    if np.isnan(slope):
        return 'bad'

    if is_cospectrum:
        # Expected: -4/3 ≈ -1.33 (Kaimal) or -1.0 (Cheng stable)
        # Area-preserving form: slope of n*Co vs f
        if -1.8 <= slope <= -0.8:
            return 'good'
        elif -2.2 <= slope <= -0.5:
            return 'acceptable'
        elif -3.0 <= slope <= 0.0:
            return 'suspect'
        else:
            return 'bad'
    else:
        # Power spectrum: expected -2/3 ≈ -0.67
        if -1.1 <= slope <= -0.3:
            return 'good'
        elif -1.5 <= slope <= 0.0:
            return 'acceptable'
        elif -2.0 <= slope <= 0.5:
            return 'suspect'
        else:
            return 'bad'


# ---------------------------------------------------------------------------
# Stationarity test (Foken & Wichura 1996)
# ---------------------------------------------------------------------------
def stationarity_test(
    w: np.ndarray,
    x: np.ndarray,
    fs: float,
    n_subwindows: int = 6,
):
    """
    Stationarity test following Foken & Wichura (1996).

    Compares the mean of sub-interval covariances to the full-interval
    covariance. A relative difference < 30% is considered stationary
    (quality class 1–2); > 30% is non-stationary.

    Parameters
    ----------
    w : np.ndarray
        Vertical wind component (detrended).
    x : np.ndarray
        Scalar or horizontal wind component (detrended).
    fs : float
        Sampling frequency.
    n_subwindows : int
        Number of equal sub-intervals (default 6 → 5-min windows for 30-min).

    Returns
    -------
    relative_diff : float
        |1 - mean(sub_cov) / full_cov| as a fraction.
    quality_class : int
        1 = good (<15%), 2 = acceptable (15–30%), 3 = suspect (30–50%),
        4 = bad (>50%).
    """
    from .core import _detrend_linear

    N = len(w)
    sub_len = N // n_subwindows

    # Full-interval covariance
    w_det = _detrend_linear(w)
    x_det = _detrend_linear(x)
    cov_full = np.nanmean(w_det * x_det)

    if abs(cov_full) < 1e-12:
        return np.nan, 4

    # Sub-interval covariances
    sub_covs = []
    for i in range(n_subwindows):
        i0 = i * sub_len
        i1 = i0 + sub_len
        ws = _detrend_linear(w[i0:i1])
        xs = _detrend_linear(x[i0:i1])
        sub_covs.append(np.nanmean(ws * xs))

    cov_sub_mean = np.mean(sub_covs)
    rel_diff = abs(1.0 - cov_sub_mean / cov_full)

    if rel_diff < 0.15:
        qc = 1
    elif rel_diff < 0.30:
        qc = 2
    elif rel_diff < 0.50:
        qc = 3
    else:
        qc = 4

    return rel_diff, qc


# ---------------------------------------------------------------------------
# Batch QC for a list of SpectralResults
# ---------------------------------------------------------------------------
def run_qc(results, f_range_cospec=(1.0, 8.0), f_range_spec=(1.0, 8.0)):
    """
    Run quality-control diagnostics on a list of SpectralResult objects.

    Adds entries to each result's qc_flags dict:
      - slope_wT, slope_wu, slope_wCO2, slope_wH2O
      - slope_class_wT, slope_class_wu, ...
      - ustar_filter (True if u* < 0.1 m/s)

    Parameters
    ----------
    results : list[SpectralResult]
    f_range_cospec : tuple
        Dimensionless frequency range for cospectral slope fitting.
    f_range_spec : tuple
        Range for spectral slope fitting.
    """
    for res in results:
        f = res.freq_nd
        if len(f) == 0:
            continue

        # Cospectral slopes
        for name, attr in [('wT', 'ncosp_wT'), ('wu', 'ncosp_wu'),
                           ('wCO2', 'ncosp_wCO2'), ('wH2O', 'ncosp_wH2O')]:
            data = getattr(res, attr)
            if len(data) > 0:
                slope, r2, _ = fit_inertial_slope(f, data, f_range_cospec)
                res.qc_flags[f'slope_{name}'] = slope
                res.qc_flags[f'slope_r2_{name}'] = r2
                res.qc_flags[f'slope_class_{name}'] = classify_slope(slope, True)

        # Power spectral slopes
        for name, attr in [('u', 'spec_u'), ('v', 'spec_v'),
                           ('w', 'spec_w'), ('T', 'spec_T')]:
            data = getattr(res, attr)
            if len(data) > 0:
                slope, r2, _ = fit_inertial_slope(f, data, f_range_spec)
                res.qc_flags[f'slope_{name}'] = slope
                res.qc_flags[f'slope_r2_{name}'] = r2
                res.qc_flags[f'slope_class_{name}'] = classify_slope(slope, False)

        # u* filter
        res.qc_flags['ustar_filter'] = res.ustar < 0.1

    return results
