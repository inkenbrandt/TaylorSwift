"""
Data quality assessment for eddy covariance measurements.

This module implements data quality checks based on:
1. Steady state tests
2. Integral turbulence characteristics (ITC)
3. Wind direction relative to CSAT orientation
4. Statistical tests and outlier detection

References:
    Foken et al. (2004) Handbook of Micrometeorology
    Foken et al. (2012) Eddy Covariance: A Practical Guide

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

from typing import Tuple, Optional, Union, Dict
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
import polars as pl


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
        return "bad"

    if is_cospectrum:
        # Expected: -4/3 ≈ -1.33 (Kaimal) or -1.0 (Cheng stable)
        # Area-preserving form: slope of n*Co vs f
        if -1.8 <= slope <= -0.8:
            return "good"
        elif -2.2 <= slope <= -0.5:
            return "acceptable"
        elif -3.0 <= slope <= 0.0:
            return "suspect"
        else:
            return "bad"
    else:
        # Power spectrum: expected -2/3 ≈ -0.67
        if -1.1 <= slope <= -0.3:
            return "good"
        elif -1.5 <= slope <= 0.0:
            return "acceptable"
        elif -2.0 <= slope <= 0.5:
            return "suspect"
        else:
            return "bad"


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
        Sampling frequency (Hz) of the input time series.
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
    from .cospectra import tf_linear_detrend

    N = len(w)
    sub_len = N // n_subwindows
    sample_window = N * fs * 0.0166667  # 1 min in samples for detrending

    # Full-interval covariance
    w_det = tf_linear_detrend(w, sample_window)
    x_det = tf_linear_detrend(x, sample_window)
    cov_full = np.nanmean(w_det * x_det)

    if abs(cov_full) < 1e-12:
        return np.nan, 4

    # Sub-interval covariances
    sub_covs = []
    for i in range(n_subwindows):
        i0 = i * sub_len
        i1 = i0 + sub_len
        ws = tf_linear_detrend(w[i0:i1], sample_window)
        xs = tf_linear_detrend(x[i0:i1], sample_window)
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
        for name, attr in [
            ("wT", "ncosp_wT"),
            ("wu", "ncosp_wu"),
            ("wCO2", "ncosp_wCO2"),
            ("wH2O", "ncosp_wH2O"),
        ]:
            data = getattr(res, attr)
            if len(data) > 0:
                slope, r2, _ = fit_inertial_slope(f, data, f_range_cospec)
                res.qc_flags[f"slope_{name}"] = slope
                res.qc_flags[f"slope_r2_{name}"] = r2
                res.qc_flags[f"slope_class_{name}"] = classify_slope(slope, True)

        # Power spectral slopes
        for name, attr in [
            ("u", "spec_u"),
            ("v", "spec_v"),
            ("w", "spec_w"),
            ("T", "spec_T"),
        ]:
            data = getattr(res, attr)
            if len(data) > 0:
                slope, r2, _ = fit_inertial_slope(f, data, f_range_spec)
                res.qc_flags[f"slope_{name}"] = slope
                res.qc_flags[f"slope_r2_{name}"] = r2
                res.qc_flags[f"slope_class_{name}"] = classify_slope(slope, False)

        # u* filter
        res.qc_flags["ustar_filter"] = res.ustar < 0.1

    return results


class QualityFlag(IntEnum):
    """
    Eddy-covariance **data-quality classification** after Foken *et al.* (2004).

    The nine-class scheme summarises stationarity tests, spectral corrections,
    and footprint considerations commonly applied in post-processing:

    ==========  ================================================================
    Member      Interpretation
    ----------  ---------------------------------------------------------------
    CLASS_1     Highest quality – meets all similarity assumptions
    CLASS_2     Good quality – minor violations (≤ 30 % correction)
    CLASS_3     Moderate quality – usable with caution
    CLASS_4     Low quality – usable only under specific conditions
    CLASS_5     Poor quality – storage-term evaluation required
    CLASS_6     Poor quality – additional flux corrections required
    CLASS_7     Poor quality – retain only for empirical relationships
    CLASS_8     Poor quality – discard for most research purposes
    CLASS_9     Very poor quality – always discard
    ==========  ================================================================

    References
    ----------
    Foken, T., Göckede, M., Mauder, M., Mahrt, L., Amiro, B., & Munger, J. W.
    (2004). *Post-field data quality control.* In X. Lee, W. Massman & B.
    Law (Eds.), **Handbook of Micrometeorology** (pp. 181–208). Springer.
    """

    CLASS_1 = 1  # Highest quality
    CLASS_2 = 2  # Good quality
    CLASS_3 = 3  # Moderate quality, usable
    CLASS_4 = 4  # Low quality, conditionally usable
    CLASS_5 = 5  # Poor quality, storage terms needed
    CLASS_6 = 6  # Poor quality, flux correction needed
    CLASS_7 = 7  # Poor quality, used for empirical relationships
    CLASS_8 = 8  # Poor quality, discarded in basic research
    CLASS_9 = 9  # Very poor quality, discarded


@dataclass
class StabilityParameters:
    """
    Container for key **surface-layer stability variables** used in
    Monin–Obukhov similarity analysis.

    Parameters
    ----------
    z : float
        Measurement height above the displacement plane, **metres**.
    L : float
        Monin–Obukhov length *L* (m); sign denotes stability
        ( + stable, – unstable, |L| → ∞ neutral ).
    u_star : float
        Friction velocity *u★* (m s⁻¹).
    sigma_w : float
        Standard deviation of the vertical wind component σ_w (m s⁻¹).
    sigma_T : float
        Standard deviation of air (or virtual) temperature σ_T (K).
    T_star : float
        Temperature scale *T★* = − w′T′/u★ (K).
    latitude : float
        Site latitude (decimal degrees, positive northward).

    Notes
    -----
    * Values typically represent **30-min averaging periods** but can be
      adapted to any timescale provided the underlying turbulence is
      stationary.
    * Additional metrics such as *β* = σ_w ⁄ u★ or the stability parameter
      ζ = z ⁄ L are easily derived from these fields.

    Examples
    --------
    >>> sp = StabilityParameters(
    ...     z=3.5, L=-50.0, u_star=0.45,
    ...     sigma_w=0.26, sigma_T=0.18,
    ...     T_star=-0.40, latitude=40.77
    ... )
    >>> sp.z, sp.u_star
    (3.5, 0.45)
    """

    z: float  # Measurement height (m)
    L: float  # Obukhov length (m)
    u_star: float  # Friction velocity (m/s)
    sigma_w: float  # Std. dev. vertical wind (m/s)
    sigma_T: float  # Std. dev. temperature (K)
    T_star: float  # Temperature scale (K)
    latitude: float  # Site latitude (degrees)


@dataclass
class StationarityTest:
    """
    Summary of **relative non-stationarity indices** derived from the
    Foken & Wichura (1996) *stationarity test*.

    Each index compares the mean covariance over the full averaging period
    (typically 30 min) with the average of covariances computed in a series of
    shorter sub-intervals (e.g. 5 min blocks).  Values ≲ 0.3 indicate
    satisfactory stationarity; values ≳ 0.6 suggest the flux is unreliable.

    Parameters
    ----------
    RN_uw : float
        Relative non-stationarity of the **momentum flux**
        :math:`u'w'` (dimensionless).
    RN_wT : float
        Relative non-stationarity of the **sensible-heat flux**
        :math:`w'T'` (dimensionless).
    RN_wq : float
        Relative non-stationarity of the **latent-heat flux**
        :math:`w'q'` (dimensionless).
    RN_wc : float
        Relative non-stationarity of the **CO₂ flux**
        :math:`w'c'` (dimensionless).

    Notes
    -----
    * A common quality-control criterion assigns **Class 1–3** to fluxes with
      all indices < 0.3, **Class 4–6** to 0.3 ≤ RN < 0.6, and **Class 7–9**
      when any RN ≥ 0.6 (Foken et al., 2004).
    * Sub-interval length and the definition of “relative” (absolute vs.
      normalised difference) must match the method used in the flux-processing
      software.

    Examples
    --------
    >>> st = StationarityTest(RN_uw=0.12, RN_wT=0.18,
    ...                       RN_wq=0.25, RN_wc=0.31)
    >>> st.RN_wT
    0.18
    """

    RN_uw: float  # Relative non-stationarity for momentum flux
    RN_wT: float  # Relative non-stationarity for sensible-heat flux
    RN_wq: float  # Relative non-stationarity for latent-heat flux
    RN_wc: float  # Relative non-stationarity for CO₂ flux


class DataQuality:
    """
    Data quality assessment following Foken et al. (2004, 2012).

    Implements comprehensive quality control including:
    - Stationarity tests
    - Integral turbulence characteristics
    - Wind direction checks
    - Overall quality flags
    """

    def __init__(self, use_wind_direction: bool = True):
        """
        Initialize data quality assessment.

        Args:
            use_wind_direction: Whether to include wind direction in quality assessment
        """
        self.use_wind_direction = use_wind_direction

    def _calculate_integral_turbulence(
        self, stability: StabilityParameters
    ) -> Tuple[float, float]:
        """
        Calculate integral turbulence characteristics.

        Args:
            stability: StabilityParameters object

        Returns:
            Tuple containing:
            - ITC for momentum flux
            - ITC for scalar flux
        """
        z_L = stability.z / stability.L

        # Parameters depending on stability following Foken et al. (2004)
        if z_L <= -0.032:
            # Unstable conditions
            itc_w = 2.00 * abs(z_L) ** 0.125  # For vertical velocity
            itc_T = abs(z_L) ** (-1 / 3)  # For temperature

        elif z_L <= 0.0:
            # Near-neutral unstable
            itc_w = 1.3
            itc_T = 0.5 * abs(z_L) ** (-0.5)

        elif z_L < 0.4:
            # Near-neutral stable
            # Calculate Coriolis parameter
            f = 2 * 7.2921e-5 * np.sin(np.radians(stability.latitude))
            itc_w = 0.21 * np.log(abs(f) / stability.u_star) + 3.1
            itc_T = 1.4 * z_L ** (-0.25)

        else:
            # Stable conditions
            itc_w = -(stability.sigma_w / stability.u_star) / 9.1
            itc_T = -(stability.sigma_T / abs(stability.T_star)) / 9.1

        return itc_w, itc_T

    def _check_wind_direction(self, wind_direction: float) -> int:
        """
        Check wind direction relative to CSAT orientation.

        Args:
            wind_direction: Wind direction in degrees

        Returns:
            Quality class (1-9) based on wind direction
        """
        if not self.use_wind_direction:
            return QualityFlag.CLASS_1

        if (wind_direction < 151.0) or (wind_direction > 209.0):
            return QualityFlag.CLASS_1
        elif (151.0 <= wind_direction < 171.0) or (189.0 <= wind_direction <= 209.0):
            return QualityFlag.CLASS_7
        else:  # 171.0 <= wind_direction <= 189.0
            return QualityFlag.CLASS_9

    def _evaluate_stationarity(
        self, stationarity: StationarityTest, flux_type: str
    ) -> int:
        """
        Evaluate stationarity test results.

        Args:
            stationarity: StationarityTest object
            flux_type: Type of flux ('momentum', 'heat', 'moisture', 'co2')

        Returns:
            Quality class (1-9) based on stationarity
        """
        # Get relevant RN value
        if flux_type == "momentum":
            rn = stationarity.RN_uw
        elif flux_type == "heat":
            rn = stationarity.RN_wT
        elif flux_type == "moisture":
            rn = stationarity.RN_wq
        else:  # CO2
            rn = stationarity.RN_wc

        # Classify based on relative non-stationarity
        if rn < 0.16:
            return QualityFlag.CLASS_1
        elif rn < 0.31:
            return QualityFlag.CLASS_2
        elif rn < 0.76:
            return QualityFlag.CLASS_3
        elif rn < 1.01:
            return QualityFlag.CLASS_4
        elif rn < 2.51:
            return QualityFlag.CLASS_5
        elif rn < 10.0:
            return QualityFlag.CLASS_6
        else:
            return QualityFlag.CLASS_9

    def _evaluate_itc(self, measured: float, modeled: float) -> int:
        """
        Evaluate integral turbulence characteristic test.

        Args:
            measured: Measured ITC
            modeled: Modeled ITC

        Returns:
            Quality class (1-9) based on ITC comparison
        """
        # Calculate relative difference
        itc_diff = abs((measured - modeled) / modeled)

        # Classify based on difference
        if itc_diff < 0.31:
            return QualityFlag.CLASS_1
        elif itc_diff < 0.76:
            return QualityFlag.CLASS_2
        elif itc_diff < 1.01:
            return QualityFlag.CLASS_3
        elif itc_diff < 2.51:
            return QualityFlag.CLASS_4
        elif itc_diff < 10.0:
            return QualityFlag.CLASS_5
        else:
            return QualityFlag.CLASS_9

    def assess_data_quality(
        self,
        stability: StabilityParameters,
        stationarity: StationarityTest,
        wind_direction: Optional[float] = None,
        flux_type: str = "momentum",
    ) -> Dict[str, Union[int, float]]:
        """
        Perform comprehensive data quality assessment.

        Args:
            stability: StabilityParameters object
            stationarity: StationarityTest object
            wind_direction: Wind direction in degrees (optional)
            flux_type: Type of flux to assess ('momentum', 'heat', 'moisture', 'co2')

        Returns:
            Dictionary containing:
            - overall_flag: Final quality classification
            - stationarity_flag: Quality based on stationarity
            - itc_flag: Quality based on ITC
            - wind_dir_flag: Quality based on wind direction
            - itc_measured: Measured ITC value
            - itc_modeled: Modeled ITC value
        """
        # Calculate ITC
        itc_w, itc_T = self._calculate_integral_turbulence(stability)

        # Get measured ITC
        measured_itc = stability.sigma_w / stability.u_star
        if flux_type == "momentum":
            modeled_itc = itc_w
        else:
            modeled_itc = itc_T

        # Evaluate individual tests
        station_flag = self._evaluate_stationarity(stationarity, flux_type)
        itc_flag = self._evaluate_itc(measured_itc, modeled_itc)
        wind_flag = (
            self._check_wind_direction(wind_direction)
            if wind_direction is not None
            else QualityFlag.CLASS_1
        )

        # Overall quality is worst of individual flags
        overall_flag = max(station_flag, itc_flag, wind_flag)

        return {
            "overall_flag": overall_flag,
            "stationarity_flag": station_flag,
            "itc_flag": itc_flag,
            "wind_dir_flag": wind_flag,
            "itc_measured": measured_itc,
            "itc_modeled": modeled_itc,
        }

    def get_quality_label(self, flag: int) -> str:
        """Get descriptive label for quality flag."""
        labels = {
            1: "Highest quality",
            2: "Good quality",
            3: "Moderate quality",
            4: "Low quality",
            5: "Poor quality (storage)",
            6: "Poor quality (flux correction)",
            7: "Poor quality (empirical only)",
            8: "Poor quality (discard research)",
            9: "Very poor quality (discard)",
        }
        return labels.get(flag, "Unknown")


def quality_filter(
    data: np.ndarray,
    quality_flags: np.ndarray,
    min_quality: int = 3,
) -> np.ndarray:
    """
    Mask *data* values that do not meet a minimum quality criterion.

    Each sample in *data* has an associated integer quality class in
    ``quality_flags``—​smaller numbers indicate higher quality.  Samples
    whose quality class exceeds ``min_quality`` are considered unreliable
    and are replaced with ``numpy.nan`` (the function returns a
    **floating-point** copy of the input so that ``NaN`` assignment is
    possible).

    Parameters
    ----------
    data : ndarray
        One-dimensional or multi-dimensional numeric array containing the
        measurements to be filtered.
    quality_flags : ndarray
        Integer array of the same shape as ``data`` that encodes the
        quality class for every sample.  A common convention is
        ``0 = best``, higher integers = lower quality.
    min_quality : int, default ``3``
        Maximum acceptable quality class (inclusive).  Any element with
        ``quality_flags > min_quality`` is treated as invalid.

    Returns
    -------
    filtered : ndarray
        A **float** array with the same shape as ``data``.  Elements that
        fail the quality test are set to ``numpy.nan``; all other samples
        retain their original value.

    Raises
    ------
    ValueError
        If ``data`` and ``quality_flags`` have incompatible shapes.
    TypeError
        If ``quality_flags`` cannot be safely cast to an integer dtype.

    Notes
    -----
    * The output dtype is promoted to floating point (``numpy.float64``)
      if the input is integral, because NaNs are not representable in
      integer arrays.
    * For multi-dimensional inputs the comparison is applied element-wise
      with no aggregation across axes.

    Examples
    --------
    >>> import numpy as np
    >>> vals = np.array([1.2, 3.4, 5.6, 7.8])
    >>> qf   = np.array([0,   2,   4,   5])   # 0 = best, 5 = worst
    >>> quality_filter(vals, qf, min_quality=3)
    array([1.2, 3.4,  nan,  nan])
    """

    filtered = np.array(data, dtype=np.float64)
    filtered[quality_flags > min_quality] = np.nan
    return filtered
