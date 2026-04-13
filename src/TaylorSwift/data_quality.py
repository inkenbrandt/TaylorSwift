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
"""

from __future__ import annotations

from typing import Tuple, Optional, Union, Dict
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
import polars as pl



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


class OutlierDetection:
    """Statistical methods for detecting outliers in flux data."""

    @staticmethod
    def mad_outliers(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
        """
        Identify **outliers** in a 1-D array using the *Median Absolute
        Deviation* (MAD) criterion.

        The modified Z-score is computed as

        .. math::

            z_i = 0.6745\\,\\frac{\\,x_i - \\tilde{x}\\,}{\\operatorname{MAD}},

        where :math:`\\tilde{x}` is the sample median and

        .. math::

            \\operatorname{MAD} = \\operatorname{median}(|x_i - \\tilde{x}|).

        An element is flagged as an outlier if ``|z_i| > threshold``.

        Parameters
        ----------
        data : ndarray
            One-dimensional numeric array to test.
        threshold : float, default ``3.5``
            Cut-off value for the modified Z-score.  A commonly used range is
            3.0 – 3.5; lowering the threshold flags more points as outliers.

        Returns
        -------
        ndarray of bool
            Boolean mask **M** with ``M[i] = True`` where *data[i]* is
            classified as an outlier and ``False`` elsewhere.

        Notes
        -----
        * The factor **0.6745** scales the MAD to be consistent with the
          standard deviation for a normal distribution.
        * If *MAD* is zero (all values identical), the function returns
          ``False`` everywhere (no outliers).
        * The method is robust up to ≈ 50 % contamination and is preferable to
          mean ± k·σ when the data distribution is heavy-tailed.

        Examples
        --------
        >>> import numpy as np
        >>> from ec import CalcFlux
        >>> x = np.array([1, 1, 1, 1, 10])  # 10 is an outlier
        >>> CalcFlux.mad_outliers(x)
        array([False, False, False, False,  True])
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return np.zeros_like(data, dtype=bool)
        modified_zscore = 0.6745 * (data - median) / mad
        return np.abs(modified_zscore) > threshold

    @staticmethod
    def spike_detection(
        data: np.ndarray,
        window_size: int = 100,
        z_threshold: float = 4.0,
    ) -> np.ndarray:
        """
        Identify isolated spikes in a univariate time-series or signal by
        comparing each point’s *z*-score to its local neighbourhood.

        A sliding window of length ``window_size`` is centred on each sample
        (truncated at the series boundaries).
        Within that window the mean (``μ``) and standard deviation (``σ``) are
        computed.
        A point is flagged as a spike when

        ``|xᵢ − μ| / σ  >  z_threshold``

        Parameters
        ----------
        data : ndarray
            One-dimensional array of numeric values.  The function assumes
            finite, non-NaN entries; pre-filter or impute missing values
            beforehand.
        window_size : int, default ``100``
            Length of the moving window (in samples).
            Must be a positive integer.  When the window extends beyond the
            series boundaries it is clipped, so edge points are compared to
            a smaller neighbourhood.
        z_threshold : float, default ``4.0``
            *z*-score above which a sample is considered an outlier.
            Typical values range from 3 to 6 depending on the desired
            sensitivity.

        Returns
        -------
        spikes : ndarray of bool
            Boolean mask of the same shape as ``data`` where ``True`` marks
            samples classified as spikes.

        Raises
        ------
        ValueError
            If ``window_size`` is not a positive integer.
        TypeError
            If ``data`` is not array-like or cannot be converted to
            ``numpy.ndarray``.

        Notes
        -----
        * A *spike* is defined relative to local variability; slowly varying
        drifts are **not** flagged.
        * The method is insensitive to window length provided the window
        spans at least ~20 points and covers the dominant noise structure.
        * For multivariate spike detection consider median absolute
        deviation (MAD) or robust Mahalanobis distance.

        Examples
        --------
        >>> import numpy as np
        >>> from mymodule import SignalTools   # doctest: +SKIP
        ...
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(0, 1, 1000)
        >>> x[[200, 600]] += 10        # inject two spikes
        >>> mask = SignalTools.spike_detection(x, window_size=51, z_threshold=4)
        >>> np.where(mask)[0]
        array([200, 600])
        >>> x_clean = np.where(mask, np.nan, x)   # simple removal
        """
        spikes = np.zeros_like(data, dtype=bool)

        for i in range(len(data)):
            # Get window indices
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2)

            # Calculate statistics for window
            window = data[start:end]
            mean = np.mean(window)
            std = np.std(window)

            # Check if point is spike
            if std > 0:  # Avoid division by zero
                z_score = abs(data[i] - mean) / std
                spikes[i] = z_score > z_threshold

        return spikes


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


def rolling_sigma_filter(
    df: pl.DataFrame,
    value_col: str = "Uz",
    time_col: str = "TIMESTAMP",
    period: str = "5s",
    sigma: float = 3.0,
    closed: str = "both",           # "both" | "left" | "right" | "none"
    output_col: str | None = None,  # default: f"{value_col}_filtered"
    keep_stats: bool = True,        # keep or drop the roll mean/std columns
    ensure_datetime: bool = True,   # cast time_col to pl.Datetime
) -> pl.DataFrame:
    """
    Apply a rolling ±sigma*std spike filter to `value_col` over a time-indexed window.

    Steps (mirrors your snippet):
      1) Sort by TIMESTAMP and (optionally) cast to pl.Datetime
      2) Compute rolling mean & std over a time window (e.g., '5s')
      3) Join stats back to original rows
      4) Null-out values outside mean ± sigma*std → write to `output_col`

    Parameters
    ----------
    df : pl.DataFrame
        Input data with at least [time_col, value_col].
    value_col : str
        Column to filter (e.g., "Uz").
    time_col : str
        Datetime-like column for rolling index (e.g., "TIMESTAMP").
    period : str
        Time window (e.g., '5s', '1m', '30m').
    sigma : float
        Threshold in standard deviations (e.g., 3.0).
    closed : str
        Window inclusion: 'both', 'left', 'right', or 'none'.
    output_col : str | None
        Name for filtered output column. Defaults to f"{value_col}_filtered".
    keep_stats : bool
        If False, drops the intermediate mean/std columns.
    ensure_datetime : bool
        If True, casts `time_col` to pl.Datetime.

    Returns
    -------
    pl.DataFrame
        Original df with:
          - {value_col}_roll_mean
          - {value_col}_roll_std
          - {output_col}  (filtered)
    """
    if output_col is None:
        output_col = f"{value_col}_filtered"

    # Step 1: ensure sort & datetime type
    out = df.sort(time_col)
    if ensure_datetime:
        out = out.with_columns(pl.col(time_col).cast(pl.Datetime))

    # Step 2: rolling mean/std on the chosen column
    roll = (
        out.rolling(index_column=time_col, period=period, closed=closed)
        .agg([
            pl.col(value_col).mean().alias(f"{value_col}_roll_mean"),
            pl.col(value_col).std().alias(f"{value_col}_roll_std"),
        ])
    )

    # Step 3: join stats back
    out = out.join(roll, on=time_col, how="left")

    mu = pl.col(f"{value_col}_roll_mean")
    sd = pl.col(f"{value_col}_roll_std")
    x  = pl.col(value_col)

    # Step 4: null out spikes beyond ± sigma*std
    out = out.with_columns(
        pl.when( (x < mu - sigma * sd) | (x > mu + sigma * sd) )
          .then(None)
          .otherwise(x)
          .alias(output_col)
    )

    if not keep_stats:
        out = out.drop([f"{value_col}_roll_mean", f"{value_col}_roll_std"])

    return out

