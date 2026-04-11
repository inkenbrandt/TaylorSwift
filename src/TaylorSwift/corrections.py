"""
corrections.py — Spectral and flux corrections for eddy covariance data.

Implements the standard suite of corrections used in the micromet community:

DESPIKING (pre-processing, applied before spectral computation):
  0. Iterative UKDE despiking — kernel density estimation identifies and
     removes statistical outliers from the raw time series before FFTs are
     computed (Metzger et al. 2012).
     • ukde_despike          – scipy Gaussian KDE, iterative (small arrays)
     • polars_ukde_despike   – FFT-KDE via KDEpy, single-pass (large arrays)
     • despike_dataframe     – multi-column wrapper for pandas DataFrames

HIGH-FREQUENCY CORRECTIONS (attenuate flux at high frequencies):
  1. Sensor frequency response — first-order time constant (Moore 1986)
  2. Path averaging / line averaging along sonic paths (Kaimal et al. 1968)
  3. Scalar path averaging for IRGA optical path (Moore 1986)
  4. Sensor separation — lateral/longitudinal displacement between sonic
     and gas analyser (Moore 1986).  For IRGASON this is ~0 (integrated).
  5. Combined transfer function approach (Massman 2000)
  6. Analytical correction factor (Horst 1997)

LOW-FREQUENCY CORRECTIONS (attenuate flux at low frequencies):
  7. Block-average (finite averaging window) transfer function
  8. Linear detrend transfer function

DENSITY CORRECTIONS:
  9. Webb-Pearman-Leuning (WPL 1980) correction for open-path CO₂/H₂O
     fluxes measured as density.

References
----------
Moore, C.J. (1986). Frequency response corrections for eddy correlation
    systems. Boundary-Layer Meteorol., 37, 17–35.
Massman, W.J. (2000). A simple method for estimating frequency response
    corrections for eddy covariance systems. Agric. For. Meteorol., 104,
    185–198.
Massman, W.J. (2001). Reply to comment by Rannik on "A simple method..."
    Agric. For. Meteorol., 107, 247–251.
Horst, T.W. (1997). A simple formula for attenuation of eddy fluxes
    measured with first-order-response scalar sensors. Boundary-Layer
    Meteorol., 82, 219–233.
Kaimal, J.C. et al. (1968). Deriving power spectra from a three-component
    sonic anemometer. J. Appl. Meteorol., 7, 827–837.
Webb, E.K., Pearman, G.I. & Leuning, R. (1980). Correction of flux
    measurements for density effects due to heat and water vapour transfer.
    Quart. J. Roy. Meteor. Soc., 106, 85–100.
Leuning, R. (2007). The correct formula for the WPL correction.
    Boundary-Layer Meteorol., 126, 263–272.
Metzger, S., Junkermann, W., Mauder, M., Beyrich, F., Butterbach-Bahl, K.,
    Schmid, H. P., & Foken, T. (2012). Eddy-covariance flux measurements
    with a weight-shift microlight aircraft. Atmospheric Measurement
    Techniques, 5, 1699–1717.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import polars as pl
import pandas as pd
from KDEpy import FFTKDE
from scipy.interpolate import interp1d

from .constants import MOLAR_MASS, R_SPECIFIC, CP_DRY_AIR, T_ZERO_C


# ===================================================================
# Instrument configuration
# ===================================================================
@dataclass
class InstrumentConfig:
    """
    Physical parameters for the sonic anemometer and gas analyser.

    Default values are for the Campbell Scientific IRGASON (integrated
    open-path sonic anemometer + CO₂/H₂O gas analyser).
    """
    # --- Sonic anemometer ---
    sonic_path_length: float = 0.10        # sonic path length [m]
                                           # IRGASON: ~10 cm vertical path
    sonic_path_separation: float = 0.0     # separation between horizontal paths [m]
                                           # IRGASON: integrated, effectively 0

    # --- Gas analyser (IRGA) ---
    irga_path_length: float = 0.154        # optical path length [m]
                                           # IRGASON: 15.4 cm
    irga_path_diameter: float = 0.005      # optical path diameter [m]

    # --- Sensor separation ---
    # Distance between sonic measurement volume and IRGA measurement volume.
    # For IRGASON these are co-located, so separation ≈ 0.
    sensor_separation_lateral: float = 0.0   # perpendicular to wind [m]
    sensor_separation_longitudinal: float = 0.0  # parallel to wind [m]
    sensor_separation_vertical: float = 0.0  # vertical [m]

    # --- Time constants ---
    # First-order response time constants for the sensors.
    tau_sonic_T: float = 0.0               # sonic temperature [s] (essentially 0)
    tau_co2: float = 0.1                   # CO₂ sensor response [s]
                                           # IRGASON: ~0.1 s at 20 Hz
    tau_h2o: float = 0.1                   # H₂O sensor response [s]
    tau_T: float = 0.0                     # sonic T response [s] (virtual instant)

    # --- Instrument type ---
    irga_type: str = 'open_path'           # 'open_path' or 'enclosed_path'
    model: str = 'IRGASON'                 # instrument model name

    @property
    def sensor_separation_total(self) -> float:
        """Total sensor separation distance [m]."""
        return np.sqrt(
            self.sensor_separation_lateral ** 2 +
            self.sensor_separation_longitudinal ** 2 +
            self.sensor_separation_vertical ** 2
        )


def ukde_despike(series, prob_threshold=1e-4, max_iter=10):
    """
    Despike a time series using the iterative UKDE-Hybrid method.

    Identifies spikes as observations whose probability density (estimated by
    a Gaussian KDE) falls below ``prob_threshold`` times the peak density of
    the distribution.  Detected spikes are replaced with NaN and then
    re-filled by linear interpolation before the next iteration, so that each
    pass operates on a progressively cleaner signal.  Iteration stops when no
    new spikes are found or ``max_iter`` is reached.

    This is a numpy / scipy implementation that is well-suited for moderate-
    length arrays (up to ~10 000 samples). For large Polars DataFrames use
    :func:`polars_ukde_despike`, which substitutes an FFT-based KDE.

    Parameters
    ----------
    series : array-like
        1-D time series to despike (any numeric type; may contain NaN).
    prob_threshold : float, optional
        Fraction of the peak kernel density below which a sample is flagged
        as a spike.  Lower values are more permissive (fewer points removed);
        higher values are more aggressive.  Default is ``1e-4``.
    max_iter : int, optional
        Maximum number of despike iterations.  In practice convergence is
        typically reached within 2–4 passes.  Default is ``10``.

    Returns
    -------
    np.ndarray
        Cleaned 1-D array of the same length as *series*.  Spike positions
        are replaced with linearly interpolated values (or extrapolated at the
        ends).  The returned array contains no NaN unless the input was
        entirely NaN.

    Notes
    -----
    The method is an adaptation of the universal KDE despiking approach
    described in Metzger et al. (2012).  Unlike threshold-based methods
    (e.g. ±3 σ), KDE despiking is robust to skewed distributions and does not
    assume Gaussianity of the underlying signal.

    The KDE is fitted only on the *bulk* population — samples within 4 × IQR
    of the median — using an IQR-based Silverman bandwidth.  This prevents
    outliers from inflating the bandwidth or accumulating density in the tails
    of the estimate.  Samples that fall outside the fitted grid are assigned a
    density of zero and are always flagged regardless of ``prob_threshold``.

    References
    ----------
    Metzger, S., Junkermann, W., Mauder, M., Beyrich, F., Butterbach-Bahl, K.,
        Schmid, H. P., & Foken, T. (2012). Eddy-covariance flux measurements
        with a weight-shift microlight aircraft. Atmospheric Measurement
        Techniques, 5, 1699–1717. https://doi.org/10.5194/amt-5-1699-2012
    """
    data = np.array(series, dtype=float)
    n = len(data)
    iter_count = 0

    while iter_count < max_iter:
        clean_indices = ~np.isnan(data)
        n_clean = int(np.sum(clean_indices))
        if n_clean < 4:
            break

        clean_data = data[clean_indices]

        # --- Robust scale and bandwidth ---
        # Use IQR-normalised Silverman bandwidth so that outliers do not
        # inflate the bandwidth and become invisible in the KDE tails.
        med = np.median(clean_data)
        q25, q75 = np.percentile(clean_data, [25, 75])
        iqr = q75 - q25
        if iqr <= 0:
            break
        sigma_robust = iqr / 1.349
        bw = 0.9 * sigma_robust * n_clean ** (-0.2)

        # --- Fit KDE on bulk data only ---
        # Exclude candidate spikes (> 4 IQR from median) from the KDE fit so
        # that a small number of extreme values cannot distort the density
        # estimate for the main population.
        bulk_mask = (clean_data >= med - 4.0 * iqr) & (clean_data <= med + 4.0 * iqr)
        bulk = clean_data[bulk_mask]
        if len(bulk) < 4:
            break

        x_grid, y_grid = FFTKDE(kernel='gaussian', bw=bw).fit(bulk).evaluate(2 ** 10)
        peak = y_grid.max()
        if peak <= 0:
            break

        # Evaluate at *all* clean samples; points outside the KDE grid get
        # fill_value=0 — ensuring extreme outliers are always flagged.
        f_density = interp1d(x_grid, y_grid, kind='linear',
                             fill_value=0.0, bounds_error=False)
        densities = np.zeros(n)
        densities[clean_indices] = f_density(clean_data)

        # Flag samples whose normalised density is below the threshold
        spikes = (densities < (prob_threshold * peak)) & clean_indices

        if not np.any(spikes):
            break  # Convergence reached

        # Replace spikes with NaN, then linearly interpolate so the next
        # pass operates on a smooth, spike-free signal
        data[spikes] = np.nan
        idx = np.arange(n)
        valid = ~np.isnan(data)
        if valid.sum() < 2:
            break
        interp_func = interp1d(idx[valid], data[valid],
                               kind='linear', fill_value='extrapolate')
        data = interp_func(idx)

        iter_count += 1

    return data


def polars_ukde_despike(df: pl.DataFrame, col_name: str, prob_threshold: float = 1e-4) -> pl.DataFrame:
    """
    Despike a single column of a Polars DataFrame using an FFT-based KDE.

    A performance-optimised variant of :func:`ukde_despike` designed for
    large high-frequency datasets.  Uses ``KDEpy.FFTKDE`` (O(n log n))
    instead of ``scipy.stats.gaussian_kde`` (O(n²)), so it is practical on
    full 30-minute blocks at 20 Hz (≈ 36 000 samples per column).

    The KDE is fitted on the *bulk* of the distribution (values within
    4 × IQR of the median) using an IQR-based Silverman bandwidth, so that
    extreme outliers cannot distort either the bandwidth or the density
    estimate.  Samples outside the fitted grid receive a density of zero
    and are therefore always flagged regardless of threshold.

    Because only a single KDE pass is performed (no iteration), this
    function is faster but slightly less thorough than the iterative
    :func:`ukde_despike`.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame containing the column to clean.
    col_name : str
        Name of the column to despike.  The column must be numeric.
    prob_threshold : float, optional
        Fraction of the peak density below which a sample is flagged as a
        spike.  Default is ``1e-4``.

    Returns
    -------
    pl.DataFrame
        A new DataFrame identical to *df* except that a column named
        ``{col_name}_cleaned`` is added (or replaced if it already exists).
        Flagged samples are set to ``null`` and then filled by Polars'
        built-in linear interpolation.

    Notes
    -----
    Original NaN values in *col_name* are assigned a density of zero, so
    they are also replaced by interpolated values in the output column.

    See Also
    --------
    ukde_despike : Iterative scipy-based version for smaller arrays.
    despike_dataframe : Apply despiking to multiple columns of a pandas
        DataFrame in one call.

    References
    ----------
    Metzger, S., Junkermann, W., Mauder, M., Beyrich, F., Butterbach-Bahl, K.,
        Schmid, H. P., & Foken, T. (2012). Eddy-covariance flux measurements
        with a weight-shift microlight aircraft. Atmospheric Measurement
        Techniques, 5, 1699–1717. https://doi.org/10.5194/amt-5-1699-2012
    Silverman, B. W. (1986). Density Estimation for Statistics and Data
        Analysis. Chapman & Hall, London.
    """
    # 1. Extract numpy array (Polars zero-copy where possible)
    series_np = df[col_name].to_numpy()
    clean_mask = ~np.isnan(series_np)
    clean_data = series_np[clean_mask]

    if len(clean_data) < 4:
        return df.with_columns(pl.col(col_name).alias(f"{col_name}_cleaned"))

    # 2. Robust bandwidth: IQR-based Silverman rule (outlier-resistant)
    med = np.median(clean_data)
    q25, q75 = np.percentile(clean_data, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return df.with_columns(pl.col(col_name).alias(f"{col_name}_cleaned"))
    sigma_robust = iqr / 1.349
    bw = 0.9 * sigma_robust * len(clean_data) ** (-0.2)

    # 3. Fit FFT-KDE on bulk data only (exclude > 4 IQR from median)
    #    Spikes excluded from the fit cannot inflate the bandwidth or
    #    smuggle themselves into the tail of the estimated density.
    bulk_mask = (clean_data >= med - 4.0 * iqr) & (clean_data <= med + 4.0 * iqr)
    bulk = clean_data[bulk_mask]
    if len(bulk) < 4:
        return df.with_columns(pl.col(col_name).alias(f"{col_name}_cleaned"))

    x_grid, y_grid = FFTKDE(kernel='gaussian', bw=bw).fit(bulk).evaluate(2 ** 12)
    peak = y_grid.max()

    # 4. Evaluate density at every sample; values outside the KDE grid get 0
    f_density = interp1d(x_grid, y_grid, kind='linear',
                         fill_value=0.0, bounds_error=False)
    densities = f_density(series_np)

    # 5. Polars masking and linear interpolation — fast even on 10⁶-row frames
    return df.with_columns(
        pl.when(pl.lit(densities) < (prob_threshold * peak))
        .then(None)           # mark spike as null
        .otherwise(pl.col(col_name))
        .alias(f"{col_name}_cleaned")
    ).with_columns(
        pl.col(f"{col_name}_cleaned").interpolate()   # fill gaps linearly
    )


def despike_dataframe(
    df: pd.DataFrame,
    columns: list,
    prob_threshold: float = 1e-4,
    max_iter: int = 10,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Apply iterative UKDE despiking to multiple columns of a pandas DataFrame.

    Loops over *columns*, calling :func:`ukde_despike` on each in turn and
    storing the cleaned values back into a copy of *df*.  Columns that are
    absent from *df* are silently skipped so that a fixed default column list
    can be used across different instrument setups.

    This function is the recommended entry-point for the ``run_cospectra.py``
    workflow, where despiking is applied to the raw high-frequency time series
    before FFT-based spectral computation.

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        Raw high-frequency eddy covariance DataFrame (e.g. as returned by
        :func:`eccospectra.io.read_toa5`).  The DataFrame is not modified
        in-place; a copy / clone is returned with the same type as the input.
    columns : list of str
        Column names to despike.  Typical choices for an IRGASON dataset are
        ``['Ux', 'Uy', 'Uz', 'Ts', 'CO2', 'H2O']``.  Missing column names
        are skipped without raising an error.
    prob_threshold : float, optional
        Passed directly to :func:`ukde_despike`.  Default is ``1e-4``.
    max_iter : int, optional
        Maximum iterations per column passed to :func:`ukde_despike`.
        Default is ``10``.
    verbose : bool, optional
        If ``True``, print a one-line summary per column showing how many
        samples changed.  Default is ``False``.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with spike-contaminated samples in *columns* replaced
        by linearly interpolated values.

    See Also
    --------
    ukde_despike : Underlying single-column despiking implementation.
    polars_ukde_despike : FFT-based variant for Polars DataFrames.

    Examples
    --------
    >>> from eccospectra.io import read_toa5
    >>> from eccospectra.corrections import despike_dataframe
    >>> df, meta = read_toa5('mydata.dat')
    >>> df_clean = despike_dataframe(
    ...     df,
    ...     columns=['Ux', 'Uy', 'Uz', 'Ts', 'CO2', 'H2O'],
    ...     prob_threshold=1e-4,
    ...     verbose=True,
    ... )
    """
    _is_polars = isinstance(df, pl.DataFrame)
    df_out = df.clone() if _is_polars else df.copy()

    for col in columns:
        if col not in df_out.columns:
            continue

        # Extract as float64 NumPy array (zero-copy when possible)
        original = df_out[col].to_numpy().astype(np.float64)

        cleaned = ukde_despike(original,
                               prob_threshold=prob_threshold,
                               max_iter=max_iter)

        if verbose:
            scale = np.nanstd(original)
            tol = 1e-6 * scale if scale > 0 else 1e-10
            n_changed = int(np.sum(
                np.abs(cleaned - np.where(np.isnan(original), cleaned, original)) > tol
            ))
            print(f"        despike {col:>10s}: {n_changed:5d} samples replaced")

        if _is_polars:
            df_out = df_out.with_columns(pl.Series(col, cleaned))
        else:
            df_out[col] = cleaned

    return df_out


# ===================================================================
# Transfer functions
# ===================================================================

def tf_block_average(freq: np.ndarray, averaging_period: float) -> np.ndarray:
    """
    Transfer function for block (running mean) averaging.

    Describes flux loss at low frequencies due to the finite averaging window.
    From Kaimal et al. (1968) Eq. 32 and Moore (1986).

    H(n) = [1 - sin(π n τ) / (π n τ)]²

    where τ is the averaging period in seconds.

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    averaging_period : float
        Averaging period [minutes].

    Returns
    -------
    np.ndarray
        Transfer function values (0 to 1).
    """
    tau = averaging_period * 60.0  # convert to seconds
    x = np.pi * freq * tau
    # Avoid division by zero at freq=0
    with np.errstate(divide='ignore', invalid='ignore'):
        tf = np.where(
            x > 1e-10,
            (1.0 - np.sin(x) / x) ** 2,
            0.0
        )
    return tf


def tf_linear_detrend(freq: np.ndarray, averaging_period: float) -> np.ndarray:
    """
    Transfer function for linear detrending.

    Rannik & Vesala (1999) showed that linear detrending has a slightly
    different low-frequency response than block averaging. This uses the
    analytical form from Rannik & Vesala (1999).

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    averaging_period : float
        Averaging period [minutes].

    Returns
    -------
    np.ndarray
        Transfer function values (0 to 1).
    """
    tau = averaging_period * 60.0
    x = np.pi * freq * tau
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc = np.where(x > 1e-10, np.sin(x) / x, 1.0)
        # Moncrieff et al. (2004) robust form
        tf = 1.0 - sinc ** 2
    return np.clip(tf, 0.0, 1.0)


def tf_first_order_response(freq: np.ndarray, tau: float) -> np.ndarray:
    """
    Transfer function for a first-order sensor response.

    T_s(n) = 1 / (1 + (2π n τ)²)

    where τ is the sensor time constant [s].  This describes the
    attenuation caused by a sensor that cannot follow rapid fluctuations
    (Moore 1986, Eq. 3).

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    tau : float
        First-order time constant [s].

    Returns
    -------
    np.ndarray
        Transfer function values (0 to 1).
    """
    if tau <= 0:
        return np.ones_like(freq)
    return 1.0 / (1.0 + (2.0 * np.pi * freq * tau) ** 2)


def tf_sonic_line_averaging(
    freq: np.ndarray,
    u_mean: float,
    path_length: float,
) -> np.ndarray:
    """
    Transfer function for sonic anemometer line-averaging.

    The sonic samples the wind averaged over its path length l. This
    attenuates fluctuations at wavelengths comparable to l. From
    Kaimal et al. (1968) Eq. 15:

    T(k₁l) = sin²(k₁l/2) / (k₁l/2)²

    Converted to frequency via Taylor's hypothesis: k₁ = 2πn/U.

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    u_mean : float
        Mean wind speed [m/s].
    path_length : float
        Sonic path length [m].

    Returns
    -------
    np.ndarray
        Transfer function values (0 to 1).
    """
    if u_mean <= 0.1 or path_length <= 0:
        return np.ones_like(freq)

    k1_l = 2.0 * np.pi * freq * path_length / u_mean
    half_k1l = k1_l / 2.0

    with np.errstate(divide='ignore', invalid='ignore'):
        tf = np.where(
            half_k1l > 1e-10,
            (np.sin(half_k1l) / half_k1l) ** 2,
            1.0
        )
    return tf


def tf_scalar_path_averaging(
    freq: np.ndarray,
    u_mean: float,
    path_length: float,
) -> np.ndarray:
    """
    Transfer function for scalar (IRGA) path averaging.

    For a cylindrical optical path of length l, the transfer function
    has the same sinc² form as line averaging (Moore 1986):

    T(n) = sin²(πnl/U) / (πnl/U)²

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    u_mean : float
        Mean wind speed [m/s].
    path_length : float
        IRGA optical path length [m].

    Returns
    -------
    np.ndarray
        Transfer function values (0 to 1).
    """
    # Same functional form as sonic line averaging
    return tf_sonic_line_averaging(freq, u_mean, path_length)


def tf_sensor_separation(
    freq: np.ndarray,
    u_mean: float,
    separation: float,
) -> np.ndarray:
    """
    Transfer function for lateral sensor separation.

    When the sonic and gas analyser are separated by distance d
    perpendicular to the mean wind, eddies passing one sensor may
    not be fully sampled by the other. From Moore (1986):

    T(n) = exp(-9.9 (nd/U)^1.5)

    For IRGASON (integrated sensor), separation ≈ 0 and this returns 1.

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    u_mean : float
        Mean wind speed [m/s].
    separation : float
        Total sensor separation distance [m].

    Returns
    -------
    np.ndarray
        Transfer function values (0 to 1).
    """
    if separation <= 0.001 or u_mean <= 0.1:
        return np.ones_like(freq)

    nd_U = freq * separation / u_mean
    return np.exp(-9.9 * nd_U ** 1.5)


# ===================================================================
# Combined transfer function (Massman 2000 approach)
# ===================================================================

def combined_transfer_function(
    freq: np.ndarray,
    u_mean: float,
    instrument: InstrumentConfig,
    averaging_period: float = 30.0,
    flux_type: str = 'wT',
) -> np.ndarray:
    """
    Compute the combined spectral transfer function for a given flux.

    Multiplies all applicable individual transfer functions following
    Massman (2000). The total transfer function describes the fraction
    of the true cospectral density that is actually measured at each
    frequency.

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    u_mean : float
        Mean horizontal wind speed [m/s].
    instrument : InstrumentConfig
        Instrument parameters.
    averaging_period : float
        Averaging period [minutes].
    flux_type : str
        Which flux: 'wT', 'wu', 'wCO2', 'wH2O'.

    Returns
    -------
    np.ndarray
        Combined transfer function T(n) at each frequency.
    """
    # 1. Low-frequency: block averaging
    T_low = tf_block_average(freq, averaging_period)

    # 2. Sonic line-averaging (affects w always, and u for momentum flux)
    T_sonic_w = tf_sonic_line_averaging(freq, u_mean, instrument.sonic_path_length)

    # 3. High-frequency: sensor-specific
    if flux_type == 'wT':
        T_sensor = tf_first_order_response(freq, instrument.tau_T)
        T_path_scalar = np.ones_like(freq)  # sonic T has same path as w
        T_sep = np.ones_like(freq)

    elif flux_type == 'wu':
        # Momentum: both components are from the sonic
        T_sensor = np.ones_like(freq)  # no extra time constant
        T_path_scalar = tf_sonic_line_averaging(
            freq, u_mean, instrument.sonic_path_length)
        T_sep = np.ones_like(freq)

    elif flux_type == 'wCO2':
        T_sensor = tf_first_order_response(freq, instrument.tau_co2)
        T_path_scalar = tf_scalar_path_averaging(
            freq, u_mean, instrument.irga_path_length)
        T_sep = tf_sensor_separation(
            freq, u_mean, instrument.sensor_separation_total)

    elif flux_type == 'wH2O':
        T_sensor = tf_first_order_response(freq, instrument.tau_h2o)
        T_path_scalar = tf_scalar_path_averaging(
            freq, u_mean, instrument.irga_path_length)
        T_sep = tf_sensor_separation(
            freq, u_mean, instrument.sensor_separation_total)
    else:
        raise ValueError(f"Unknown flux_type: {flux_type}")

    # Combined: product of all transfer functions
    T_combined = T_low * T_sonic_w * T_sensor * T_path_scalar * T_sep

    return np.clip(T_combined, 1e-10, 1.0)


# ===================================================================
# Correction factor computation
# ===================================================================

def _kaimal_cospec_model(f_nd, flux_type='wT'):
    """
    Kaimal et al. (1972) / Massman (2000) model cospectrum.

    Returns n Co(n) / cov(w'x') as a function of dimensionless
    frequency f = nz/U.  Used as the "true" cospectral shape for
    computing correction factors.
    """
    if flux_type in ('wT', 'wCO2', 'wH2O'):
        # Massman (2000) Eq. 12 — scalar cospectra
        # n Co / cov = a f / (1 + b f)^(7/4)
        # with a = 12.92, b = 26.7 for heat (Kaimal 1972 neutral)
        return 12.92 * f_nd / (1.0 + 26.7 * f_nd) ** (7.0 / 4.0)
    elif flux_type == 'wu':
        # Momentum cospectrum (Kaimal 1972)
        return 9.6 * f_nd / (1.0 + 14.0 * f_nd) ** (7.0 / 4.0)
    else:
        return 12.92 * f_nd / (1.0 + 26.7 * f_nd) ** (7.0 / 4.0)


def compute_spectral_correction_factor(
    u_mean: float,
    z_eff: float,
    instrument: InstrumentConfig,
    averaging_period: float = 30.0,
    flux_type: str = 'wT',
    n_freqs: int = 10000,
    f_nd_range: tuple = (1e-4, 1e3),
) -> float:
    """
    Compute the multiplicative correction factor for a flux.

    The correction factor CF is:
        CF = ∫ Co_model(f) d(ln f)  /  ∫ T(f) · Co_model(f) d(ln f)

    where Co_model is the Kaimal (1972) model cospectrum and T(f) is the
    combined transfer function.  The corrected flux is:
        F_corrected = CF × F_measured

    This follows Massman (2000) and is equivalent to the approach used in
    EddyPro and other standard EC processing software.

    Parameters
    ----------
    u_mean : float
        Mean wind speed [m/s].
    z_eff : float
        Effective measurement height (z - d) [m].
    instrument : InstrumentConfig
        Instrument parameters.
    averaging_period : float
        Averaging period [minutes].
    flux_type : str
        'wT', 'wu', 'wCO2', or 'wH2O'.
    n_freqs : int
        Number of frequencies for numerical integration.
    f_nd_range : tuple
        Range of dimensionless frequencies for integration.

    Returns
    -------
    float
        Correction factor (≥ 1.0). Multiply measured flux by this value.
    """
    if u_mean < 0.5:
        return np.nan

    # Dimensionless frequency grid
    f_nd = np.logspace(np.log10(f_nd_range[0]),
                       np.log10(f_nd_range[1]), n_freqs)

    # Convert to natural frequency: n = f_nd * U / z
    freq = f_nd * u_mean / z_eff

    # Model cospectrum (the "true" shape)
    Co_model = _kaimal_cospec_model(f_nd, flux_type)

    # Combined transfer function
    T = combined_transfer_function(freq, u_mean, instrument,
                                   averaging_period, flux_type)

    # Integration in log-frequency space: ∫ Co d(ln f)
    # Numerator: integral of true cospectrum
    num = np.trapz(Co_model, np.log(f_nd))

    # Denominator: integral of attenuated cospectrum
    den = np.trapz(T * Co_model, np.log(f_nd))

    if den > 1e-12:
        cf = num / den
    else:
        cf = np.nan

    return max(cf, 1.0)  # correction factor should always be ≥ 1


def horst_analytical_correction(
    u_mean: float,
    z_eff: float,
    tau_eff: float,
    flux_type: str = 'wT',
) -> float:
    """
    Horst (1997) analytical correction factor.

    A simple closed-form approximation that avoids numerical integration.
    Good for quick estimates; less accurate than the full Massman approach
    for complex instrument configurations.

    CF = 1 / (1 + (2π f_m τ_eff)^α)

    where f_m is the dimensionless frequency of the cospectral peak,
    τ_eff is the effective time constant, and α depends on the cospectral
    model (α ≈ 1 for the Kaimal model).

    Parameters
    ----------
    u_mean : float
        Mean wind speed [m/s].
    z_eff : float
        Effective measurement height [m].
    tau_eff : float
        Effective combined time constant [s] (from all high-freq sources).
    flux_type : str
        'wT', 'wu', 'wCO2', or 'wH2O'.

    Returns
    -------
    float
        Correction factor (≥ 1.0).
    """
    if u_mean < 0.5 or tau_eff <= 0:
        return 1.0

    # Cospectral peak frequency (dimensionless) — neutral stability
    if flux_type == 'wu':
        f_peak = 0.085  # Kaimal (1972) momentum
    else:
        f_peak = 0.065  # Kaimal (1972) scalars

    # Convert to natural frequency
    n_peak = f_peak * u_mean / z_eff

    # Horst (1997) Eq. 9: α ≈ 7/8 for Kaimal cospectrum
    alpha = 7.0 / 8.0
    cf = 1.0 / (1.0 - (2.0 * np.pi * n_peak * tau_eff) ** alpha)

    return max(cf, 1.0)


# ===================================================================
# WPL (Webb-Pearman-Leuning) density correction
# ===================================================================

def wpl_correction(
    Fc_raw: float,
    Fe_raw: float,
    H: float,
    T_mean: float,
    P_mean: float,
    co2_mean: float,
    h2o_mean: float,
) -> dict:
    """
    Webb-Pearman-Leuning (1980) density correction for open-path fluxes.

    Open-path IRGAs measure number density (or mass density). Vertical
    transport of heat and water vapour causes density fluctuations that
    are not actual flux. The WPL correction accounts for this.

    This implements the standard WPL equations as formulated by
    Webb et al. (1980) and clarified by Leuning (2007).

    Parameters
    ----------
    Fc_raw : float
        Uncorrected CO₂ flux (w'ρc') [mg m⁻² s⁻¹].
    Fe_raw : float
        Uncorrected H₂O flux (w'ρv') [g m⁻² s⁻¹] (= mmol m⁻² s⁻¹ × 18.02).
    H : float
        Sensible heat flux [W m⁻²].
    T_mean : float
        Mean air temperature [°C].
    P_mean : float
        Mean atmospheric pressure [kPa].
    co2_mean : float
        Mean CO₂ density [mg m⁻³].
    h2o_mean : float
        Mean H₂O density [g m⁻³].

    Returns
    -------
    dict with keys:
        'Fc_wpl'  : WPL-corrected CO₂ flux [mg m⁻² s⁻¹]
        'Fe_wpl'  : WPL-corrected H₂O flux [g m⁻² s⁻¹]
        'Fc_correction' : additive CO₂ correction term
        'Fe_correction' : additive H₂O correction term
        'mu'      : ratio of molecular weights (dry air / water)
        'sigma'   : ρv / ρd (mixing ratio by density)
    """
    Md = MOLAR_MASS['air_dry']
    Mv = MOLAR_MASS['h2o']
    mu = Md / Mv
    Rd = R_SPECIFIC['dry_air']
    cp = CP_DRY_AIR

    T_K = T_mean + T_ZERO_C
    P_Pa = P_mean * 1000.0  # kPa to Pa

    # Dry air density [kg m⁻³]
    # ρv in kg/m³ (h2o_mean is in g/m³)
    rho_v = h2o_mean * 1e-3
    rho_d = (P_Pa / (Rd * T_K)) - rho_v

    # σ = ρv / ρd
    sigma = rho_v / rho_d if rho_d > 0 else 0.0

    # WPL correction for H₂O flux (Webb et al. 1980, Eq. 25)
    # Fe_wpl = (1 + μσ) × [Fe_raw + (ρv/T) × (H / (ρd cp))]
    Fe_wpl = (1.0 + mu * sigma) * (
        Fe_raw + (h2o_mean / T_K) * (H / (rho_d * cp))
    )

    # WPL correction for CO₂ flux (Webb et al. 1980, Eq. 24)
    # Fc_wpl = Fc_raw + μ (ρc/ρd) Fe_raw_kg + (1+μσ)(ρc/T)(H/(ρd cp))
    # Note: Fe_raw needs to be in kg/m²/s for the cross-term
    Fe_raw_kg = Fe_raw * 1e-3  # g → kg
    Fc_wpl = (
        Fc_raw
        + mu * (co2_mean * 1e-6 / rho_d) * Fe_raw_kg * 1e6
        + (1.0 + mu * sigma) * (co2_mean / T_K) * (H / (rho_d * cp))
    )

    return {
        'Fc_wpl': Fc_wpl,
        'Fe_wpl': Fe_wpl,
        'Fc_correction': Fc_wpl - Fc_raw,
        'Fe_correction': Fe_wpl - Fe_raw,
        'mu': mu,
        'sigma': sigma,
    }


# ===================================================================
# Apply corrections to SpectralResult objects
# ===================================================================

def apply_spectral_corrections(
    results,
    site_config,
    instrument: InstrumentConfig,
    apply_high_freq: bool = True,
    apply_low_freq: bool = True,
    apply_wpl: bool = True,
    method: str = 'massman',
    verbose: bool = False,
):
    """
    Apply all spectral and density corrections to a list of SpectralResults.

    Parameters
    ----------
    results : list[SpectralResult]
        Output from process_file().
    site_config : SiteConfig
        Station configuration.
    instrument : InstrumentConfig
        Instrument parameters.
    apply_high_freq : bool
        Apply high-frequency spectral corrections (default True).
    apply_low_freq : bool
        Apply low-frequency corrections (default True).
    apply_wpl : bool
        Apply WPL density correction for open-path CO₂/H₂O (default True).
        Only applied if instrument.irga_type == 'open_path'.
    method : str
        'massman' for full numerical integration (Massman 2000),
        'horst' for the analytical approximation (Horst 1997).
    verbose : bool
        Print correction factors.

    Returns
    -------
    list[SpectralResult]
        The same results list, with corrections applied in-place.
        New attributes added to qc_flags:
          'cf_wT', 'cf_wu', 'cf_wCO2', 'cf_wH2O' — correction factors
          'cov_wT_corrected', etc. — corrected covariances
          'wpl_Fc', 'wpl_Fe' — WPL-corrected fluxes
    """
    z_eff = site_config.z_eff

    for res in results:
        if not np.isfinite(res.u_mean) or res.u_mean < 0.5:
            continue

        # ---------------------------------------------------------------
        # Spectral correction factors
        # ---------------------------------------------------------------
        for flux_type in ['wT', 'wu', 'wCO2', 'wH2O']:
            if method == 'massman':
                cf = compute_spectral_correction_factor(
                    u_mean=res.u_mean,
                    z_eff=z_eff,
                    instrument=instrument,
                    averaging_period=site_config.averaging_period,
                    flux_type=flux_type,
                )
            elif method == 'horst':
                # Compute effective time constant for this flux
                if flux_type == 'wT':
                    tau_eff = instrument.tau_T
                elif flux_type == 'wu':
                    tau_eff = 0.0
                elif flux_type == 'wCO2':
                    tau_eff = instrument.tau_co2
                elif flux_type == 'wH2O':
                    tau_eff = instrument.tau_h2o
                else:
                    tau_eff = 0.0

                # Add path-averaging equivalent time constant
                # τ_path ≈ l / (2π U) for a path of length l
                if res.u_mean > 0.5:
                    tau_path = instrument.irga_path_length / (
                        2.0 * np.pi * res.u_mean)
                    tau_eff = np.sqrt(tau_eff**2 + tau_path**2)

                cf = horst_analytical_correction(
                    res.u_mean, z_eff, tau_eff, flux_type)
            else:
                raise ValueError(f"Unknown method: {method}")

            res.qc_flags[f'cf_{flux_type}'] = cf

            # Apply correction factor to covariances
            cov_attr = f'cov_{flux_type}'
            cov_raw = getattr(res, cov_attr)
            if np.isfinite(cf) and np.isfinite(cov_raw):
                res.qc_flags[f'{cov_attr}_corrected'] = cov_raw * cf

            # Also correct the cospectral arrays by dividing by T(f)
            # at each frequency bin (spectral correction)
            if apply_high_freq and len(res.freq) > 0 and np.isfinite(cf):
                T_f = combined_transfer_function(
                    res.freq, res.u_mean, instrument,
                    site_config.averaging_period, flux_type)

                cosp_attr = f'cosp_{flux_type}'
                cosp = getattr(res, cosp_attr)
                if len(cosp) > 0:
                    cosp_corrected = cosp / T_f
                    setattr(res, cosp_attr, cosp_corrected)

                    # Update normalised version
                    ncosp_attr = f'ncosp_{flux_type}'
                    cov_val = getattr(res, cov_attr)
                    if abs(cov_val) > 1e-12:
                        setattr(res, ncosp_attr, cosp_corrected / cov_val)

        # ---------------------------------------------------------------
        # WPL density correction (open-path only)
        # ---------------------------------------------------------------
        if apply_wpl and instrument.irga_type == 'open_path':
            # Need pressure — check if available, otherwise estimate
            P_mean = getattr(res, 'P_mean', None)
            if P_mean is None or not np.isfinite(P_mean):
                P_mean = 101.3  # standard atmosphere [kPa]

            # Get mean scalar densities from raw covariances context
            co2_mean = getattr(res, 'co2_mean', None)
            h2o_mean = getattr(res, 'h2o_mean', None)

            if (co2_mean is not None and h2o_mean is not None and
                    np.isfinite(co2_mean) and np.isfinite(h2o_mean)):

                # Use spectrally-corrected covariances if available
                cov_wCO2 = res.qc_flags.get(
                    'cov_wCO2_corrected', res.cov_wCO2)
                cov_wH2O = res.qc_flags.get(
                    'cov_wH2O_corrected', res.cov_wH2O)
                H_corr = res.qc_flags.get(
                    'cov_wT_corrected', res.cov_wT) * 1200.0

                wpl = wpl_correction(
                    Fc_raw=cov_wCO2,
                    Fe_raw=cov_wH2O,
                    H=H_corr,
                    T_mean=res.T_mean,
                    P_mean=P_mean,
                    co2_mean=co2_mean,
                    h2o_mean=h2o_mean,
                )
                res.qc_flags['wpl_Fc'] = wpl['Fc_wpl']
                res.qc_flags['wpl_Fe'] = wpl['Fe_wpl']
                res.qc_flags['wpl_Fc_correction'] = wpl['Fc_correction']
                res.qc_flags['wpl_Fe_correction'] = wpl['Fe_correction']

        if verbose:
            ts = res.timestamp_start
            tstr = ts.strftime('%H:%M') if ts is not None else '??'
            cfs = [f"{res.qc_flags.get(f'cf_{ft}', np.nan):.3f}"
                   for ft in ['wT', 'wu', 'wCO2', 'wH2O']]
            print(f"  {tstr}  CF: wT={cfs[0]} wu={cfs[1]} "
                  f"wCO2={cfs[2]} wH2O={cfs[3]}")

    return results


# ===================================================================
# Convenience: store mean scalars during processing
# ===================================================================

def enrich_results_with_means(results, df, site_config):
    """
    Add mean scalar values (CO₂, H₂O, pressure) to SpectralResult objects.

    These are needed for WPL corrections but are not computed during the
    core spectral processing. Call this after process_file() and before
    apply_spectral_corrections().

    Parameters
    ----------
    results : list[SpectralResult]
    df : pd.DataFrame
        The original high-frequency DataFrame with columns like
        CO2_density, H2O_density, PA, etc.
    site_config : SiteConfig
    """
    # Normalise to Polars (no-op if already a pl.DataFrame)
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    for res in results:
        if res.timestamp_start is None or res.timestamp_end is None:
            continue

        sub = df.filter(
            (pl.col('TIMESTAMP') >= res.timestamp_start) &
            (pl.col('TIMESTAMP') <  res.timestamp_end)
        )

        if len(sub) == 0:
            continue

        if 'CO2_density' in sub.columns:
            val = sub['CO2_density'].mean()
            if val is not None:
                res.co2_mean = float(val)
        if 'H2O_density' in sub.columns:
            val = sub['H2O_density'].mean()
            if val is not None:
                res.h2o_mean = float(val)
        if 'PA' in sub.columns:
            val = sub['PA'].mean()
            if val is not None:
                res.P_mean = float(val)

    return results
