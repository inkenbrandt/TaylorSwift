from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from KDEpy import FFTKDE
from scipy.interpolate import interp1d

from .frame_utils import rolling_median_centered


def despike(arr, nstd: float = 4.5) -> np.ndarray:
    """
    Remove outliers from a time series using a standard deviation threshold.

    Parameters:
    -----------

    arr (_type_): sorted numpy array of values to despike
    nstd (float, optional): Number of standard deviations to use as the threshold. Defaults to 4.5.

    Returns
    -------
    np.ndarray: despiked array with outliers replaced by NaN and interpolated values.
    """
    arr = np.asarray(arr, dtype=float)
    stdd = np.nanstd(arr) * nstd
    avg = np.nanmean(arr)
    avgdiff = stdd - np.abs(arr - avg)
    y = np.where(avgdiff >= 0, arr, np.nan)
    nans = np.isnan(y)
    valid = np.where(~nans)[0]
    if len(valid) > 0:
        y[nans] = np.interp(np.where(nans)[0], valid, y[~nans])
    return y


def despike_ewma_fb(df_column: pd.Series,
                    span: int | float,
                    delta: float) -> np.ndarray:
    """
    Despike a time series using the exponential weighted moving average (EWMA) forward-backward method.

    Parameters
    ----------
    df_column : pd.Series
        The time series to despike.
    span : int | float
        The span of the EWMA.
    delta : float
        The threshold for identifying outliers.

    Returns
    -------
    np.ndarray
        The despiked time series.
    """
    fwd = pd.Series.ewm(df_column, span=span).mean()
    bwd = pd.Series.ewm(df_column[::-1], span=span).mean()
    stacked_ewma = np.vstack((fwd, bwd[::-1]))
    np_fbewma = np.mean(stacked_ewma, axis=0)
    np_spikey = np.array(df_column)
    cond_delta = np.abs(np_spikey - np_fbewma) > delta
    return np.where(cond_delta, np.nan, np_spikey)


def despike_med_mod(df_column: pd.Series,
                    win: int = 800,
                    fill_na: bool = True,
                    addNoise: bool = False) -> pd.Series:
    """Despike a time series using the median filter method.

    Parameters
    ----------
    df_column : pd.Series
        The time series to despike.
    win : int, optional
        The window size for the median filter. Defaults to 800.
    fill_na : bool, optional
        Whether to fill NaN values with interpolated values. Defaults to True.
    addNoise : bool, optional
        Whether to add noise to the filled NaN values. Defaults to False.

    Returns
    -------
    pd.Series
        The despiked time series.
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError(
            "despike_med_mod requires statsmodels (used for the robust "
            "linear-model fit). Install it with 'pip install statsmodels' "
            "or reinstall TaylorSwift with its declared dependencies."
        ) from exc

    np_spikey = np.array(df_column)
    y = df_column.interpolate().bfill().ffill()
    x = rolling_median_centered(df_column, win).to_pandas()
    X = sm.add_constant(x)
    mod_rlm = sm.RLM(y, X)
    mod_fit = mod_rlm.fit(maxiter=300, scale_est="mad")
    cond_delta = np.abs(mod_fit.resid) > 3 * mod_fit.scale
    np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
    nanind = np.array(np.where(np.isnan(np_remove_outliers)))[0]
    data_out = pd.Series(np_remove_outliers, index=df_column.index)
    if fill_na:
        data_out = data_out.interpolate()
        data_outnaind = data_out.index[nanind]
        rando = (
            np.random.default_rng().normal(
                scale=mod_fit.scale, size=len(data_outnaind)
            )
            if addNoise
            else 0.0
        )
        data_out.loc[data_outnaind] = data_out.loc[data_outnaind] + rando
    return data_out


def despike_quart_filter(
    df_column: pd.Series,
    win: int = 600,
    fill_na: bool = True,
    top_quant: float = 0.97,
    bot_quant: float = 0.03,
    thresh: float | pd.Series | None = None,
) -> pd.Series:
    """
    Despike a time series using the quartile filter method.

    Parameters
    ----------
    df_column : pd.Series
        The time series to despike.
    win : int, optional
        The window size for the rolling quantiles. Defaults to 600.
    fill_na : bool, optional
        Whether to fill NaN values with interpolated values. Defaults to True.
    top_quant : float, optional
        The upper quantile to use for filtering. Defaults to 0.97.
    bot_quant : float, optional
        The lower quantile to use for filtering. Defaults to 0.03.
    thresh : float | pd.Series | None, optional
        The threshold for identifying outliers. If None, the difference between the upper and lower quantiles is used. Defaults to None.

    Returns
    -------
    pd.Series
        The despiked time series.
    """
    upper = df_column.rolling(win, center=True).quantile(top_quant)
    lower = df_column.rolling(win, center=True).quantile(bot_quant)
    med = df_column.rolling(win, center=True).median()
    upper = upper.interpolate().bfill().ffill()
    lower = lower.interpolate().bfill().ffill()
    med = med.interpolate().bfill().ffill()
    threshold = (upper - lower) if thresh is None else thresh
    cleaned = df_column.where((df_column - med).abs() <= threshold, np.nan)
    if fill_na:
        cleaned = cleaned.interpolate().bfill().ffill()
    return cleaned


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

        x_grid, y_grid = FFTKDE(kernel="gaussian", bw=bw).fit(bulk).evaluate(2**10)
        peak = y_grid.max()
        if peak <= 0:
            break

        # Evaluate at *all* clean samples; points outside the KDE grid get
        # fill_value=0 — ensuring extreme outliers are always flagged.
        f_density = interp1d(
            x_grid, y_grid, kind="linear", fill_value=0.0, bounds_error=False
        )
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
        interp_func = interp1d(
            idx[valid], data[valid], kind="linear", fill_value="extrapolate"
        )
        data = interp_func(idx)

        iter_count += 1

    return data


def _validate_kde_options(prob_threshold, max_iter, bulk_iqr):
    if not np.isfinite(prob_threshold) or not 0 < prob_threshold < 1:
        raise ValueError("prob_threshold must be between 0 and 1 (exclusive)")
    if (isinstance(max_iter, bool)
            or not isinstance(max_iter, (int, np.integer)) or max_iter < 0):
        raise ValueError("max_iter must be a non-negative integer")
    if bulk_iqr is not None and (not np.isfinite(bulk_iqr) or bulk_iqr <= 0):
        raise ValueError("bulk_iqr must be positive and finite, or None")


def polars_ukde_despike(
    df: pl.DataFrame,
    col_name: str,
    prob_threshold: float = 1e-4,
    max_iter: int = 1,
    *,
    bulk_iqr: float | None = 4.0,
) -> pl.DataFrame:
    """Add ``{col_name}_cleaned`` using FFT-based UKDE despiking.

    The default is one pass. Lower ``prob_threshold`` removes fewer samples;
    increasing ``max_iter`` permits repeated trimming. Zero passes is a no-op.
    ``bulk_iqr`` sets the KDE fitting range to median +/- this multiple of
    IQR. Increase it to retain broader tails, or use None to fit all finite
    values. Values outside the fitted KDE grid have zero density, so lowering
    the probability threshold alone cannot protect them. The bandwidth uses
    the IQR-based Silverman rule.

    Input columns are preserved. Spikes and non-finite values become null and
    internal gaps are linearly interpolated; leading/trailing gaps stay null
    (no extrapolation). Fewer than four finite values or zero IQR skips spike
    detection. Missing values are still interpolated unless max_iter is zero.
    Numeric integer columns produce floating-point cleaned values.
    """
    _validate_kde_options(prob_threshold, max_iter, bulk_iqr)
    if not df.schema[col_name].is_numeric():
        raise TypeError(f"Column {col_name!r} must be numeric")
    output_col = f"{col_name}_cleaned"
    if max_iter == 0:
        return df.with_columns(pl.col(col_name).alias(output_col))
    values = df[col_name].cast(pl.Float64)
    values = values.set(~values.is_finite().fill_null(False), None)

    for _ in range(max_iter):
        data = values.to_numpy()
        finite = np.isfinite(data)
        clean_data = data[finite]
        spikes = np.zeros(len(data), dtype=bool)
        if len(clean_data) >= 4:
            med = np.median(clean_data)
            q25, q75 = np.percentile(clean_data, [25, 75])
            iqr = q75 - q25
            if iqr > 0:
                bw = 0.9 * (iqr / 1.349) * len(clean_data) ** (-0.2)
                bulk = clean_data if bulk_iqr is None else clean_data[
                    np.abs(clean_data - med) <= bulk_iqr * iqr
                ]
                if len(bulk) >= 4:
                    x_grid, y_grid = FFTKDE(kernel="gaussian", bw=bw).fit(
                        bulk
                    ).evaluate(2**12)
                    peak = y_grid.max()
                    if peak > 0:
                        densities = np.interp(
                            clean_data, x_grid, y_grid, left=0.0, right=0.0
                        )
                        spikes[finite] = densities < prob_threshold * peak
        values = values.set(pl.Series(spikes), None).interpolate()
        if not spikes.any():
            break
    return df.with_columns(values.alias(output_col))


def despike_dataframe(
    df: pd.DataFrame | pl.DataFrame,
    columns: list,
    prob_threshold: float = 1e-4,
    max_iter: int = 1,
    verbose: bool = False,
    *,
    bulk_iqr: float | None = 4.0,
) -> pd.DataFrame | pl.DataFrame:
    """Apply :func:`polars_ukde_despike` to selected columns of a copy.

    Accepts pandas or Polars and returns the same frame type, retaining row
    order, pandas index, and unselected columns. Missing column names are
    skipped. Only selected numeric columns are converted to Polars; no helper
    columns are added to the returned frame.

    ``prob_threshold``, ``max_iter`` (default one pass), and ``bulk_iqr`` are
    passed to :func:`polars_ukde_despike`. For gentler filtering, lower the
    probability threshold and increase bulk_iqr (or set it to None). Use
    max_iter=0 to bypass cleaning. Internal gaps are interpolated, while
    boundary gaps remain missing. With verbose=True, report changed finite
    samples separately from filled missing/non-finite samples.
    """
    _validate_kde_options(prob_threshold, max_iter, bulk_iqr)
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError("df must be a pandas or Polars DataFrame")
    is_polars = isinstance(df, pl.DataFrame)
    df_out = df.clone() if isinstance(df, pl.DataFrame) else df.copy()
    if max_iter == 0:
        return df_out

    for col in columns:
        if col not in df_out.columns:
            continue
        column = df_out.select(col) if is_polars else pl.from_pandas(df_out[[col]])
        cleaned = polars_ukde_despike(
            column, col, prob_threshold=prob_threshold, max_iter=max_iter,
            bulk_iqr=bulk_iqr,
        )[f"{col}_cleaned"]
        if verbose:
            original = column[col].cast(pl.Float64).to_numpy()
            result = cleaned.to_numpy()
            finite = np.isfinite(original)
            changed = finite & (~np.isfinite(result) | (original != result))
            filled = ~finite & np.isfinite(result)
            print(
                f"        despike {col:>10s}: {changed.sum():5d} samples replaced"
                f" ({100 * changed.sum() / max(1, finite.sum()):.3f}%);"
                f" {filled.sum()} missing/non-finite samples filled"
            )
        if is_polars:
            df_out = df_out.with_columns(cleaned.alias(col))
        else:
            df_out[col] = cleaned.to_numpy()
    return df_out


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
    if not isinstance(window_size, (int, np.integer)) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    data = np.asarray(data)
    spikes = np.zeros_like(data, dtype=bool)
    n = len(data)
    half = window_size // 2

    def _loop(indices):
        """Per-sample reference path, used for edges and non-1-D input."""
        for i in indices:
            start = max(0, i - half)
            end = min(n, i + half)
            window = data[start:end]
            if window.size == 0:
                continue
            mean = np.mean(window)
            std = np.std(window)
            if std > 0:  # Avoid division by zero
                spikes[i] = abs(data[i] - mean) / std > z_threshold

    width = 2 * half  # length of a full (untruncated) window [i-half, i+half)
    if data.ndim != 1 or half <= 0 or n < width:
        _loop(range(n))
        return spikes

    # Interior samples i in [half, n - half] all see a full window, so their
    # statistics come from one vectorised pass over a sliding-window view.
    # Chunked so the temporaries stay bounded on long records.
    windows = np.lib.stride_tricks.sliding_window_view(data, width)
    chunk = max(1, 4_000_000 // width)
    for s in range(0, windows.shape[0], chunk):
        block = windows[s : s + chunk]
        mean = block.mean(axis=1)
        std = block.std(axis=1)
        centre = data[s + half : s + half + block.shape[0]]
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.abs(centre - mean) / std
            spikes[s + half : s + half + block.shape[0]] = (std > 0) & (
                z > z_threshold
            )

    # Truncated windows at the boundaries keep the original per-sample path.
    _loop(range(half))
    _loop(range(n - half + 1, n))
    return spikes


def rolling_sigma_filter(
    df: pl.DataFrame,
    value_col: str = "Uz",
    time_col: str = "TIMESTAMP",
    period: str = "5s",
    sigma: float = 3.0,
    closed: Literal["left", "right", "both", "none"] = "both",
    output_col: str | None = None,  # default: f"{value_col}_filtered"
    keep_stats: bool = True,  # keep or drop the roll mean/std columns
    ensure_datetime: bool = True,  # cast time_col to pl.Datetime
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
    roll = out.rolling(index_column=time_col, period=period, closed=closed).agg(
        [
            pl.col(value_col).mean().alias(f"{value_col}_roll_mean"),
            pl.col(value_col).std().alias(f"{value_col}_roll_std"),
        ]
    )

    # Step 3: join stats back
    out = out.join(roll, on=time_col, how="left")

    mu = pl.col(f"{value_col}_roll_mean")
    sd = pl.col(f"{value_col}_roll_std")
    x = pl.col(value_col)

    # Step 4: null out spikes beyond ± sigma*std
    out = out.with_columns(
        pl.when((x < mu - sigma * sd) | (x > mu + sigma * sd))
        .then(None)
        .otherwise(x)
        .alias(output_col)
    )

    if not keep_stats:
        out = out.drop([f"{value_col}_roll_mean", f"{value_col}_roll_std"])

    return out
