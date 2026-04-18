from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .frame_utils import rolling_median_centered


def despike(arr, nstd: float = 4.5) -> np.ndarray:
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


def despike_ewma_fb(
    df_column: pd.Series, span: int | float, delta: float
) -> np.ndarray:
    fwd = pd.Series.ewm(df_column, span=span).mean()
    bwd = pd.Series.ewm(df_column[::-1], span=span).mean()
    stacked_ewma = np.vstack((fwd, bwd[::-1]))
    np_fbewma = np.mean(stacked_ewma, axis=0)
    np_spikey = np.array(df_column)
    cond_delta = np.abs(np_spikey - np_fbewma) > delta
    return np.where(cond_delta, np.nan, np_spikey)


def despike_med_mod(
    df_column: pd.Series, win: int = 800, fill_na: bool = True, addNoise: bool = False
) -> pd.Series:
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
            np.random.normal(scale=mod_fit.scale, size=len(data_outnaind))
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


def polars_ukde_despike(
    df: pl.DataFrame, col_name: str, prob_threshold: float = 1e-4
) -> pl.DataFrame:
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

    x_grid, y_grid = FFTKDE(kernel="gaussian", bw=bw).fit(bulk).evaluate(2**12)
    peak = y_grid.max()

    # 4. Evaluate density at every sample; values outside the KDE grid get 0
    f_density = interp1d(
        x_grid, y_grid, kind="linear", fill_value=0.0, bounds_error=False
    )
    densities = f_density(series_np)

    # 5. Polars masking and linear interpolation — fast even on 10⁶-row frames
    return df.with_columns(
        pl.when(pl.lit(densities) < (prob_threshold * peak))
        .then(None)  # mark spike as null
        .otherwise(pl.col(col_name))
        .alias(f"{col_name}_cleaned")
    ).with_columns(
        pl.col(f"{col_name}_cleaned").interpolate()  # fill gaps linearly
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

        cleaned = ukde_despike(
            original, prob_threshold=prob_threshold, max_iter=max_iter
        )

        if verbose:
            scale = np.nanstd(original)
            tol = 1e-6 * scale if scale > 0 else 1e-10
            n_changed = int(
                np.sum(
                    np.abs(cleaned - np.where(np.isnan(original), cleaned, original))
                    > tol
                )
            )
            print(f"        despike {col:>10s}: {n_changed:5d} samples replaced")

        if _is_polars:
            df_out = df_out.with_columns(pl.Series(col, cleaned))
        else:
            df_out[col] = cleaned

    return df_out
