"""Utility functions for working with pandas and polars DataFrames."""

import pathlib

import numpy as np
import pandas as pd
import polars as pl


def to_pl_df(df):
    """Convert a pandas DataFrame to a polars DataFrame, or return the polars DataFrame as-is."""
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError("df must be pandas or polars DataFrame")


def to_same_type(df_like, df_pl: pl.DataFrame):
    """Convert a polars DataFrame to the same type as df_like (pandas or polars)."""
    return df_pl.to_pandas() if isinstance(df_like, pd.DataFrame) else df_pl


def get_series(df, col):
    """Get a column from a DataFrame as a polars Series, regardless of whether the DataFrame is pandas or polars."""
    if isinstance(df, pl.DataFrame):
        return df[col]
    if isinstance(df, pd.DataFrame):
        return pl.Series(col, df[col].to_numpy())
    raise TypeError("df must be pandas or polars DataFrame")


def assign(df, **cols):
    """Assign new columns to a DataFrame, regardless of whether it's pandas or polars."""
    if isinstance(df, pl.DataFrame):
        newcols = []
        for k, v in cols.items():
            if isinstance(v, pl.Series):
                s = v if v.name == k else v.alias(k)
            else:
                s = pl.Series(k, np.asarray(v))
            newcols.append(s)
        return df.with_columns(newcols)

    for k, v in cols.items():
        df[k] = np.asarray(v)
    return df


def interpolate_bfill_ffill(s):
    """Interpolate a Series, filling NaNs with backward and forward fill."""
    if not isinstance(s, pl.Series):
        s = pl.Series(np.asarray(s))
    return s.interpolate().fill_null(strategy="backward").fill_null(strategy="forward")


def rolling_median_centered(s, win: int):
    """Compute the rolling median of a Series, centered, with backward and forward fill for NaNs."""
    if not isinstance(s, pl.Series):
        s = pl.Series(np.asarray(s))
    return (
        s.rolling_median(window_size=win, center=True)
        .fill_null(strategy="backward")
        .fill_null(strategy="forward")
    )


def shift(s, n: int):
    """Shift a Series by n, regardless of whether it's pandas or polars."""
    if isinstance(s, pl.Series):
        return s.shift(n)
    return pl.Series(np.asarray(s)).shift(n)


def first_last_index_duration(df: pd.DataFrame, unit: str = "D") -> float:
    """
    Calculate the duration between the first and last valid indices of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the duration for.
    unit : str, optional
        The unit of the returned duration. Defaults to "D" (days).

    Returns
    -------
    float
        The duration between the first and last valid indices of the DataFrame.
    """
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        delta = df.last_valid_index() - df.first_valid_index()
        return delta / pd.to_timedelta(1, unit=unit)
    for name in ("TIMESTAMP", "timestamp", "time", "TIMESTAMP_START", "datetime"):
        if name in df.columns:
            timestamps = pd.to_datetime(df[name], errors="raise")
            seconds = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
            return seconds / 86400.0 if unit == "D" else seconds
    raise ValueError("Could not infer datetime information for duration.")


def normalize_input_frame(df, rename_func=None, rename_map=None, ts_col=None):
    """Normalize the input DataFrame to a pandas DataFrame, optionally renaming columns and converting a timestamp column to datetime.
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame or str or bytes or pathlib.Path
        The DataFrame to normalize, or a file path to read it from.
    rename_func : callable, optional
        A function to rename the columns of the DataFrame.
    rename_map : dict, optional
        A dictionary mapping old column names to new column names.
    ts_col : str, optional
        The name of the timestamp column to convert to datetime.

    Returns
    -------
    pd.DataFrame
        The normalized pandas DataFrame.
    """
    if isinstance(df, (str, bytes, pathlib.Path)):
        p = str(df)
        if p.lower().endswith(".parquet"):
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
    elif "polars" in type(df).__module__:
        df = df.to_pandas()
    elif not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas.DataFrame, a polars.DataFrame, or a file path"
        )

    if ts_col and ts_col in df.columns:
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
        except Exception:
            pass

    if rename_func is not None:
        try:
            if rename_map is not None:
                df = rename_func(df, rename_map=rename_map)
            else:
                df = rename_func(df)
        except TypeError:
            df = rename_func(df)
    return df
