"""Shared numerical kernels with explicit caller-specific data policies."""

import numpy as np


def detrend_linear(arr: np.ndarray, *, promote_float: bool = False) -> np.ndarray:
    """Remove a linear trend using finite samples, without mutating the input.

    Spectral processing preserves its input dtype; quality control promotes
    integer and lower-precision inputs to float64 before subtracting the trend.
    """
    out = arr.copy().astype(np.float64) if promote_float else arr.copy()
    valid = np.isfinite(arr)
    if valid.sum() < 2:
        return out
    t = np.arange(len(arr), dtype=np.float64)
    t_v = t[valid]
    a_v = arr[valid]
    t_c = t_v - t_v.mean()
    t_var = float(np.dot(t_c, t_c))
    if t_var == 0:
        return out - a_v.mean()
    slope = float(np.dot(t_c, a_v - a_v.mean())) / t_var
    intercept = a_v.mean() - slope * t_v.mean()
    out -= slope * t + intercept
    return out


def mad_outlier_mask(data: np.ndarray, threshold: float, *, ignore_nan: bool):
    """Modified Z-score mask, retaining each API's missing-value policy."""
    median_func = np.nanmedian if ignore_nan else np.median
    median = median_func(data)
    mad = median_func(np.abs(data - median))
    if mad == 0:
        if ignore_nan:
            return np.zeros(len(data), dtype=bool)
        return np.zeros_like(data, dtype=bool)
    modified_zscore = 0.6745 * (data - median) / mad
    return np.abs(modified_zscore) > threshold
