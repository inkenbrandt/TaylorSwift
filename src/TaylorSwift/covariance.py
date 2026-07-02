from __future__ import annotations

import numpy as np
from scipy import fft as _fft


def calc_cov(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return float("nan")
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 2:
        return float("nan")
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (n - 1))


def calc_MSE(y) -> float:
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if not np.any(mask):
        return float("nan")
    y = y[mask]
    return float(np.mean((y - y.mean()) ** 2))


class _PreparedSeries:
    """Per-array state for the FFT lag-covariance search.

    Preparing a series once (finite mask, centering, forward FFT, prefix
    sums) lets :func:`build_covariance_dict` share the expensive transforms
    across every pair the series participates in.
    """

    __slots__ = ("data", "finite", "any_finite", "all_finite", "centered",
                 "L", "_spec", "_mask_spec", "_prefix")

    def __init__(self, arr, L: int):
        self.data = np.asarray(arr, dtype=float)
        self.finite = np.isfinite(self.data)
        self.any_finite = bool(self.finite.any())
        self.all_finite = bool(self.finite.all())
        # Centre with the global finite mean: covariance is shift-invariant
        # and centred data keeps the one-pass formula in _max_cov_pair
        # well-conditioned even for signals with large means (CO2 density,
        # pressure, ...).
        self.centered = (
            np.where(self.finite, self.data - self.data[self.finite].mean(), 0.0)
            if self.any_finite
            else None
        )
        self.L = L
        self._spec = None
        self._mask_spec = None
        self._prefix = None

    @property
    def spec(self):
        """rfft of the centred, zero-filled series."""
        if self._spec is None:
            self._spec = _fft.rfft(self.centered, self.L)
        return self._spec

    @property
    def mask_spec(self):
        """rfft of the finite-mask indicator (NaN-aware path only)."""
        if self._mask_spec is None:
            self._mask_spec = _fft.rfft(self.finite.astype(np.float64), self.L)
        return self._mask_spec

    @property
    def prefix(self):
        """Prefix sums of the centred series: sum(centered[a:b]) in O(1)."""
        if self._prefix is None:
            self._prefix = np.concatenate(([0.0], np.cumsum(self.centered)))
        return self._prefix


def _pad_length(n: int, lag: int) -> int:
    """FFT length with zero-padding to L >= n + max|k|, so the circular
    correlation is free of wrap-around for every lag in [-lag, lag]."""
    if n == 0:
        return 1
    return _fft.next_fast_len(n + min(lag, n - 1))


def _max_cov_pair(px: _PreparedSeries, py: _PreparedSeries, lag: int):
    """Lag search shared by calc_max_covariance and build_covariance_dict."""
    n = len(px.data)
    if len(py.data) != n:
        raise ValueError(
            f"calc_max_covariance requires equal-length inputs "
            f"(got {n} and {len(py.data)})"
        )

    # Lags with a non-empty overlap (mirrors the slicing of the naive loop).
    lags = [k for k in range(-lag, lag + 1) if n - abs(k) > 0]
    if not lags:
        return []

    if not px.any_finite or not py.any_finite:
        return [(lags[0], float("nan"))]

    # One inverse transform yields the sliding products at every lag:
    # sxy_all[k % L] = sum_i centered_x[i + k] * centered_y[i].
    L = px.L
    sxy_all = _fft.irfft(px.spec * np.conj(py.spec), L)

    covs = np.empty(len(lags))
    if px.all_finite and py.all_finite:
        # Prefix sums give each lag's segment sums in O(1).
        cx, cy = px.prefix, py.prefix
        for j, k in enumerate(lags):
            nk = n - abs(k)
            if nk < 2:
                covs[j] = np.nan
                continue
            if k >= 0:
                sx, sy = cx[n] - cx[k], cy[nk]
            else:
                sx, sy = cx[nk], cy[n] - cy[-k]
            covs[j] = (sxy_all[k % L] - sx * sy / nk) / (nk - 1.0)
    else:
        # NaN-aware path: pair counts and segment sums restricted to samples
        # where both series are finite, via correlations with the masks.
        sx_all = _fft.irfft(px.spec * np.conj(py.mask_spec), L)
        sy_all = _fft.irfft(px.mask_spec * np.conj(py.spec), L)
        nk_all = np.rint(  # integer pair counts
            _fft.irfft(px.mask_spec * np.conj(py.mask_spec), L)
        )
        for j, k in enumerate(lags):
            i = k % L
            nk = nk_all[i]
            if nk < 2:
                covs[j] = np.nan
                continue
            covs[j] = (sxy_all[i] - sx_all[i] * sy_all[i] / nk) / (nk - 1.0)

    # Selection mirrors max(candidates, key=abs): NaNs never displace an
    # earlier candidate, and a leading NaN is never displaced.
    abs_covs = np.abs(covs)
    if np.isnan(abs_covs[0]):
        best = 0
    else:
        best = int(np.nanargmax(abs_covs))
    return [(lags[best], float(covs[best]))]


def calc_max_covariance(x, y, lag: int = 10):
    """Find the lag in ``[-lag, lag]`` that maximises ``|cov(x, y)|``.

    Equivalent to computing :func:`calc_cov` on the overlapping segments at
    every lag, but all lags are evaluated in a single pass: the sliding
    products come from one FFT cross-correlation and the per-lag means from
    prefix sums, instead of a full mask + mean + covariance traversal per lag.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    L = _pad_length(len(x), lag)
    return _max_cov_pair(_PreparedSeries(x, L), _PreparedSeries(y, L), lag)


def calc_covar(Ux, Uy, Uz, Ts, Q, pV) -> dict[str, float]:
    data = {"Ux": Ux, "Uy": Uy, "Uz": Uz, "Ts": Ts, "Q": Q, "pV": pV}
    out = {}
    for k1, v1 in data.items():
        for k2, v2 in data.items():
            out[f"{k1}-{k2}"] = calc_cov(v1, v2)
    return out


def build_covariance_dict(
    velocities: dict[str, np.ndarray], variables: dict[str, np.ndarray], lag: int = 10
) -> dict[str, float]:
    """Max-lag covariance for every velocity-variable pair.

    Each distinct input array is prepared (and Fourier-transformed) exactly
    once, no matter how many pairs it appears in.
    """
    prepared: dict[int, _PreparedSeries] = {}

    def _prep(arr) -> _PreparedSeries:
        key = id(arr)
        p = prepared.get(key)
        if p is None:
            a = np.asarray(arr, dtype=float)
            p = _PreparedSeries(a, _pad_length(len(a), lag))
            prepared[key] = p
        return p

    out = {}
    for ik, iv in velocities.items():
        pi = _prep(iv)
        for jk, jv in variables.items():
            result = _max_cov_pair(pi, _prep(jv), lag)
            out[f"{ik}-{jk}"] = result[0][1] if result else calc_cov(iv, jv)
    return out
