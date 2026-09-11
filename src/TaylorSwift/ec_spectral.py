"""
ec_spectral.py
==============

Identify, plot, and correct spectral (frequency-response) issues in eddy
covariance data.

Adapted from the supplied standalone ec_spectral module. Models referenced there:

* Moncrieff, Massheder, de Bruin, et al. (1997) J. Hydrol. 188-189, 589-611
    - lag detection by cross-correlation maximisation (Fig. 2)
    - Kaimal et al. (1972) model cospectra, Eqs 12-18 (incl. the -1.1 exponent
      typo fix in Eq. 14)
    - transfer functions of Moore (1986), Appendix A (Tr, Td, Tm, Tw, Ts, Tt)
    - integral flux-loss estimate, Eq. 10; Fig. 5 / 6 / 10 style plots
* Massman & Clement (2004) Handbook of Micrometeorology, Ch. 4
    - block-averaging (high-pass) transfer function, Eq. 4.1
    - generalised cospectral model, Eq. 4.2, fitted by bounded least squares
    - analytical correction factor F, Eq. 4.4 (Massman 2000)
    - relative uncertainty dF/F, Eq. 4.7 (C = 1.2, d_alpha = 0.2,
      d_fx/fx = 0.4-0.5)
    - ogives (Fig. 4.9) to check for low-frequency flux contributions

Typical use
-----------
    from TaylorSwift import ec_spectral as ecs
    import numpy as np

    sysc = ecs.ECSystem(fs=20, z_ref=3.0, d=0.6, lateral_sep=0.0)   # IRGASON-like
    df   = ecs.load_toa5("TOA5_site_ts_data.dat")

    u, v, w = ecs.double_rotation(df.Ux.values, df.Uy.values, df.Uz.values)
    lag, lags, r = ecs.find_lag(w, df.CO2.values, sysc.fs)
    c = ecs.shift(df.CO2.values, lag)

    co  = ecs.cospectrum(w, c, sysc.fs)
    fit = ecs.fit_cospectrum(co["f_bin"], co["fCo_bin"], fs=sysc.fs, T=sysc.avg_period)
    ubar = np.hypot(u, v).mean()

    F_int = ecs.correction_factor_integral(sysc, ubar, fx=fit["fx"], mu=fit["mu"])
    taus  = ecs.equivalent_time_constants(sysc, ubar)
    F_an  = ecs.correction_factor_analytical(fit["fx"], taus["tau_b"], taus["tau_e"])
    dF    = ecs.correction_uncertainty(fit["fx"], taus["tau_b"], taus["tau_e"])

    ecs.plot_cospectrum(co, fit, sysc, ubar)          # Moncrieff Fig. 5
    ecs.plot_transfer_functions(sysc, ubar)           # Moncrieff Fig. 10
    ecs.plot_flux_loss(sysc)                          # Moncrieff Fig. 6
    ecs.plot_ogive(co)                                # Massman Fig. 4.9
    ecs.plot_correction_vs_wind(sysc, eta_x=0.11)     # Massman Fig. 4.6

Then correct a half-hourly flux table with `correct_flux_table`.

Note: spectral correction is applied to the *covariance* (w'c'), i.e. before
the WPL density correction (Massman & Clement §3.1; Moncrieff §4).

Dependencies: numpy, scipy, pandas, matplotlib.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import xlogy

from .transfer_functions import _trapezoid as _trapz


def _pyplot():
    """Load plotting only when requested."""
    import matplotlib.pyplot as plt

    return plt


def _positive(name, value, allow_zero=False):
    value = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value < 0 if allow_zero else value <= 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )


def _pair(w, c):
    w, c = np.asarray(w, float), np.asarray(c, float)
    if w.ndim != 1 or c.shape != w.shape or w.size < 3:
        raise ValueError(
            "signals must be equal-length 1-D arrays with at least three samples"
        )
    return w, c


# molecular diffusion coefficients in air, m2 s-1 (used for laminar tube loss)
D_MOL = {"co2": 1.6e-5, "h2o": 2.5e-5, "ch4": 2.2e-5}
NU_AIR = 1.5e-5  # kinematic viscosity of air, m2 s-1


# Kaimal et al. (1972) flat-terrain neutral peak frequencies, eta'_x = fx (z-d)/u
# (Massman & Clement §3.2) and the GLEES / Griffin values from their Tables 4.1-4.2
ETA_X_REFERENCE = {
    "kaimal_uw": 0.085,
    "kaimal_wT": 0.079,
    "glees_uw": 0.07,
    "glees_wT": 0.11,
    "glees_wc": 0.11,
    "griffin_neutral_wc_open": 0.047,
    "griffin_neutral_wT": 0.073,
}


# ---------------------------------------------------------------------------
# System description
# ---------------------------------------------------------------------------
@dataclass
class ECSystem:
    """Physical description of the eddy covariance system.

    Defaults approximate a CSAT3/EC150-style open-path system on a short
    tower. Set `closed_path=True` and the tube parameters for a closed-path
    analyser (Moncrieff et al.'s EdiSol used r = 3 mm, X = 9 m, 6 L/min).
    """

    fs: float = 20.0  # sampling frequency, Hz
    avg_period: float = 1800.0  # flux averaging period Tb, s
    z_ref: float = 3.0  # measurement height, m
    d: float = 0.0  # zero-plane displacement, m
    sonic_path: float = (
        0.115  # sonic vertical path length, m (CSAT3 ~0.115, Solent 0.15)
    )
    scalar_path: float = (
        0.15  # open-path IRGA path length, m (EC150/IRGASON ~0.15, LI-7500 0.125)
    )
    lateral_sep: float = 0.0  # lateral sonic-IRGA separation, m
    tau_sonic: float = 0.0  # sonic first-order time constant, s (0 = ignore)
    tau_scalar: float = 0.0  # scalar-sensor first-order time constant, s
    detrend: str = "block"  # 'block' (EasyFlux/EddyPro default) or 'running_mean'
    running_mean_tau: float = (
        200.0  # s, only for detrend='running_mean' (EdiSol used 200 s)
    )
    closed_path: bool = False
    tube_radius: float = 3e-3  # m
    tube_length: float = 9.0  # m
    flow_rate: float = 6.0  # L min-1
    scalar: str = "co2"  # 'co2' | 'h2o' | 'ch4'

    def __post_init__(self):
        for name in (
            "fs",
            "avg_period",
            "running_mean_tau",
            "tube_radius",
            "flow_rate",
        ):
            _positive(name, getattr(self, name))
        for name in (
            "d",
            "sonic_path",
            "scalar_path",
            "lateral_sep",
            "tau_sonic",
            "tau_scalar",
            "tube_length",
        ):
            _positive(name, getattr(self, name), allow_zero=True)
        _positive("z_ref - d", self.zd)
        if self.detrend not in ("block", "running_mean"):
            raise ValueError("detrend must be 'block' or 'running_mean'")
        if self.scalar not in D_MOL:
            raise ValueError("scalar must be 'co2', 'h2o', or 'ch4'")

    @property
    def zd(self) -> float:
        return self.z_ref - self.d

    @property
    def tube_velocity(self) -> float:
        """Mean discharge velocity U in the tube, m s-1."""
        q = self.flow_rate / 1000.0 / 60.0
        return q / (np.pi * self.tube_radius**2)

    @property
    def reynolds(self) -> float:
        return 2.0 * self.tube_radius * self.tube_velocity / NU_AIR

    @property
    def tube_laminar(self) -> bool:
        return self.reynolds < 2300.0


# ---------------------------------------------------------------------------
# I/O and preprocessing
# ---------------------------------------------------------------------------
def load_toa5(path: str, usecols=None) -> pd.DataFrame:
    """Read a Campbell Scientific TOA5 file (CR6 'ts_data' tables)."""
    df = pd.read_csv(
        path,
        skiprows=[0, 2, 3],
        header=0,
        usecols=usecols,
        na_values=["NAN", "-9999"],
        low_memory=False,
    )
    if "TIMESTAMP" in df.columns:
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
        df = df.set_index("TIMESTAMP")
    return df


def despike(x: np.ndarray, window: int = 600, n_mad: float = 6.0) -> np.ndarray:
    """Rolling median/MAD despiking; spikes are linearly interpolated.

    A simpler stand-in for Højstrup (1993) / Vickers & Mahrt (1997).
    """
    s = pd.Series(np.asarray(x, dtype=float))
    med = s.rolling(window, center=True, min_periods=10).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=10).median()
    bad = (s - med).abs() > n_mad * 1.4826 * mad
    bad |= s.isna()
    out = s.mask(bad).interpolate(limit_direction="both")
    return out.to_numpy()


def double_rotation(u, v, w):
    """Double coordinate rotation (McMillen 1986): v_bar = 0, w_bar = 0."""
    u, v, w = (np.asarray(a, float) for a in (u, v, w))
    theta = np.arctan2(np.nanmean(v), np.nanmean(u))
    u1 = u * np.cos(theta) + v * np.sin(theta)
    v1 = -u * np.sin(theta) + v * np.cos(theta)
    phi = np.arctan2(np.nanmean(w), np.nanmean(u1))
    u2 = u1 * np.cos(phi) + w * np.sin(phi)
    w2 = -u1 * np.sin(phi) + w * np.cos(phi)
    return u2, v1, w2


def find_lag(
    w,
    c,
    fs: float,
    max_lag_s: float = 2.0,
    expected_s: float | None = None,
    search_window_s: float = 0.5,
):
    """Lag (in samples) that maximises |r_wc| (Moncrieff et al. 1997, §3.3, Fig. 2).

    Positive lag means `c` lags `w` (typical for a tube). If `expected_s` is
    given, the search is restricted to ±search_window_s around it, as EdiSol
    did (0.5 s window around the previously determined value).

    Returns (lag_samples, lags_s, r) where r is the correlation at each lag.
    """
    w, c = _pair(w, c)
    _positive("fs", fs)
    _positive("max_lag_s", max_lag_s, allow_zero=True)
    _positive("search_window_s", search_window_s, allow_zero=True)
    if expected_s is not None and not np.isfinite(expected_s):
        raise ValueError("expected_s must be finite")
    lo, hi = (
        (-max_lag_s, max_lag_s)
        if expected_s is None
        else (expected_s - search_window_s, expected_s + search_window_s)
    )
    lags = np.arange(
        max(int(np.ceil(lo * fs)), 3 - len(w)),
        min(int(np.floor(hi * fs)), len(w) - 3) + 1,
    )
    if not lags.size:
        raise ValueError("lag search has no overlapping samples")
    r = np.full(lags.size, np.nan)
    for i, k in enumerate(lags):
        x, y = (w[: len(w) - k], c[k:]) if k >= 0 else (w[-k:], c[: len(c) + k])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            continue
        x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
        den = np.linalg.norm(x) * np.linalg.norm(y)
        if den > 0:
            r[i] = np.dot(x, y) / den
    if not np.isfinite(r).any():
        raise ValueError("lag requires overlapping finite, nonconstant samples")
    return int(lags[np.nanargmax(np.abs(r))]), lags / fs, r


def shift(c, lag: int):
    """Align scalar with w after `find_lag` (edges filled with the edge value)."""
    c = np.asarray(c, float)
    if (
        c.ndim != 1
        or not c.size
        or not isinstance(lag, (int, np.integer))
        or abs(lag) >= c.size
    ):
        raise ValueError(
            "lag must be an integer smaller than the nonempty signal length"
        )
    if lag == 0:
        return c.copy()
    out = np.roll(c, -lag)
    if lag > 0:
        out[-lag:] = c[-1]
    else:
        out[:-lag] = c[0]
    return out


# ---------------------------------------------------------------------------
# Cospectra and ogives
# ---------------------------------------------------------------------------
def cospectrum(
    w, c, fs: float, taper: bool = True, nbins: int = 60, detrend: str = "linear"
):
    """One-sided cospectrum Co_wc(f) such that sum(Co * df) = cov(w, c).

    Follows the Griffin-forest recipe in Massman & Clement §3.3: linear
    detrend, Hamming taper (variance-restored), FFT, then logarithmic bin
    averaging. Returns a dict with raw and binned quantities plus the ogive.
    """
    w, c = _pair(w, c)
    _positive("fs", fs)
    if not isinstance(nbins, (int, np.integer)) or nbins < 1:
        raise ValueError("nbins must be a positive integer")
    if detrend not in ("linear", "constant", "block"):
        raise ValueError("detrend must be linear, constant, or block")
    ok = np.isfinite(w) & np.isfinite(c)
    if ok.sum() < 3:
        raise ValueError("cospectrum requires at least three finite pairs")
    w = np.interp(np.arange(len(w)), np.flatnonzero(ok), w[ok])
    c = np.interp(np.arange(len(c)), np.flatnonzero(ok), c[ok])
    if np.ptp(w) == 0 or np.ptp(c) == 0:
        raise ValueError("normalised cospectrum requires nonconstant signals")
    N = len(w)
    t = np.arange(N)
    if detrend == "linear":
        w = w - np.polyval(np.polyfit(t, w, 1), t)
        c = c - np.polyval(np.polyfit(t, c, 1), t)
    else:
        w = w - w.mean()
        c = c - c.mean()
    cov = np.mean(w * c)
    scale = np.sqrt(np.mean(w * w) * np.mean(c * c))
    if scale == 0 or abs(cov) <= 1e-12 * scale:
        raise ValueError(
            "normalised cospectrum is undefined for zero or near-zero covariance"
        )

    if taper:
        win = np.hamming(N)
        win /= np.sqrt(np.mean(win**2))  # restore variance
        w = w * win
        c = c * win

    W = np.fft.rfft(w)
    C = np.fft.rfft(c)
    f = np.fft.rfftfreq(N, 1.0 / fs)
    df = fs / N
    co = 2.0 * np.real(W * np.conj(C)) / N**2 / df  # density, units of cov / Hz
    co[0] = 0.0
    if N % 2 == 0:
        co[-1] /= 2.0  # Nyquist bin not doubled
    f, co = f[1:], co[1:]
    integral = np.sum(co) * df
    if abs(integral) <= 1e-12 * scale or integral * cov <= 0:
        raise ValueError(
            "tapered covariance cannot be safely rescaled; try taper=False"
        )
    co *= cov / integral  # restore exact covariance lost to tapering

    # ogive: cumulative cospectral power from high frequency downward, normalised
    ogive = np.cumsum((co * df)[::-1])[::-1] / cov

    # logarithmic bin averaging
    edges = np.logspace(np.log10(f[0]), np.log10(f[-1]), nbins + 1)
    idx = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, nbins - 1)
    f_bin: Any = []
    co_bin: Any = []
    for b in range(nbins):
        m = idx == b
        if m.sum() > 0:
            f_bin.append(np.exp(np.mean(np.log(f[m]))))
            co_bin.append(np.mean(co[m]))
    f_bin = np.array(f_bin)
    co_bin = np.array(co_bin)

    return {
        "f": f,
        "Co": co,
        "cov": cov,
        "ogive": ogive,
        "f_bin": f_bin,
        "Co_bin": co_bin,
        "fCo_bin": f_bin * co_bin / cov,  # frequency-weighted, normalised
        "N": N,
        "fs": fs,
    }


def inertial_slope(f_bin, fCo_bin, fmin: float, fmax: float):
    """Log-log slope of f*Co in [fmin, fmax]. Expect -4/3 for f*Co (-7/3 for Co)."""
    f_bin, fCo_bin = np.asarray(f_bin), np.asarray(fCo_bin)
    m = (f_bin >= fmin) & (f_bin <= fmax) & (fCo_bin > 0)
    if m.sum() < 3:
        return np.nan
    return np.polyfit(np.log(f_bin[m]), np.log(fCo_bin[m]), 1)[0]


# ---------------------------------------------------------------------------
# Model cospectra
# ---------------------------------------------------------------------------
def kaimal_cospectrum(fk, kind: str = "scalar", zL: float = 0.0):
    """Normalised frequency-weighted cospectrum n*Co(n) of Kaimal et al. (1972)
    as written in Moncrieff et al. (1997) Eqs 12-18, versus fk = n(z-d)/u.

    kind: 'scalar' (w'c', w'T') or 'momentum' (u'w').
    zL: (z-d)/L; >0 stable uses Eq. 12-14 (with the -1.1 exponent), <=0 unstable.
    """
    if kind not in ("scalar", "momentum"):
        raise ValueError("kind must be scalar or momentum")
    fk = np.asarray(fk, float)
    if zL > 0:
        if kind == "momentum":
            A = 0.124 * (1 + 7.9 * zL) ** 0.75
        else:
            A = 0.284 * (1 + 6.4 * zL) ** 0.75
        B = 2.34 * A ** (-1.1)
        return fk / (A + B * fk**2.1)
    if kind == "momentum":
        return np.where(
            fk < 0.24,
            20.78 * fk / (1 + 31 * fk) ** 1.575,
            12.66 * fk / (1 + 9.6 * fk) ** 2.4,
        )
    return np.where(
        fk < 0.54,
        12.92 * fk / (1 + 26.7 * fk) ** 1.375,
        4.378 * fk / (1 + 3.8 * fk) ** 2.4,
    )


def massman_cospectrum(f, A0: float, fx: float, mu: float, m: float = 0.75):
    """Frequency-weighted cospectrum f*Co(f), Massman & Clement Eq. 4.2.

    m = 3/4 gives the -7/3 inertial-subrange law for cospectra
    (m = 3/2 gives -5/3 for spectra). mu = broadness: ~0.5 flat-terrain
    unstable, ~7/6 stable, ~1 at the complex GLEES site.
    """
    for name, value in (("fx", fx), ("mu", mu), ("m", m)):
        _positive(name, value)
    x = np.asarray(f, float) / fx
    return (
        A0 * x / (1.0 + m * x ** (2.0 * mu)) ** ((1.0 / (2.0 * mu)) * ((m + 1.0) / m))
    )


def fit_cospectrum(
    f_bin,
    fCo_bin,
    fs: float,
    T: float,
    fmin: float | None = None,
    fmax: float | None = None,
    fix_m: bool = True,
    p0=None,
) -> dict:
    """Fit Eq. 4.2 to a binned frequency-weighted cospectrum (bounded least squares).

    Uses bounded nonlinear least squares (trust-region reflective).
    Only the central portion is fitted, as in Massman & Clement §3.2, to
    avoid contaminating fx and mu with high- or low-frequency attenuation.
    Defaults: fmin = 2/T, fmax = fs/10.
    Returns dict(A0, fx, mu, m, r2, fmin, fmax).
    """
    f_bin, y = _pair(f_bin, fCo_bin)
    _positive("fs", fs)
    _positive("T", T)
    fmin = 2.0 / T if fmin is None else fmin
    fmax = fs / 10.0 if fmax is None else fmax
    msk = (f_bin >= fmin) & (f_bin <= fmax) & np.isfinite(y)
    msk &= np.isfinite(f_bin) & (f_bin > 0)
    if msk.sum() < (4 if fix_m else 5) or not np.any(y[msk] > 0) or np.ptp(y[msk]) == 0:
        raise ValueError(
            "fit requires enough finite, varying bins with positive signal"
        )
    fx0 = f_bin[msk][np.argmax(y[msk])]
    if fix_m:

        def fun(f, A0, fx, mu):
            return massman_cospectrum(f, A0, fx, mu, 0.75)

        p0 = p0 if p0 is not None else (y[msk].max() * 2, fx0, 0.6)
        bounds = ([0, f_bin[msk].min() / 3, 0.1], [np.inf, f_bin[msk].max() * 3, 3.0])
    else:
        fun = massman_cospectrum
        p0 = p0 if p0 is not None else (y[msk].max() * 2, fx0, 0.6, 0.75)
        bounds = (
            [0, f_bin[msk].min() / 3, 0.1, 0.4],
            [np.inf, f_bin[msk].max() * 3, 3.0, 1.2],
        )
    popt, _ = optimize.curve_fit(
        fun, f_bin[msk], y[msk], p0=p0, bounds=bounds, max_nfev=20000
    )
    yhat = fun(f_bin[msk], *popt)
    r2 = 1 - np.sum((y[msk] - yhat) ** 2) / np.sum((y[msk] - y[msk].mean()) ** 2)
    out = {
        "A0": popt[0],
        "fx": popt[1],
        "mu": popt[2],
        "m": 0.75 if fix_m else popt[3],
        "r2": r2,
        "fmin": fmin,
        "fmax": fmax,
    }
    return out


# ---------------------------------------------------------------------------
# Transfer functions (cospectral, i.e. amplitude form; Moore 1986)
# ---------------------------------------------------------------------------
def tf_block_average(n, T):
    """High-pass from block averaging over period T (Massman & Clement Eq. 4.1)."""
    return 1.0 - np.sinc(np.asarray(n) * T) ** 2  # np.sinc(x) = sin(pi x)/(pi x)


def tf_running_mean(n, tau):
    """Recursive digital running mean, Moore (1986) / Moncrieff App. A (Tr)."""
    x = 2 * np.pi * np.asarray(n) * tau
    return x / np.sqrt(1 + x**2)


def tf_sensor_response(n, tau):
    """First-order dynamic response, Td(n)."""
    return 1.0 / np.sqrt(1 + (2 * np.pi * np.asarray(n) * tau) ** 2)


def tf_response_mismatch(n, t1, t2):
    """Sensor response mismatch, Tm(n)."""
    n = np.asarray(n)
    num = 1 + (2 * np.pi * n) ** 2 * t1 * t2
    den = np.sqrt((1 + (2 * np.pi * n * t1) ** 2) * (1 + (2 * np.pi * n * t2) ** 2))
    return num / den


def tf_sonic_path(n, p, u):
    """Sonic path averaging of w, Tw(fp), fp = n p / u (Moore 1986)."""
    fp = np.asarray(n, float) * p / u
    x = 2 * np.pi * fp
    with np.errstate(divide="ignore", invalid="ignore"):
        tw = (2 / (np.pi * fp)) * (1 + np.exp(-x) / 2 - 3 * (1 - np.exp(-x)) / (2 * x))
    return np.where(fp < 1e-4, 1.0, tw)


def tf_scalar_path(n, p, u):
    """Scalar line averaging along an open path of length p (Moore 1986): sinc(np/u)."""
    return np.abs(np.sinc(np.asarray(n, float) * p / u))


def tf_lateral_separation(n, s, u):
    """Lateral sensor separation, Ts(fs) = exp(-9.9 fs^1.5), fs = n s / u."""
    return np.exp(-9.9 * (np.asarray(n, float) * s / u) ** 1.5)


def tf_tube_laminar(n, r, X, U, D):
    """Tube attenuation, laminar flow (Leuning & Moncrieff 1990), Tt(n)."""
    tt = X / U
    return np.exp(-(np.pi**2) * r**2 * np.asarray(n, float) ** 2 * tt / (6 * D))


def tf_tube_turbulent(n, r, X, U, Re):
    """Tube attenuation, turbulent flow (Lenschow & Raupach 1991)."""
    return np.exp(-160 * Re * r * np.asarray(n, float) ** 2 * X / U**2)


def transfer_function_components(
    n, sysc: ECSystem, u: float, include_lowpass: bool = True
) -> dict:
    """Component responses. Historical ``include_lowpass`` includes the
    low-frequency loss (high-pass) filter; False omits averaging only.
    """
    _positive("u", u)
    n = np.asarray(n, float)
    comps = {}
    if include_lowpass:
        if sysc.detrend == "running_mean":
            comps["running mean"] = tf_running_mean(n, sysc.running_mean_tau)
        else:
            comps["block average"] = tf_block_average(n, sysc.avg_period)
    comps["sonic path averaging"] = tf_sonic_path(n, sysc.sonic_path, u)
    if not sysc.closed_path and sysc.scalar_path > 0:
        comps["scalar path averaging"] = tf_scalar_path(n, sysc.scalar_path, u)
    if sysc.lateral_sep > 0:
        comps["sensor separation"] = tf_lateral_separation(n, sysc.lateral_sep, u)
    if sysc.tau_sonic > 0:
        comps["sonic response"] = tf_sensor_response(n, sysc.tau_sonic)
    if sysc.tau_scalar > 0:
        comps["scalar response"] = tf_sensor_response(n, sysc.tau_scalar)
    if sysc.tau_sonic > 0 and sysc.tau_scalar > 0:
        comps["response mismatch"] = tf_response_mismatch(
            n, sysc.tau_sonic, sysc.tau_scalar
        )
    if sysc.closed_path:
        D = D_MOL[sysc.scalar]
        if sysc.tube_laminar:
            comps["tube loss (laminar)"] = tf_tube_laminar(
                n, sysc.tube_radius, sysc.tube_length, sysc.tube_velocity, D
            )
        else:
            comps["tube loss (turbulent)"] = tf_tube_turbulent(
                n, sysc.tube_radius, sysc.tube_length, sysc.tube_velocity, sysc.reynolds
            )
    return comps


def system_transfer_function(n, sysc: ECSystem, u: float, include_lowpass: bool = True):
    """Combined H(n) = product of all component transfer functions (Moncrieff Eq. 11)."""
    H = np.ones_like(np.asarray(n, float))
    for v in transfer_function_components(n, sysc, u, include_lowpass).values():
        H = H * v
    return H


# ---------------------------------------------------------------------------
# Correction factors
# ---------------------------------------------------------------------------
def _freq_grid(fs, nmin=1e-4, nmax_factor=5.0, npts=4000):
    return np.logspace(np.log10(nmin), np.log10(fs * nmax_factor), npts)


def model_fCo(
    n,
    sysc: ECSystem,
    u: float,
    model: str = "massman",
    fx: float | None = None,
    eta_x: float | None = None,
    mu: float = 0.5,
    m: float = 0.75,
    kind: str = "scalar",
    zL: float = 0.0,
):
    """Frequency-weighted model cospectrum on grid n (unnormalised)."""
    _positive("u", u)
    if model not in ("massman", "kaimal"):
        raise ValueError("model must be massman or kaimal")
    n = np.asarray(n, float)
    if model == "kaimal":
        return kaimal_cospectrum(n * sysc.zd / u, kind=kind, zL=zL)
    if fx is None:
        if eta_x is None:
            raise ValueError("give fx (Hz) or eta_x = fx (z-d)/u")
        fx = eta_x * u / sysc.zd
    return massman_cospectrum(n, 1.0, fx, mu, m)


def correction_factor_integral(
    sysc: ECSystem, u: float, include_lowpass: bool = True, n=None, **model_kw
) -> float:
    """F = true / measured = int Co dn / int H Co dn   (Moncrieff Eq. 10, Massman Eq. 4.1).

    Flux loss fraction = 1 - 1/F. Integration is over ln n using n*Co(n).
    model_kw are passed to `model_fCo` (model='kaimal' or 'massman', fx/eta_x, mu, zL...).
    """
    n = _freq_grid(sysc.fs) if n is None else np.asarray(n)
    if (
        n.ndim != 1
        or n.size < 2
        or not np.all(np.isfinite(n))
        or np.any(n <= 0)
        or np.any(np.diff(n) <= 0)
    ):
        raise ValueError(
            "frequency grid must be finite, positive, and strictly increasing"
        )
    fco = model_fCo(n, sysc, u, **model_kw)
    H = system_transfer_function(n, sysc, u, include_lowpass)
    lnn = np.log(n)
    true = _trapz(fco, lnn)
    meas = _trapz(fco * H, lnn)
    if not np.isfinite(meas) or meas <= 0:
        raise ValueError("model has no measurable covariance on this grid")
    return float(true / meas)


def _half_power_tau(tf: Callable[[np.ndarray], np.ndarray]) -> float:
    """Equivalent first-order time constant of a low-pass transfer function:
    tau = 1/(2 pi f_c) with tf(f_c) = 1/sqrt(2). Returns 0 if the filter never
    drops that far within [1e-4, 1e4] Hz."""

    def g(lf):
        return float(tf(np.array([10.0**lf]))[0]) - 1 / np.sqrt(2)

    if g(-4) < 0 or g(4) > 0:
        return 0.0
    lf = optimize.brentq(g, -4, 4)
    return 1.0 / (2 * np.pi * 10.0**lf)


def equivalent_time_constants(sysc: ECSystem, u, method: str = "massman") -> dict:
    """tau_e (high-frequency) and tau_b (low-frequency) for Massman's analytical model.

    tau_e = sqrt(sum tau_i^2) over the low-pass components (Massman 2000).
    method='massman': closed-form equivalents from Massman (2000) Table 1 --
        sonic line averaging l_w/(2.8u), scalar path l_s/(4u), lateral
        separation l_lat/(1.1u), first-order sensors tau -- vectorised over u.
        The tube time constant (u-independent) is the half-power equivalent of
        the tube transfer function.
    method='halfpower': every component's tau found numerically as the
        half-power point of its transfer function in this module (scalar u only).
    tau_b = Tb/2.8 for block averaging (Massman 2000) or the running-mean
    time constant.
    """
    _positive("u", u)
    if method not in ("massman", "halfpower"):
        raise ValueError("method must be massman or halfpower")
    tau_b = (
        sysc.running_mean_tau
        if sysc.detrend == "running_mean"
        else sysc.avg_period / 2.8
    )
    u = np.asarray(u, float)
    taus = {}
    if method == "massman":
        taus["sonic path averaging"] = sysc.sonic_path / (2.8 * u)
        if not sysc.closed_path and sysc.scalar_path > 0:
            taus["scalar path averaging"] = sysc.scalar_path / (4.0 * u)
        if sysc.lateral_sep > 0:
            taus["sensor separation"] = sysc.lateral_sep / (1.1 * u)
        if sysc.tau_sonic > 0:
            taus["sonic response"] = np.full_like(u, sysc.tau_sonic)
        if sysc.tau_scalar > 0:
            taus["scalar response"] = np.full_like(u, sysc.tau_scalar)
        if sysc.closed_path:
            D = D_MOL[sysc.scalar]
            if sysc.tube_laminar:

                def tf(n):
                    return tf_tube_laminar(
                        n, sysc.tube_radius, sysc.tube_length, sysc.tube_velocity, D
                    )
            else:

                def tf(n):
                    return tf_tube_turbulent(
                        n,
                        sysc.tube_radius,
                        sysc.tube_length,
                        sysc.tube_velocity,
                        sysc.reynolds,
                    )

            taus["tube loss"] = np.full_like(u, _half_power_tau(tf))
    else:
        u = float(u)
        comps = transfer_function_components(
            np.array([1.0]), sysc, u, include_lowpass=False
        )
        for name in comps:

            def component_tf(n, name=name):
                return transfer_function_components(n, sysc, u, False)[name]

            taus[name] = _half_power_tau(component_tf)
    tau_e: Any = np.sqrt(np.sum([np.asarray(t) ** 2 for t in taus.values()], axis=0))
    if np.ndim(tau_e) == 0:
        tau_e = float(tau_e)
        taus = {k: float(v) for k, v in taus.items()}
    return {"tau_e": tau_e, "tau_b": tau_b, "components": taus}


def correction_factor_analytical(fx, tau_b, tau_e, alpha: float = 1.0):
    """Analytical correction (Massman & Clement Eq. 4.4).

    ``F = (1 + 1/(2*pi*fx*tau_b)**alpha) * (1 + (2*pi*fx*tau_e)**alpha)``.
    Scalar and broadcastable array inputs are supported.
    """
    fx, tau_b, tau_e = (
        np.asarray(fx, float),
        np.asarray(tau_b, float),
        np.asarray(tau_e, float),
    )
    _positive("fx", fx)
    _positive("tau_b", tau_b)
    _positive("tau_e", tau_e, allow_zero=True)
    b = 2 * np.pi * fx * tau_b
    p = 2 * np.pi * fx * tau_e
    return (1 + 1 / b**alpha) * (1 + p**alpha)


def correction_uncertainty(
    fx,
    tau_b,
    tau_e,
    dfx_rel: float = 0.4,
    dalpha: float = 0.2,
    dtau_rel: float = 0.0,
    C: float = 1.2,
):
    """Relative uncertainty dF/F in the correction factor, Massman & Clement Eq. 4.7.

    Defaults are the paper's recommendations: dfx/fx = 0.4 (GLEES upper bound;
    0.5 for Griffin forest), d_alpha = 0.2, C = 1.2. dtau_rel applies to
    closed-path systems (they used 0.25 of tau_1); 0 for open path.
    """
    fx = np.asarray(fx, float)
    fx, tau_b, tau_e = (
        np.asarray(fx, float),
        np.asarray(tau_b, float),
        np.asarray(tau_e, float),
    )
    _positive("fx", fx)
    _positive("tau_b", tau_b)
    _positive("tau_e", tau_e, allow_zero=True)
    b = 2 * np.pi * fx * tau_b
    p = 2 * np.pi * fx * tau_e
    t1 = ((1 + 1 / b) * p - (1 + p) / b) ** 2 * dfx_rel**2
    t2 = ((1 + 1 / b) * xlogy(p, p) - (1 + p) / b * np.log(b)) ** 2 * dalpha**2
    t3 = ((1 + 1 / b) * p) ** 2 * dtau_rel**2
    return C * np.sqrt(t1 + t2 + t3) / ((1 + p) * (1 + 1 / b))


def correct_flux_table(
    df: pd.DataFrame,
    sysc: ECSystem,
    eta_x: float,
    u_col: str = "u",
    cov_cols=("w_c_cov",),
    method: str = "analytical",
    dfx_rel: float = 0.4,
    dtau_rel: float = 0.0,
    mu: float = 0.5,
    max_F: float = 1.5,
) -> pd.DataFrame:
    """Apply period-by-period spectral corrections to a half-hourly table.

    eta_x : site-specific normalised peak frequency fx (z-d)/u (from fits of
            many cospectra, e.g. mean of `fit_cospectrum` results, or a
            ETA_X_REFERENCE value if you have nothing better).
    method: 'analytical' (fast, Eq. 4.4) or 'integral' (Eq. 4.1 evaluated at each distinct valid wind speed).
    Adds columns F, dF_rel, and <col>_corr for each covariance column.
    Periods with F > max_F are flagged (Massman & Lee 2002 caution).
    Note: correct covariances before WPL.
    """
    if method not in ("analytical", "integral"):
        raise ValueError("method must be analytical or integral")
    _positive("eta_x", eta_x)
    _positive("max_F", max_F)
    out = df.copy()
    u = out[u_col].to_numpy(float)
    valid = np.isfinite(u) & (u > 0)
    fx = eta_x * u[valid] / sysc.zd
    taus = equivalent_time_constants(sysc, u[valid])
    tau_b, tau_e = taus["tau_b"], taus["tau_e"]
    F = np.full(u.shape, np.nan)
    if method == "analytical":
        F[valid] = correction_factor_analytical(fx, tau_b, tau_e)
    else:
        speeds, inverse = np.unique(u[valid], return_inverse=True)
        factors = np.array(
            [correction_factor_integral(sysc, ui, eta_x=eta_x, mu=mu) for ui in speeds]
        )
        F[valid] = factors[inverse]
    uncertainty = np.full(u.shape, np.nan)
    uncertainty[valid] = correction_uncertainty(
        fx, tau_b, tau_e, dfx_rel=dfx_rel, dtau_rel=dtau_rel
    )

    out["F"] = F
    out["dF_rel"] = uncertainty
    out["F_flag"] = (F > max_F) | ~np.isfinite(F)
    for col in cov_cols:
        out[f"{col}_corr"] = out[col] * F
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_cospectrum(
    co: dict,
    fit: dict | None,
    sysc: ECSystem,
    u: float,
    ax=None,
    kaimal_kind: str = "scalar",
    zL: float = 0.0,
    label: str = "w'c'",
):
    """Moncrieff Fig. 5: measured, model, and model x transfer function cospectra."""
    ax = ax or _pyplot().subplots(figsize=(6.5, 4.5))[1]
    f = co["f_bin"]
    y = co["fCo_bin"]
    ax.loglog(f[y > 0], y[y > 0], "o", mfc="none", color="k", ms=4, label="measured")
    if (y <= 0).any():
        ax.loglog(
            f[y <= 0], -y[y <= 0], "x", color="0.5", ms=4, label="measured (neg.)"
        )
    n = _freq_grid(sysc.fs, nmin=f.min() / 2, nmax_factor=0.6, npts=600)
    H = system_transfer_function(n, sysc, u)
    fk = n * sysc.zd / u
    kai = kaimal_cospectrum(fk, kaimal_kind, zL)
    kai *= y.max() / kai.max()
    ax.loglog(n, kai, "-", color="tab:blue", lw=1, label="Kaimal (1972)")
    if fit is not None:
        mod = massman_cospectrum(n, fit["A0"], fit["fx"], fit["mu"], fit["m"])
        ax.loglog(
            n,
            mod,
            "-",
            color="tab:red",
            lw=1.5,
            label=f"Massman fit: fx={fit['fx']:.3g} Hz, mu={fit['mu']:.2f}",
        )
        ax.loglog(
            n, mod * H, "--", color="tab:red", lw=1.5, label="fit x transfer functions"
        )
        ax.axvspan(fit["fmin"], fit["fmax"], color="tab:red", alpha=0.06, lw=0)
    else:
        ax.loglog(
            n,
            kai * H,
            "--",
            color="tab:blue",
            lw=1,
            label="Kaimal x transfer functions",
        )
    # -4/3 reference line for f*Co
    f0 = n[-1] / 8
    ax.loglog(
        [f0, n[-1]],
        [y.max() * 0.5, y.max() * 0.5 * (n[-1] / f0) ** (-4 / 3)],
        color="k",
        lw=0.8,
    )
    ax.text(n[-1] / 3, y.max() * 0.8, "-4/3", fontsize=8)
    ax.set_ylim(max(y[y > 0].min() / 3, 1e-5), y.max() * 3)
    ax.set_xlabel("natural frequency n (Hz)")
    ax.set_ylabel(f"n Co{label}(n) / cov")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    return ax


def plot_transfer_functions(sysc: ECSystem, u: float, ax=None):
    """Moncrieff Fig. 10: individual and combined transfer functions."""
    ax = ax or _pyplot().subplots(figsize=(6.5, 4.5))[1]
    n = _freq_grid(sysc.fs, nmin=1e-3, nmax_factor=0.5, npts=800)
    comps = transfer_function_components(n, sysc, u)
    for name, v in comps.items():
        ax.semilogx(n, v, lw=1, label=name)
    ax.semilogx(n, system_transfer_function(n, sysc, u), "k", lw=2.5, label="combined")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("natural frequency (Hz)")
    ax.set_ylabel("transfer function")
    ax.set_title(f"u = {u:.1f} m/s, z-d = {sysc.zd:.1f} m", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    return ax


def plot_flux_loss(sysc: ECSystem, u_range=None, zd_list=None, ax=None, **model_kw):
    """Moncrieff Fig. 6: % flux loss vs wind speed for several z-d."""
    ax = ax or _pyplot().subplots(figsize=(6, 4.5))[1]
    u_range = np.linspace(0.5, 8, 16) if u_range is None else u_range
    zd_list = [sysc.zd] if zd_list is None else zd_list
    kw = {"model": "kaimal"} | model_kw
    for zd in zd_list:
        s = ECSystem(**{**sysc.__dict__, "z_ref": zd, "d": 0.0})
        loss = [
            100 * (1 - 1 / correction_factor_integral(s, ui, **kw)) for ui in u_range
        ]
        ax.plot(u_range, loss, "o-", ms=3, label=f"z-d = {zd:g} m")
    ax.set_xlabel("wind speed (m/s)")
    ax.set_ylabel("% flux loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_ogive(co: dict, ax=None):
    """Massman Fig. 4.9: normalised cumulative flux vs frequency."""
    ax = ax or _pyplot().subplots(figsize=(6, 4))[1]
    ax.semilogx(co["f"], co["ogive"], "k", lw=1)
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.axvline(1 / co["N"] * co["fs"], color="0.5", ls=":", lw=0.8)
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("fraction of cumulative flux (ogive)")
    ax.set_ylim(-0.5, 1.5)
    ax.grid(alpha=0.3, which="both")
    return ax


def plot_lag(lags_s, r, ax=None):
    """Moncrieff Fig. 2: correlation coefficient vs lag."""
    ax = ax or _pyplot().subplots(figsize=(6, 4))[1]
    ax.plot(lags_s, r, "k", lw=1)
    k = np.nanargmax(np.abs(r))
    ax.axvline(
        lags_s[k], color="tab:red", ls="--", lw=1, label=f"lag = {lags_s[k]:.3f} s"
    )
    ax.set_xlabel("lag (s), + = scalar lags w")
    ax.set_ylabel("r_wc")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_correction_vs_wind(
    sysc: ECSystem,
    eta_x: float,
    u_range=None,
    ax=None,
    dfx_rel: float = 0.4,
    dtau_rel: float = 0.0,
    mu: float = 0.5,
):
    """Massman Fig. 4.6 / 4.8: F vs wind speed with the Eq. 4.7 uncertainty band."""
    ax = ax or _pyplot().subplots(figsize=(6, 4.5))[1]
    u_range = np.logspace(-1, np.log10(20), 40) if u_range is None else u_range
    F_an: Any = []
    dF: Any = []
    F_int: Any = []
    for ui in u_range:
        fx = eta_x * ui / sysc.zd
        t = equivalent_time_constants(sysc, ui)
        F_an.append(correction_factor_analytical(fx, t["tau_b"], t["tau_e"]))
        dF.append(
            correction_uncertainty(
                fx, t["tau_b"], t["tau_e"], dfx_rel, dtau_rel=dtau_rel
            )
        )
        F_int.append(correction_factor_integral(sysc, ui, eta_x=eta_x, mu=mu))
    F_an = np.array(F_an)
    dF = np.array(dF)
    F_int = np.array(F_int)
    ax.fill_between(
        u_range, F_an * (1 - dF), F_an * (1 + dF), color="0.8", label="±dF (Eq. 4.7)"
    )
    ax.semilogx(u_range, F_an, "k", lw=2, label="analytical (Eq. 4.4)")
    ax.semilogx(
        u_range, F_int, "--", color="tab:red", lw=1.2, label="integral (Eq. 4.1)"
    )
    ax.set_xlabel("wind speed (m/s)")
    ax.set_ylabel("correction factor F")
    ax.set_title(f"eta'_x = {eta_x:g}, z-d = {sysc.zd:.1f} m", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    return ax


# ---------------------------------------------------------------------------
# Synthetic data (for testing) and demo
# ---------------------------------------------------------------------------
def synthetic_turbulence(
    fs=20.0, T=1800.0, u=3.0, zd=3.0, seed=0, lag_s=0.25, tau_atten=0.15, noise=0.6
):
    """Synthetic w and scalar with a Kaimal-shaped cospectrum, a scalar lag, and
    first-order attenuation of the scalar (to exercise the diagnostics)."""
    rng = np.random.default_rng(seed)
    N = int(fs * T)
    f = np.fft.rfftfreq(N, 1 / fs)
    fk = f * zd / u
    shape = np.zeros_like(f)
    shape[1:] = np.sqrt(kaimal_cospectrum(fk[1:]) / f[1:])
    phase = np.exp(2j * np.pi * rng.random(len(f)))
    s = np.fft.irfft(shape * phase, N)
    s /= s.std()
    w = s + noise * rng.standard_normal(N)
    c = -0.8 * s + noise * rng.standard_normal(N)
    # first-order low-pass on the scalar (sensor / tube attenuation)
    if tau_atten > 0:
        a = np.exp(-1 / (fs * tau_atten))
        from scipy.signal import lfilter

        c = lfilter([1 - a], [1, -a], c)
    k = int(round(lag_s * fs))
    c = np.roll(c, k)
    return w, c


def demo(outdir: str = ".", sysc: ECSystem | None = None):
    """Run the whole pipeline on synthetic data and write figures to outdir."""
    import os

    sysc = sysc or ECSystem(
        fs=20, avg_period=1800, z_ref=3.0, d=0.0, lateral_sep=0.1, tau_scalar=0.0
    )
    u = 3.0
    w, c = synthetic_turbulence(
        fs=sysc.fs, T=sysc.avg_period, u=u, zd=sysc.zd, lag_s=0.25, tau_atten=0.15
    )

    lag, lags, r = find_lag(w, c, sysc.fs)
    c_al = shift(c, lag)
    co = cospectrum(w, c_al, sysc.fs)
    fit = fit_cospectrum(co["f_bin"], co["fCo_bin"], sysc.fs, sysc.avg_period)
    slope = inertial_slope(co["f_bin"], co["fCo_bin"], 3 * fit["fx"], sysc.fs / 4)
    taus = equivalent_time_constants(sysc, u)
    F_int = correction_factor_integral(sysc, u, fx=fit["fx"], mu=fit["mu"])
    F_an = correction_factor_analytical(fit["fx"], taus["tau_b"], taus["tau_e"])
    dF = correction_uncertainty(fit["fx"], taus["tau_b"], taus["tau_e"])
    eta_x = fit["fx"] * sysc.zd / u

    print(f"lag             : {lag} samples ({lag / sysc.fs:.3f} s)")
    print(
        f"cov(w,c)        : {co['cov']:.4f}   (check: sum Co df = {np.sum(co['Co']) * sysc.fs / co['N']:.4f})"
    )
    print(
        f"fit             : fx={fit['fx']:.3f} Hz  eta'_x={eta_x:.3f}  mu={fit['mu']:.2f}  R2={fit['r2']:.3f}"
    )
    print(
        f"inertial slope  : {slope:.2f}  (expect -1.33 for f*Co; steeper = attenuation)"
    )
    print(
        f"tau_e, tau_b    : {taus['tau_e']:.4f} s, {taus['tau_b']:.1f} s  "
        + str({k: round(v, 4) for k, v in taus["components"].items()})
    )
    print(f"F (integral)    : {F_int:.3f}   flux loss {100 * (1 - 1 / F_int):.1f}%")
    print(f"F (analytical)  : {F_an:.3f} ± {100 * dF:.1f}%")

    fig, axs = _pyplot().subplots(2, 3, figsize=(17, 9))
    plot_lag(lags, r, axs[0, 0])
    plot_cospectrum(co, fit, sysc, u, axs[0, 1])
    plot_ogive(co, axs[0, 2])
    plot_transfer_functions(sysc, u, axs[1, 0])
    plot_flux_loss(sysc, zd_list=[2, 3, 5, 8], ax=axs[1, 1])
    plot_correction_vs_wind(sysc, eta_x=eta_x, ax=axs[1, 2])
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "ec_spectral_demo.png")
    fig.savefig(path, dpi=130)
    _pyplot().close(fig)
    print("wrote", path)
    return {
        "lag": lag,
        "co": co,
        "fit": fit,
        "taus": taus,
        "F_int": F_int,
        "F_an": F_an,
        "dF": dF,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Eddy covariance spectral diagnostics/corrections"
    )
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--closed-path", action="store_true")
    args = ap.parse_args()
    if args.demo:
        s = ECSystem(
            closed_path=args.closed_path,
            lateral_sep=0.1,
            tau_scalar=0.1 if args.closed_path else 0.0,
        )
        demo(args.outdir, s)
    else:
        ap.print_help()
