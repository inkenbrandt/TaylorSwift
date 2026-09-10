"""
transfer_functions.py — Spectral transfer functions and model cospectra.

Each transfer function T(n) gives the fraction of the true (co)spectral
density that survives a given instrument or processing effect at natural
frequency n [Hz].  They are combined multiplicatively (Massman 2000) and
integrated against a model cospectrum to obtain flux correction factors
(see :mod:`TaylorSwift.corrections`).

HIGH-FREQUENCY LOSSES (attenuate flux at high frequencies):
  * Sensor frequency response — first-order time constant (Moore 1986)
  * Path averaging / line averaging along sonic paths (Kaimal et al. 1968)
  * Scalar path averaging for the IRGA optical path (Moore 1986)
  * Sensor separation — displacement between sonic and gas analyser
    (Moore 1986).  For IRGASON this is ~0 (integrated sensor).

LOW-FREQUENCY LOSSES (attenuate flux at low frequencies):
  * Block-average (finite averaging window) transfer function
  * Linear detrend transfer function

References
----------
Moore, C.J. (1986). Frequency response corrections for eddy correlation
    systems. Boundary-Layer Meteorol., 37, 17–35.
Massman, W.J. (2000). A simple method for estimating frequency response
    corrections for eddy covariance systems. Agric. For. Meteorol., 104,
    185–198.
Massman, W.J. (2001). Reply to comment by Rannik on "A simple method..."
    Agric. For. Meteorol., 107, 247–251.
Kaimal, J.C. et al. (1968). Deriving power spectra from a three-component
    sonic anemometer. J. Appl. Meteorol., 7, 827–837.
Kaimal, J.C. et al. (1972). Spectral characteristics of surface-layer
    turbulence. Quart. J. Roy. Meteor. Soc., 98, 563–589.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import SiteConfig

# np.trapz was renamed to np.trapezoid in NumPy 2.0 (and removed); support both.
# Re-exported for use by corrections.py and plotting.py.
try:
    from numpy import trapezoid as _trapezoid  # noqa: F401
except ImportError:  # NumPy < 2.0
    from numpy import trapz as _trapezoid  # type: ignore[no-redef]  # noqa: F401

__all__ = [
    "tf_block_average",
    "tf_linear_detrend",
    "tf_first_order_response",
    "tf_sonic_line_averaging",
    "tf_scalar_path_averaging",
    "tf_sensor_separation",
    "combined_transfer_function",
    "kaimal_cospec_model",
    "massman_alpha_x",
    "massman_spectral_factor",
]


# ===================================================================
# Individual transfer functions
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
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(x > 1e-10, (1.0 - np.sin(x) / x) ** 2, 0.0)
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
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = np.where(x > 1e-10, np.sin(x) / x, 1.0)
        # Moncrieff et al. (2004) robust form
        tf = 1.0 - sinc**2
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

    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(half_k1l > 1e-10, (np.sin(half_k1l) / half_k1l) ** 2, 1.0)
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
    return np.exp(-9.9 * nd_U**1.5)


# ===================================================================
# Combined transfer function (Massman 2000 approach)
# ===================================================================
_SCALAR_FLUXES = frozenset({"wT", "wCO2", "wH2O"})

_KAIMAL_PARAMS: dict[str, tuple[float, float]] = {
    "wT": (12.92, 26.7),
    "wCO2": (12.92, 26.7),
    "wH2O": (12.92, 26.7),
    "wu": (9.6, 14.0),
}


def _sensor_tau_for_flux(flux_type: str, instrument: SiteConfig) -> float:
    if flux_type == "wT":
        return instrument.tau_T
    if flux_type == "wCO2":
        return instrument.tau_co2
    if flux_type == "wH2O":
        return instrument.tau_h2o
    return 0.0


def _validate_flux_type(flux_type: str) -> None:
    if flux_type not in _KAIMAL_PARAMS:
        raise ValueError(f"Unknown flux_type: {flux_type}")


def massman_alpha_x(UHeight: float, L: float) -> tuple[float, float]:
    """
    Massman (2000) α and X stability parameters.

    Parameters
    ----------
    UHeight : float
        Measurement height [m].
    L : float
        Monin-Obukhov length [m].

    Returns
    -------
    tuple[float, float]
        (α, X) — broadness parameter and cospectral-peak factor.
    """
    if not np.isfinite(L) or (UHeight / L) <= 0:
        return 0.925, 0.085
    return 1.0, 2.0 - 1.915 / (1.0 + 0.5 * UHeight / L)


def massman_spectral_factor(B: float, alpha: float, V: float) -> float:
    """
    Massman (2000, 2001) analytic spectral attenuation factor.

    Parameters
    ----------
    B : float
        Block-averaging parameter 2π f_x τ_b.
    alpha : float
        Broadness parameter from :func:`massman_alpha_x`.
    V : float
        Sensor parameter 2π f_x τ_e.

    Returns
    -------
    float
        Measured-to-true flux ratio (0 to 1); divide the measured flux
        by this value to correct it.
    """
    B_a = B**alpha
    V_a = V**alpha
    return (B_a / (B_a + 1.0)) * (B_a / (B_a + V_a)) * (1.0 / (V_a + 1.0))


# Backward-compatible private aliases (deprecated; use the public names).
_calc_alph_x = massman_alpha_x
_correct_spectral = massman_spectral_factor


def combined_transfer_function(
    freq: np.ndarray,
    u_mean: float,
    instrument: SiteConfig,
    averaging_period: float = 30.0,
    flux_type: str = "wT",
    *,
    apply_low_freq: bool = True,
    apply_high_freq: bool = True,
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
    instrument : SiteConfig
        Instrument parameters.
    averaging_period : float
        Averaging period [minutes].
    flux_type : str
        Which flux: 'wT', 'wu', 'wCO2', 'wH2O'.
    apply_low_freq, apply_high_freq : bool
        Independently include block averaging and instrument response.
        Linear detrending is not included. Both False returns unity.

    Returns
    -------
    np.ndarray
        Combined transfer function T(n) at each frequency.
    """
    _validate_flux_type(flux_type)

    # 1. Low-frequency: block averaging
    T_low = (tf_block_average(freq, averaging_period)
             if apply_low_freq else np.ones_like(freq))
    if not apply_high_freq:
        return np.clip(T_low, 1e-10, 1.0)

    # 2. Sonic line-averaging (affects w always, and u for momentum flux)
    T_sonic_w = tf_sonic_line_averaging(freq, u_mean, instrument.sonic_path_length)

    # 3. High-frequency: sensor-specific
    if flux_type == "wT":
        T_sensor = tf_first_order_response(
            freq, _sensor_tau_for_flux(flux_type, instrument)
        )
        T_path_scalar = np.ones_like(freq)  # sonic T has same path as w
        T_sep = np.ones_like(freq)

    elif flux_type == "wu":
        # Momentum: both components are from the sonic
        T_sensor = np.ones_like(freq)  # no extra time constant
        T_path_scalar = tf_sonic_line_averaging(
            freq, u_mean, instrument.sonic_path_length
        )
        T_sep = np.ones_like(freq)

    else:  # wCO2 / wH2O — guaranteed by _validate_flux_type
        T_sensor = tf_first_order_response(
            freq, _sensor_tau_for_flux(flux_type, instrument)
        )
        T_path_scalar = tf_scalar_path_averaging(
            freq, u_mean, instrument.irga_path_length
        )
        T_sep = tf_sensor_separation(freq, u_mean, instrument.sensor_separation_total)

    # Combined: product of all transfer functions
    T_combined = T_low * T_sonic_w * T_sensor * T_path_scalar * T_sep

    return np.clip(T_combined, 1e-10, 1.0)


# ===================================================================
# Model cospectrum
# ===================================================================


def kaimal_cospec_model(f_nd: np.ndarray, flux_type: str = "wT") -> np.ndarray:
    """
    Kaimal et al. (1972) / Massman (2000) model cospectrum.

    Returns n Co(n) / cov(w'x') as a function of dimensionless
    frequency f = nz/U.  Used as the "true" cospectral shape for
    computing correction factors.
    """
    if flux_type not in _KAIMAL_PARAMS:
        flux_type = "wT"
    a, b = _KAIMAL_PARAMS[flux_type]
    return a * f_nd / (1.0 + b * f_nd) ** (7.0 / 4.0)
