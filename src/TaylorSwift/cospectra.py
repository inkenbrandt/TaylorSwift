"""
cospectra.py — FFT (co)spectral estimation and log-frequency binning.

Contains the pure spectral computations: :func:`compute_cospectrum`,
:func:`compute_spectrum`, and :func:`log_bin`.

The pieces that historically lived here have moved to dedicated modules
(re-exported below for backward compatibility):

* transfer functions and model cospectra → :mod:`TaylorSwift.transfer_functions`
* correction factors and :func:`apply_spectral_corrections`
  → :mod:`TaylorSwift.corrections`
* :class:`SpectralResult` → :mod:`TaylorSwift.results`
"""

from __future__ import annotations

import numpy as np

# --- Backward-compatible re-exports (previous home of these symbols) --------
from .corrections import (  # noqa: F401
    apply_spectral_corrections,
    compute_spectral_correction_factor,
    horst_analytical_correction,
)
from .results import SpectralResult  # noqa: F401
from .transfer_functions import (  # noqa: F401
    _KAIMAL_PARAMS,
    _SCALAR_FLUXES,
    _calc_alph_x,
    _correct_spectral,
    _sensor_tau_for_flux,
    _trapezoid,
    _validate_flux_type,
    combined_transfer_function,
    kaimal_cospec_model,
    massman_alpha_x,
    massman_spectral_factor,
    tf_block_average,
    tf_first_order_response,
    tf_linear_detrend,
    tf_scalar_path_averaging,
    tf_sensor_separation,
    tf_sonic_line_averaging,
)

__all__ = [
    "compute_cospectrum",
    "compute_spectrum",
    "log_bin",
    # Re-exports
    "SpectralResult",
    "apply_spectral_corrections",
    "compute_spectral_correction_factor",
    "horst_analytical_correction",
    "combined_transfer_function",
    "kaimal_cospec_model",
    "massman_alpha_x",
    "massman_spectral_factor",
    "tf_block_average",
    "tf_first_order_response",
    "tf_linear_detrend",
    "tf_scalar_path_averaging",
    "tf_sensor_separation",
    "tf_sonic_line_averaging",
]


def _scale_one_sided(spectrum: np.ndarray, n: int) -> None:
    """Scale an rFFT density in place along its last axis.

    Only even-length records have a Nyquist bin, which must remain undoubled.
    """
    spectrum[..., 1:-1 if n % 2 == 0 else None] *= 2.0


def compute_cospectrum(
    x: np.ndarray, y: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the one-sided cospectrum of two real signals.

    The cospectrum Co_xy(n) is the real part of the cross-spectral density.
    Uses a symmetric Hamming window and window-energy normalization, with DC
    omitted. For window ``h`` and ``S2 = sum(h**2)``, the exact discrete identity
    is ``sum(cospec) * fs/N = sum(h**2*x*y)/S2
    - sum(h*x)*sum(h*y)/(N*S2)``. This need not equal the unwindowed sample
    covariance for a finite record, even when the inputs are detrended.

    Parameters
    ----------
    x, y : np.ndarray
        Detrended time series of equal length.
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    freq : np.ndarray
        Frequency array [Hz] (positive only, excluding DC).
    cospec : np.ndarray
        One-sided cospectral density [units of x * y / Hz].
    """
    N = len(x)
    # Apply Hamming window to reduce spectral leakage
    window = np.hamming(N)
    # Window energy for density normalization
    S2 = np.sum(window**2)

    xw = x * window
    yw = y * window

    X = np.fft.rfft(xw)
    Y = np.fft.rfft(yw)

    # Cross-spectral density (two-sided -> one-sided)
    Sxy = X * np.conj(Y)

    # Normalise: divide by fs * S2 to get spectral density
    # (S2 corrects for the window energy)
    Sxy = Sxy / (fs * S2)

    _scale_one_sided(Sxy, N)

    # Cospectrum = real part
    cospec = np.real(Sxy)

    # Frequencies
    freq = np.fft.rfftfreq(N, d=1.0 / fs)

    # Drop DC component
    return freq[1:], cospec[1:]


def compute_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the one-sided power spectrum of a real signal.

    Parameters
    ----------
    x : np.ndarray
        Detrended time series.
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    freq : np.ndarray
        Frequency array [Hz] (positive only, excluding DC).
    psd : np.ndarray
        One-sided power spectral density [units²/Hz].

    Notes
    -----
    Uses the windowed identity in :func:`compute_cospectrum` with ``y = x``;
    its discrete integral need not equal the unwindowed sample variance.
    """
    freq, cospec = compute_cospectrum(x, x, fs)
    return freq, cospec


# ---------------------------------------------------------------------------
# Logarithmic frequency binning
# ---------------------------------------------------------------------------
def log_bin(
    freq: np.ndarray, spec: np.ndarray, bins_per_decade: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """
    Average spectral estimates into logarithmically spaced bins.

    This is the standard approach in the micromet community for producing
    smooth spectral curves (e.g., Kaimal et al. 1972; Moraes et al. 2008).

    Parameters
    ----------
    freq : np.ndarray
        Frequency array [Hz] (positive, no DC).
    spec : np.ndarray
        Spectral or cospectral density at each frequency.
    bins_per_decade : int
        Number of bins per frequency decade (default 20).

    Returns
    -------
    freq_bin : np.ndarray
        Bin-centre frequencies.
    spec_bin : np.ndarray
        Bin-averaged spectral density.
    """
    if len(freq) == 0:
        return np.array([]), np.array([])

    log_f = np.log10(freq)
    f_min, f_max = log_f.min(), log_f.max()
    n_bins = max(int((f_max - f_min) * bins_per_decade), 1)
    bin_edges = np.linspace(f_min, f_max, n_bins + 1)

    bin_centers = 10.0 ** (0.5 * (bin_edges[:-1] + bin_edges[1:]))

    # Assign each frequency to a 0-indexed bin; clip so the rightmost
    # value (= f_max = bin_edges[-1]) lands in the last bin rather than
    # spilling to n_bins.
    bin_idx = np.clip(np.digitize(log_f, bin_edges[1:]), 0, n_bins - 1)

    # Vectorised per-bin mean, handling NaN in spec without a Python loop
    valid = np.isfinite(spec)
    spec_safe = np.where(valid, spec, 0.0)
    bin_sum = np.bincount(bin_idx, weights=spec_safe, minlength=n_bins)
    bin_cnt = np.bincount(bin_idx, weights=valid.astype(float), minlength=n_bins)

    populated = bin_cnt > 0
    return bin_centers[populated], bin_sum[populated] / bin_cnt[populated]
