"""
compat.py — Backward-compatibility shim for the legacy CalcFlux API.

Provides CalcFlux as a thin wrapper around the current FluxConfig dataclass,
preserving the original constructor signature and utility methods.
"""

import numpy as np

from .config import FluxConfig


class CalcFlux:
    """
    Compatibility wrapper for the legacy CalcFlux processing class.

    Exposes the old attribute-based API while delegating to the current
    FluxConfig dataclass.  Keyword arguments are forwarded directly to
    FluxConfig, so any FluxConfig field can be overridden at construction.

    Examples
    --------
    >>> cf = CalcFlux(UHeight=5.0, meter_type="KH20")
    >>> cf.config.UHeight
    5.0
    >>> cf.convert_CtoK(0.0)
    273.15
    """

    def __init__(self, **kwargs):
        self.config = FluxConfig(**kwargs)
        self.covar: dict = {}
        self.avgvals: dict = {}

    # ------------------------------------------------------------------
    # Temperature conversions
    # ------------------------------------------------------------------

    def convert_KtoC(self, T):
        """Convert Kelvin to Celsius."""
        return np.asarray(T, dtype=float) - 273.15 if not np.isscalar(T) else float(T) - 273.15

    def convert_CtoK(self, T):
        """Convert Celsius to Kelvin."""
        return np.asarray(T, dtype=float) + 273.15 if not np.isscalar(T) else float(T) + 273.15

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    def calc_cov(self, x, y) -> float:
        """Covariance of x and y with ddof=1."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return float(np.cov(x, y)[0, 1])

    def calc_MSE(self, x) -> float:
        """Mean squared deviation from the mean (variance with ddof=0)."""
        x = np.asarray(x, dtype=float)
        return float(np.mean((x - np.mean(x)) ** 2))

    # ------------------------------------------------------------------
    # Thermodynamic helpers
    # ------------------------------------------------------------------

    def calc_Es(self, T):
        """
        Saturation vapour pressure [Pa] via the Magnus–Tetens formula.

        Parameters
        ----------
        T : float or array-like
            Temperature [K].
        """
        T_c = np.asarray(T, dtype=float) - 273.15
        es = 611.2 * np.exp(17.67 * T_c / (T_c + 243.5))
        return float(es) if np.isscalar(T) else es

    def tetens(self, T_C):
        """
        Saturation vapour pressure [hPa] using the Tetens formula.

        Parameters
        ----------
        T_C : array-like
            Temperature [°C].
        """
        T_C = np.asarray(T_C, dtype=float)
        return 6.1078 * 10.0 ** (7.5 * T_C / (237.3 + T_C))
