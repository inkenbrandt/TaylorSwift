"""
compat.py — Backward-compatibility shim for the legacy CalcFlux API.

Provides CalcFlux as a thin wrapper around the current FluxConfig dataclass,
preserving the original constructor signature and utility methods.  All
physics and statistics delegate to :mod:`TaylorSwift.thermo` and
:mod:`TaylorSwift.covariance` — nothing is re-implemented here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import covariance, thermo
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

    def __init__(self, **kwargs: Any) -> None:
        self.config = FluxConfig(**kwargs)
        self.covar: dict[str, float] = {}
        self.avgvals: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Temperature conversions — delegate to thermo
    # ------------------------------------------------------------------

    def convert_KtoC(self, T):
        """Convert Kelvin to Celsius."""
        if np.isscalar(T):
            return float(thermo.convert_KtoC(float(T)))
        return thermo.convert_KtoC(np.asarray(T, dtype=float))

    def convert_CtoK(self, T):
        """Convert Celsius to Kelvin."""
        if np.isscalar(T):
            return float(thermo.convert_CtoK(float(T)))
        return thermo.convert_CtoK(np.asarray(T, dtype=float))

    # ------------------------------------------------------------------
    # Statistical helpers — delegate to covariance
    # ------------------------------------------------------------------

    def calc_cov(self, x, y) -> float:
        """Covariance of x and y with ddof=1."""
        return covariance.calc_cov(x, y)

    def calc_MSE(self, x) -> float:
        """Mean squared deviation from the mean (variance with ddof=0)."""
        return covariance.calc_MSE(x)

    # ------------------------------------------------------------------
    # Thermodynamic helpers — delegate to thermo
    # ------------------------------------------------------------------

    def calc_Es(self, T):
        """
        Saturation vapour pressure [Pa] via the Tetens formula.

        Parameters
        ----------
        T : float or array-like
            Temperature [K].
        """
        es = thermo.calc_Es(np.asarray(T, dtype=float))
        return float(es) if np.isscalar(T) else es

    def tetens(self, T_C):
        """
        Saturation vapour pressure [hPa] using the Tetens formula.

        Parameters
        ----------
        T_C : array-like
            Temperature [°C].
        """
        # thermo.tetens returns kPa; the legacy API contract is hPa.
        return thermo.tetens(np.asarray(T_C, dtype=float)) * 10.0
