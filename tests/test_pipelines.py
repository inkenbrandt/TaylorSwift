"""
Smoke test for pipelines._compute_fluxes.

The full run_irga / run_kh20 pipelines need optional despiking dependencies,
but the core flux computation can be exercised directly with canonical
columns. This pins the covariance-block rewiring done in Phase 1 (shared
FFT preparation via build_covariance_dict) to a working end-to-end result.
"""

import numpy as np
import pandas as pd
import pytest

from TaylorSwift import thermo
from TaylorSwift.config import FluxConfig
from TaylorSwift.pipelines import _OUTPUT_COLUMNS, _compute_fluxes


@pytest.fixture
def canonical_df():
    rng = np.random.default_rng(8)
    n = 6000
    Uz = rng.normal(0.0, 0.15, n)
    Ts = 293.15 + 0.3 * Uz + rng.normal(0.0, 0.1, n)  # upward heat flux
    pV = 0.010 + 2e-4 * Uz + rng.normal(0.0, 5e-4, n)  # kg m-3
    Pr = np.full(n, 86_000.0)  # Pa

    E = thermo.calc_E(pV, Ts)
    Q = thermo.calc_Q(Pr, E)
    Tsa = thermo.calc_Tsa(Ts, Q)
    Sd = thermo.calc_Q(Pr, thermo.calc_Es(Tsa)) - Q

    return pd.DataFrame(
        {
            "Ux": 2.0 + rng.normal(0.0, 0.5, n),
            "Uy": 0.5 + rng.normal(0.0, 0.3, n),
            "Uz": Uz,
            "Ts": Ts,
            "pV": pV,
            "Pr": Pr,
            "E": E,
            "Q": Q,
            "Tsa": Tsa,
            "Sd": Sd,
        }
    )


class TestComputeFluxes:
    def test_returns_all_outputs_finite(self, canonical_df):
        result = _compute_fluxes(canonical_df, FluxConfig())
        assert list(result.index) == _OUTPUT_COLUMNS
        assert np.isfinite(result.to_numpy(dtype=float)).all()

    def test_physical_plausibility(self, canonical_df):
        result = _compute_fluxes(canonical_df, FluxConfig())
        assert result["H"] > 0  # constructed with positive w'T'
        assert 0.0 <= result["direction"] <= 360.0
        assert result["Ustr"] >= 0.0
        assert -100.0 < result["Ta"] < 60.0  # deg C
