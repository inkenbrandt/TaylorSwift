"""
Shared pytest fixtures for eccospectra tests.
"""

import numpy as np
import pytest
from datetime import datetime, timedelta

from eccospectra import SiteConfig
from eccospectra.core import SpectralResult


# ---------------------------------------------------------------------------
# Random-number seed for reproducibility
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)

# Default sampling parameters used across tests
FS = 20.0          # Hz
N_30MIN = int(30 * 60 * FS)   # samples in a 30-min block


# ---------------------------------------------------------------------------
# Site configuration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def site_config():
    """Standard site configuration for a 3 m grassland tower."""
    return SiteConfig(
        z_measurement=3.0,
        z_canopy=0.3,
        sampling_freq=FS,
        averaging_period=30.0,
    )


@pytest.fixture
def site_config_tall():
    """Configuration for a taller (10 m) forest tower."""
    return SiteConfig(
        z_measurement=10.0,
        z_canopy=5.0,
        sampling_freq=FS,
        averaging_period=30.0,
    )


# ---------------------------------------------------------------------------
# Synthetic wind / scalar time series
# ---------------------------------------------------------------------------

@pytest.fixture
def short_white_noise():
    """Short (1000-point) white-noise arrays for unit tests."""
    n = 1000
    u = RNG.normal(3.0, 0.5, n)
    v = RNG.normal(0.0, 0.3, n)
    w = RNG.normal(0.0, 0.2, n)
    T = RNG.normal(20.0, 0.5, n)
    co2 = RNG.normal(700.0, 10.0, n)
    h2o = RNG.normal(10.0, 0.5, n)
    return u, v, w, T, co2, h2o


@pytest.fixture
def full_interval_arrays():
    """Full 30-min synthetic arrays with realistic turbulence structure."""
    n = N_30MIN
    t = np.arange(n) / FS

    # Mean flow + turbulent fluctuations
    u = 5.0 + RNG.normal(0.0, 0.5, n)
    v = 0.5 + RNG.normal(0.0, 0.3, n)
    w = RNG.normal(0.0, 0.15, n)

    # Temperature with slight positive w-T correlation (upward heat flux)
    T_fluc = 0.3 * w + RNG.normal(0.0, 0.1, n)
    T = 20.0 + T_fluc

    co2 = 700.0 + RNG.normal(0.0, 5.0, n)
    h2o = 10.0 + RNG.normal(0.0, 0.5, n)

    return u, v, w, T, co2, h2o


@pytest.fixture
def interval_with_nans(full_interval_arrays):
    """30-min arrays with a small fraction (<5%) of NaNs in wind."""
    u, v, w, T, co2, h2o = [a.copy() for a in full_interval_arrays]
    spike_idx = RNG.choice(len(u), size=50, replace=False)
    u[spike_idx] = np.nan
    return u, v, w, T, co2, h2o


@pytest.fixture
def spectral_result_stub():
    """Minimal SpectralResult with plausible freq / cospectrum arrays."""
    n_bins = 40
    freq_nd = np.logspace(-1, 1, n_bins)     # dimensionless f ∈ [0.1, 10]
    # Kaimal-like shape: peak around f ~ 0.1, declining inertial range
    cosp = 0.5 * freq_nd / (1 + 10 * freq_nd) ** (7 / 3)
    cosp += RNG.normal(0, 0.005, n_bins)

    res = SpectralResult(
        timestamp_start=datetime(2023, 6, 10, 0, 0),
        timestamp_end=datetime(2023, 6, 10, 0, 30),
        u_mean=5.0,
        ustar=0.3,
        L=-50.0,
        zL=-0.05,
        H=20.0,
        cov_wT=0.017,
        cov_wu=-0.09,
        cov_wCO2=-0.05,
        cov_wH2O=0.002,
        freq=freq_nd * 5.0 / 3.0,    # Hz (z=3, U=5 → f = freq_nd*U/z)
        freq_nd=freq_nd,
        ncosp_wT=cosp,
        ncosp_wu=cosp * 0.8,
        ncosp_wCO2=cosp * 0.6,
        ncosp_wH2O=cosp * 0.5,
        spec_u=cosp * 1.2,
        spec_v=cosp * 0.8,
        spec_w=cosp * 0.6,
        spec_T=cosp * 1.0,
    )
    return res
