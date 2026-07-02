"""
Shared fixtures for the TaylorSwift benchmark suite.

Two synthetic datasets mirror the Phase 1 roadmap targets:

* a single 30-minute interval at 20 Hz (36 000 samples) — used to benchmark
  per-interval work (lag-search covariances, despiking, spectral processing);
* a multi-day 20 Hz file — used to benchmark whole-file interval slicing in
  ``process_file``.

Run with::

    pytest benchmarks --benchmark-save=<label>

The directory is intentionally outside ``testpaths`` so the regular test
suite stays fast; benchmarks only run when requested explicitly.
"""

import numpy as np
import polars as pl
import pytest

from TaylorSwift.config import SiteConfig

FS = 20.0  # Hz
N_30MIN = int(30 * 60 * FS)  # 36 000 samples


def _synthetic_signals(rng: np.random.Generator, n: int):
    """Mean flow + correlated turbulent fluctuations (as in tests/conftest)."""
    u = 5.0 + rng.normal(0.0, 0.5, n)
    v = 0.5 + rng.normal(0.0, 0.3, n)
    w = rng.normal(0.0, 0.15, n)
    T = 20.0 + 0.3 * w + rng.normal(0.0, 0.1, n)
    co2 = 700.0 + rng.normal(0.0, 5.0, n)
    h2o = 10.0 + rng.normal(0.0, 0.5, n)
    return u, v, w, T, co2, h2o


def _make_file_frame(hours: float, fs: float = FS, drop_frac: float = 0.0, seed: int = 7):
    """Build a synthetic TOA5-like Polars frame spanning *hours* at *fs* Hz."""
    n = int(hours * 3600 * fs)
    rng = np.random.default_rng(seed)
    u, v, w, T, co2, h2o = _synthetic_signals(rng, n)
    step_us = int(1_000_000 / fs)
    ts = np.datetime64("2024-06-01T00:00:00", "us") + (
        np.arange(n, dtype=np.int64) * step_us
    ).astype("timedelta64[us]")
    df = pl.DataFrame(
        {
            "TIMESTAMP": ts,
            "Ux": u,
            "Uy": v,
            "Uz": w,
            "T_SONIC": T,
            "CO2_density": co2,
            "H2O_density": h2o,
        }
    )
    if drop_frac > 0.0:
        keep = rng.random(n) >= drop_frac
        df = df.filter(pl.Series(keep))
    return df


@pytest.fixture(scope="session")
def site_config():
    return SiteConfig(
        z_measurement=3.0,
        z_canopy=0.3,
        sampling_freq=FS,
        averaging_period=30.0,
    )


@pytest.fixture(scope="session")
def interval_arrays():
    """One 30-minute interval at 20 Hz (36 000 samples per signal)."""
    rng = np.random.default_rng(42)
    return _synthetic_signals(rng, N_30MIN)


@pytest.fixture(scope="session")
def multiday_df():
    """Two days of continuous 20 Hz data (~3.46 M rows, 96 full intervals)."""
    return _make_file_frame(hours=48.0)


@pytest.fixture(scope="session")
def sparse_multiday_df():
    """Two days at 20 Hz with 20% of rows dropped at random.

    Every interval fails the 90%-completeness test, so ``process_file``
    does only the interval-slicing work — isolating the per-interval
    DataFrame scan that Phase 1 targets.
    """
    return _make_file_frame(hours=48.0, drop_frac=0.20)
