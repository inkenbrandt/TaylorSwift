"""
Benchmarks for the Phase 1 hot spots (see ROADMAP.md).

Each test both benchmarks and sanity-checks its target so a broken
optimization cannot post a fast-but-wrong number.

Usage::

    pytest benchmarks                          # run once
    pytest benchmarks --benchmark-save=before  # save a baseline
    pytest benchmarks --benchmark-compare      # compare to last save
"""

import numpy as np
import pytest

from TaylorSwift import covariance, despike
from TaylorSwift.core import process_file, process_interval

pytest.importorskip("pytest_benchmark")


# ---------------------------------------------------------------------------
# Roadmap 1.2 — process_file interval slicing
# ---------------------------------------------------------------------------
class TestProcessFile:
    def test_process_file_two_days(self, benchmark, multiday_df, site_config):
        """End-to-end file processing: slicing + full spectral pipeline."""
        results = benchmark.pedantic(
            process_file, args=(multiday_df, site_config), rounds=3, iterations=1
        )
        assert len(results) == 96
        assert all(np.isfinite(r.ustar) for r in results)

    def test_interval_slicing_only(self, benchmark, sparse_multiday_df, site_config):
        """Pure slicing cost: every interval is skipped as incomplete."""
        results = benchmark.pedantic(
            process_file, args=(sparse_multiday_df, site_config), rounds=3, iterations=1
        )
        assert results == []


# ---------------------------------------------------------------------------
# Roadmap 1.3 — lag-search covariances
# ---------------------------------------------------------------------------
class TestLagSearch:
    def test_max_covariance_single_pair(self, benchmark, interval_arrays):
        _, _, w, T, _, _ = interval_arrays
        result = benchmark(covariance.calc_max_covariance, w, T, 10)
        assert result and np.isfinite(result[0][1])

    def test_max_covariance_flux_block(self, benchmark, interval_arrays):
        """The 21 velocity-scalar pairs computed per interval in _compute_fluxes."""
        u, v, w, T, co2, h2o = interval_arrays
        velocities = {"Ux": u, "Uy": v, "Uz": w}
        variables = {
            "Ux": u,
            "Uy": v,
            "Uz": w,
            "Ts": T,
            "pV": h2o,
            "Q": co2,
            "Sd": co2,
        }
        out = benchmark(covariance.build_covariance_dict, velocities, variables, 10)
        assert len(out) == 21
        assert np.isfinite(out["Uz-Ts"])


# ---------------------------------------------------------------------------
# Roadmap 1.4 — spike_detection
# ---------------------------------------------------------------------------
class TestSpikeDetection:
    def test_spike_detection_30min(self, benchmark, interval_arrays):
        w = interval_arrays[2].copy()
        w[::1000] += 5.0
        mask = benchmark(despike.spike_detection, w, 100, 4.0)
        assert mask.dtype == bool
        assert mask[1000]  # injected spike found


# ---------------------------------------------------------------------------
# Baseline for the spectral core (roadmap 1.7 lands against this number)
# ---------------------------------------------------------------------------
class TestProcessInterval:
    def test_process_interval_30min(self, benchmark, interval_arrays, site_config):
        u, v, w, T, co2, h2o = interval_arrays
        res = benchmark(process_interval, u, v, w, T, co2, h2o, site_config)
        assert np.isfinite(res.ustar)
