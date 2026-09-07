"""
Tests for TaylorSwift.despike.spike_detection.

The rolling z-score spike detector was vectorised in Phase 1 of the roadmap;
these tests pin the new implementation to a reference copy of the original
per-sample loop, expecting *identical* boolean masks.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from TaylorSwift.despike import rolling_sigma_filter, spike_detection


def _reference_spike_detection(data, window_size=100, z_threshold=4.0):
    """The pre-optimisation per-sample loop."""
    data = np.asarray(data)
    spikes = np.zeros_like(data, dtype=bool)
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2)
        window = data[start:end]
        if window.size == 0:
            continue
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            spikes[i] = abs(data[i] - mean) / std > z_threshold
    return spikes


class TestSpikeDetectionEquivalence:
    @pytest.mark.parametrize("window_size", [20, 51, 100])
    def test_random_data_with_spikes(self, window_size):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 2000)
        x[::250] += 12.0
        result = spike_detection(x, window_size=window_size, z_threshold=4.0)
        expected = _reference_spike_detection(
            x, window_size=window_size, z_threshold=4.0
        )
        assert np.array_equal(result, expected)
        assert result[250]  # injected spike found

    def test_large_mean_signal(self):
        rng = np.random.default_rng(1)
        x = 700.0 + rng.normal(0, 5.0, 3000)
        x[1500] += 60.0
        result = spike_detection(x, window_size=100, z_threshold=4.0)
        expected = _reference_spike_detection(x, window_size=100, z_threshold=4.0)
        assert np.array_equal(result, expected)
        assert result[1500]

    def test_with_nans(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, 1000)
        x[100:110] = np.nan
        x[500] += 15.0
        result = spike_detection(x, window_size=50, z_threshold=4.0)
        expected = _reference_spike_detection(x, window_size=50, z_threshold=4.0)
        assert np.array_equal(result, expected)
        # windows touching the NaN run can never be flagged
        assert not result[95:115].any()

    def test_constant_data_no_spikes(self):
        x = np.full(500, 42.0)
        assert not spike_detection(x, window_size=50).any()

    @pytest.mark.parametrize("n", [0, 1, 5, 30, 99, 100, 101])
    def test_short_arrays_and_window_edge_sizes(self, n):
        rng = np.random.default_rng(n)
        x = rng.normal(0, 1, n)
        if n > 2:
            x[n // 2] += 20.0
        result = spike_detection(x, window_size=100, z_threshold=4.0)
        expected = _reference_spike_detection(x, window_size=100, z_threshold=4.0)
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize("window_size", [1, 2, 3])
    def test_degenerate_windows(self, window_size):
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, 50)
        result = spike_detection(x, window_size=window_size, z_threshold=4.0)
        expected = _reference_spike_detection(
            x, window_size=window_size, z_threshold=4.0
        )
        assert np.array_equal(result, expected)

    def test_integer_input(self):
        x = np.zeros(300, dtype=int)
        x[150] = 500
        result = spike_detection(x, window_size=40, z_threshold=4.0)
        expected = _reference_spike_detection(x, window_size=40, z_threshold=4.0)
        assert np.array_equal(result, expected)

    def test_list_input(self):
        x = [0.0] * 100 + [50.0] + [0.0] * 100
        rng = np.random.default_rng(4)
        x = list(np.asarray(x) + rng.normal(0, 0.5, 201))
        result = spike_detection(x, window_size=30, z_threshold=4.0)
        expected = _reference_spike_detection(x, window_size=30, z_threshold=4.0)
        assert np.array_equal(result, expected)

    def test_output_shape_and_dtype(self):
        x = np.random.default_rng(5).normal(0, 1, 400)
        mask = spike_detection(x)
        assert mask.shape == x.shape
        assert mask.dtype == bool

    def test_nonpositive_window_is_rejected(self):
        with pytest.raises(ValueError):
            spike_detection([1.0, 2.0], window_size=0)

    def test_slow_ramp_is_not_flagged_as_isolated_spike(self):
        x = np.linspace(0.0, 10.0, 201)
        assert not spike_detection(x, window_size=31, z_threshold=4.0).any()


class TestRollingSigmaFilter:
    def test_sorts_time_and_drops_stats_when_requested(self):
        frame = pl.DataFrame(
            {
                "TIMESTAMP": [
                    datetime(2023, 1, 1, 0, 0, 2),
                    datetime(2023, 1, 1, 0, 0, 0),
                    datetime(2023, 1, 1, 0, 0, 1),
                ],
                # The spike rides on the *last* timestamp: a trailing window
                # placed earlier would still contain it and drag the mean far
                # enough to flag the baseline samples too.
                "Uz": [20.0, 0.0, 0.0],
            }
        )
        result = rolling_sigma_filter(
            frame, period="3s", sigma=0.5, keep_stats=False
        )
        assert result["TIMESTAMP"].is_sorted()
        assert "Uz_roll_mean" not in result.columns
        assert "Uz_roll_std" not in result.columns
        # Values follow the sort, so only the final spike is nulled.
        assert result["Uz"].to_list() == [0.0, 0.0, 20.0]
        assert result["Uz_filtered"].to_list() == [0.0, 0.0, None]
