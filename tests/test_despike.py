"""
Tests for TaylorSwift.despike.spike_detection.

The rolling z-score spike detector was vectorised in Phase 1 of the roadmap;
these tests pin the new implementation to a reference copy of the original
per-sample loop, expecting *identical* boolean masks.
"""

import numpy as np
import pytest

from TaylorSwift.despike import spike_detection


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
