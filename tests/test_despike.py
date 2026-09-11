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

from TaylorSwift.despike import (
    despike_dataframe,
    polars_ukde_despike,
    rolling_sigma_filter,
    spike_detection,
)


class TestPolarsUKDE:
    @pytest.mark.parametrize("pandas_input", [False, True])
    def test_default_preserves_frame_and_removes_spike(self, pandas_input):
        import pandas as pd

        values = np.random.default_rng(42).normal(size=2000)
        values[500] = 100
        raw = pl.DataFrame({"x": values, "x_cleaned": ["keep"] * len(values)})
        expected = polars_ukde_despike(raw.select("x"), "x")["x_cleaned"].to_numpy()
        if pandas_input:
            raw = raw.to_pandas()
            raw.index = pd.date_range("2024-01-01", periods=len(raw), freq="s")
        result = despike_dataframe(raw, ["x", "absent"])
        np.testing.assert_allclose(result["x"].to_numpy(), expected, equal_nan=True)
        assert abs(result["x"].to_numpy()[500]) < 5
        assert list(result.columns) == list(raw.columns)
        assert raw["x"].to_numpy()[500] == 100
        assert result["x_cleaned"].to_list() == ["keep"] * len(values)
        if pandas_input:
            pd.testing.assert_index_equal(raw.index, result.index)

    def test_bulk_range_can_retain_a_rare_population(self):
        values = np.random.default_rng(24).normal(size=2000)
        values[900:920] = 12
        raw = pl.DataFrame({"x": values})
        narrow = despike_dataframe(raw, ["x"], prob_threshold=1e-6)
        wide = despike_dataframe(raw, ["x"], prob_threshold=1e-6, bulk_iqr=None)
        assert np.all(narrow["x"].to_numpy()[900:920] != 12)
        np.testing.assert_array_equal(wide["x"].to_numpy()[900:920], 12)

    def test_missing_and_infinite_values_and_boundaries(self):
        raw = pl.DataFrame({"x": [None, 1., np.nan, 3., np.inf, 5., None]})
        result = despike_dataframe(raw, ["x"], verbose=True)
        assert result["x"].to_list() == [None, 1., 2., 3., 4., 5., None]

    @pytest.mark.parametrize("values", [[], [None] * 5, [2] * 8, [1, 2, 3]])
    def test_degenerate_input(self, values):
        raw = pl.DataFrame({"x": pl.Series(values, dtype=pl.Float64)})
        assert despike_dataframe(raw, ["x"]).equals(raw)

    def test_zero_passes_is_noop(self):
        raw = pl.DataFrame({"x": [1., np.nan, 100., None]})
        assert despike_dataframe(raw, ["x"], max_iter=0).equals(raw)

    def test_multiple_passes_match_explicit_repetition(self):
        raw = pl.DataFrame({"x": np.random.default_rng(7).normal(size=2000)})
        once = despike_dataframe(raw, ["x"], prob_threshold=0.05)
        twice = despike_dataframe(once, ["x"], prob_threshold=0.05)
        result = despike_dataframe(raw, ["x"], prob_threshold=0.05, max_iter=2)
        assert result.equals(twice)

    @pytest.mark.parametrize("options", [
        {"prob_threshold": 0}, {"prob_threshold": 1}, {"prob_threshold": np.nan},
        {"max_iter": -1}, {"max_iter": 1.5}, {"bulk_iqr": 0}, {"bulk_iqr": np.inf},
    ])
    def test_invalid_options(self, options):
        with pytest.raises(ValueError):
            despike_dataframe(pl.DataFrame({"x": [1, 2, 3]}), ["x"], **options)


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
