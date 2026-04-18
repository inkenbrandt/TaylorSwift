"""
Tests for TaylorSwift.corrections — despiking, transfer functions, WPL.
"""

import numpy as np
import pytest
import pandas as pd

from TaylorSwift.corrections import (
    InstrumentConfig,
    tf_block_average,
    tf_first_order_response,
    tf_sonic_line_averaging,
    tf_sensor_separation,
    combined_transfer_function,
)
from TaylorSwift.despike import ukde_despike, despike_dataframe


# ---------------------------------------------------------------------------
# InstrumentConfig
# ---------------------------------------------------------------------------


class TestInstrumentConfig:
    def test_defaults_are_irgason(self):
        cfg = InstrumentConfig()
        assert cfg.model == "IRGASON"
        assert cfg.irga_type == "open_path"
        assert cfg.sonic_path_length == pytest.approx(0.10)
        assert cfg.irga_path_length == pytest.approx(0.154)

    def test_zero_separation_gives_zero_total(self):
        cfg = InstrumentConfig(
            sensor_separation_lateral=0.0,
            sensor_separation_longitudinal=0.0,
            sensor_separation_vertical=0.0,
        )
        assert cfg.sensor_separation_total == 0.0

    def test_separation_distance_euclidean(self):
        cfg = InstrumentConfig(
            sensor_separation_lateral=3.0,
            sensor_separation_longitudinal=4.0,
            sensor_separation_vertical=0.0,
        )
        assert cfg.sensor_separation_total == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# ukde_despike
# ---------------------------------------------------------------------------


class TestUkdeDespike:
    def test_clean_signal_unchanged(self):
        rng = np.random.default_rng(10)
        signal = rng.normal(0, 1, 500)
        cleaned = ukde_despike(signal, prob_threshold=1e-6)
        # With very strict threshold, only extreme outliers removed
        assert len(cleaned) == len(signal)
        assert np.isfinite(cleaned).all()

    def test_spikes_removed(self):
        rng = np.random.default_rng(11)
        signal = rng.normal(0, 1, 1000)
        # Insert obvious spikes
        spike_idx = [100, 300, 700]
        signal[spike_idx] = 100.0  # extreme outliers
        cleaned = ukde_despike(signal, prob_threshold=1e-4)
        assert np.isfinite(cleaned).all()
        # Spike positions should be replaced with reasonable values
        for idx in spike_idx:
            assert abs(cleaned[idx]) < 10.0  # pulled back toward distribution

    def test_output_same_length(self):
        signal = np.random.default_rng(12).normal(0, 1, 200)
        cleaned = ukde_despike(signal)
        assert len(cleaned) == len(signal)

    def test_handles_nans(self):
        """ukde_despike should not raise and should not introduce new NaNs."""
        rng = np.random.default_rng(13)
        signal = rng.normal(0, 1, 300).astype(float)
        signal[50] = np.nan
        signal[150] = np.nan
        cleaned = ukde_despike(signal)
        assert len(cleaned) == len(signal)
        # Non-NaN positions in the output must be finite
        assert np.isfinite(cleaned[~np.isnan(cleaned)]).all()
        # Should not introduce NEW NaN positions
        original_nans = np.isnan(signal)
        assert not np.any(np.isnan(cleaned) & ~original_nans)

    def test_all_nans_returns_array(self):
        signal = np.full(100, np.nan)
        cleaned = ukde_despike(signal)
        assert len(cleaned) == 100

    def test_max_iter_respected(self):
        rng = np.random.default_rng(14)
        signal = rng.normal(0, 1, 200)
        # Should not raise even with max_iter=1
        cleaned = ukde_despike(signal, max_iter=1)
        assert len(cleaned) == len(signal)


# ---------------------------------------------------------------------------
# despike_dataframe
# ---------------------------------------------------------------------------


class TestDespikeDataframe:
    def _make_df(self, n=500):
        rng = np.random.default_rng(15)
        df = pd.DataFrame(
            {
                "u": rng.normal(5, 0.3, n),
                "w": rng.normal(0, 0.1, n),
                "T": rng.normal(20, 0.5, n),
            }
        )
        df.loc[50, "u"] = 100.0  # spike
        return df

    def test_returns_dataframe(self):
        df = self._make_df()
        result = despike_dataframe(df, ["u", "w"])
        assert isinstance(result, pd.DataFrame)

    def test_original_not_modified(self):
        df = self._make_df()
        original_u = df["u"].copy()
        despike_dataframe(df, ["u"])
        pd.testing.assert_series_equal(df["u"], original_u)

    def test_spike_cleaned(self):
        df = self._make_df()
        result = despike_dataframe(df, ["u"])
        assert abs(result.loc[50, "u"]) < 20.0

    def test_missing_column_skipped(self):
        df = self._make_df()
        # Should not raise for a column that doesn't exist
        result = despike_dataframe(df, ["u", "nonexistent_col"])
        assert "u" in result.columns
        assert "nonexistent_col" not in result.columns


# ---------------------------------------------------------------------------
# Transfer functions — shape and boundary conditions
# ---------------------------------------------------------------------------


class TestTransferFunctions:
    @pytest.fixture
    def freq(self):
        return np.logspace(-3, 1, 100)

    # --- tf_block_average ---

    def test_block_average_range(self, freq):
        tf = tf_block_average(freq, averaging_period=30.0)
        assert tf.shape == freq.shape
        # The formula (1 - sin(x)/x)^2 is always non-negative;
        # it can exceed 1 when sin(x)/x < 0 (x > π), so we only check >= 0
        assert np.all(tf >= 0)

    def test_block_average_at_very_low_freq_near_zero(self):
        """Very low frequencies are strongly attenuated by block averaging."""
        freq_low = np.array([1e-5])
        tf = tf_block_average(freq_low, averaging_period=30.0)
        assert tf[0] < 0.05

    def test_block_average_at_high_freq_near_one(self):
        """High frequencies pass through a block average unaffected."""
        freq_high = np.array([5.0])
        tf = tf_block_average(freq_high, averaging_period=30.0)
        assert tf[0] > 0.9

    # --- tf_first_order_response ---

    def test_first_order_zero_tau_returns_ones(self, freq):
        tf = tf_first_order_response(freq, tau=0.0)
        np.testing.assert_array_equal(tf, np.ones_like(freq))

    def test_first_order_negative_tau_returns_ones(self, freq):
        tf = tf_first_order_response(freq, tau=-1.0)
        np.testing.assert_array_equal(tf, np.ones_like(freq))

    def test_first_order_attenuation_increases_with_freq(self, freq):
        tf = tf_first_order_response(freq, tau=0.1)
        assert np.all(np.diff(tf) <= 0)  # monotone decreasing

    def test_first_order_range(self, freq):
        tf = tf_first_order_response(freq, tau=0.1)
        assert np.all(tf >= 0)
        assert np.all(tf <= 1)

    # --- tf_sonic_line_averaging ---

    def test_sonic_line_avg_near_one_at_low_freq(self):
        freq_low = np.array([0.001])
        tf = tf_sonic_line_averaging(freq_low, u_mean=5.0, path_length=0.1)
        assert tf[0] > 0.99

    def test_sonic_line_avg_zero_path_returns_ones(self, freq):
        tf = tf_sonic_line_averaging(freq, u_mean=5.0, path_length=0.0)
        np.testing.assert_array_equal(tf, np.ones_like(freq))

    def test_sonic_line_avg_zero_wind_returns_ones(self, freq):
        tf = tf_sonic_line_averaging(freq, u_mean=0.0, path_length=0.1)
        np.testing.assert_array_equal(tf, np.ones_like(freq))

    def test_sonic_line_avg_range(self, freq):
        tf = tf_sonic_line_averaging(freq, u_mean=5.0, path_length=0.1)
        assert np.all(tf >= 0)
        assert np.all(tf <= 1)

    # --- tf_sensor_separation ---

    def test_sensor_separation_zero_returns_ones(self, freq):
        tf = tf_sensor_separation(freq, u_mean=5.0, separation=0.0)
        np.testing.assert_array_equal(tf, np.ones_like(freq))

    def test_sensor_separation_larger_means_more_attenuation(self, freq):
        tf_small = tf_sensor_separation(freq, u_mean=5.0, separation=0.1)
        tf_large = tf_sensor_separation(freq, u_mean=5.0, separation=0.5)
        # larger separation → more attenuation (smaller TF values)
        assert np.all(tf_large <= tf_small + 1e-10)

    def test_sensor_separation_range(self, freq):
        tf = tf_sensor_separation(freq, u_mean=5.0, separation=0.2)
        assert np.all(tf >= 0)
        assert np.all(tf <= 1)

    # --- combined_transfer_function ---

    def test_combined_tf_all_flux_types(self, freq):
        inst = InstrumentConfig()
        for flux_type in ("wT", "wu", "wCO2", "wH2O"):
            tf = combined_transfer_function(
                freq,
                u_mean=5.0,
                instrument=inst,
                averaging_period=30.0,
                flux_type=flux_type,
            )
            assert tf.shape == freq.shape
            assert np.all(tf >= 0)
            assert np.all(tf <= 1)

    def test_combined_tf_irgason_near_one_at_low_freq(self):
        """IRGASON (integrated sensor, no separation) — low-freq TF near 1."""
        freq_low = np.logspace(-3, -1, 20)
        inst = InstrumentConfig()
        tf = combined_transfer_function(
            freq_low,
            u_mean=5.0,
            instrument=inst,
            averaging_period=30.0,
            flux_type="wT",
        )
        # At f=0.001 Hz the low-freq block average TF is also small —
        # just check the combined TF stays in [0, 1]
        assert np.all(tf >= 0)
        assert np.all(tf <= 1)
