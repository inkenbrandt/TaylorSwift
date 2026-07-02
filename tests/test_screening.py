"""
Tests for TaylorSwift.screening — Vickers & Mahrt (1997) raw-data screening.

Covers each individual test (spike, amplitude resolution, dropout, absolute
limits, higher moments), the ``vickers_mahrt_screen`` orchestrator, and the
integration into ``process_interval`` / ``process_file``.
"""

import numpy as np
import pytest

from TaylorSwift.screening import (
    ScreeningConfig,
    absolute_limits_test,
    amplitude_resolution_test,
    dropout_test,
    higher_moment_test,
    spike_test,
    vickers_mahrt_screen,
)

RNG = np.random.default_rng(1234)


# ---------------------------------------------------------------------------
# spike_test
# ---------------------------------------------------------------------------
class TestSpikeTest:
    def test_clean_signal_few_spikes(self):
        x = RNG.normal(0, 1, 2000)
        n_spikes, frac, mask = spike_test(x)
        assert frac < 0.02
        assert mask.shape == x.shape
        assert mask.dtype == bool

    def test_injected_spike_detected(self):
        x = RNG.normal(0, 1, 1000)
        x[500] = 30.0
        cfg = ScreeningConfig(spike_window=101)
        n_spikes, frac, mask = spike_test(x, cfg)
        assert mask[500]
        assert n_spikes >= 1

    def test_real_ramp_not_flagged(self):
        # A long, smoothly varying excursion is real turbulence, not a spike:
        # a run longer than spike_max_consecutive must be kept.
        x = RNG.normal(0, 0.1, 1000)
        x[400:500] += 5.0  # 100-sample sustained excursion
        cfg = ScreeningConfig(spike_window=201, spike_max_consecutive=3)
        _, _, mask = spike_test(x, cfg)
        assert mask[400:500].sum() == 0

    def test_fraction_matches_mask(self):
        x = RNG.normal(0, 1, 1000)
        x[100] = 40.0
        x[900] = -40.0
        n_spikes, frac, mask = spike_test(x, ScreeningConfig(spike_window=101))
        assert n_spikes == int(mask.sum())
        assert frac == pytest.approx(n_spikes / np.isfinite(x).sum())

    def test_all_nan_returns_zero(self):
        x = np.full(100, np.nan)
        n_spikes, frac, mask = spike_test(x)
        assert n_spikes == 0
        assert frac == 0.0
        assert not mask.any()


# ---------------------------------------------------------------------------
# amplitude_resolution_test
# ---------------------------------------------------------------------------
class TestAmplitudeResolution:
    def test_well_resolved_signal_passes(self):
        x = RNG.normal(0, 1, 2000)
        flag, empty = amplitude_resolution_test(x)
        assert flag is False
        assert empty < 0.7

    def test_coarse_signal_flagged(self):
        # Rounding to integers leaves only a handful of distinct levels, so a
        # 100-bin histogram is mostly empty.
        x = np.round(RNG.normal(0, 1, 500))
        flag, empty = amplitude_resolution_test(x)
        assert flag is True
        assert empty > 0.7

    def test_too_short_returns_false(self):
        x = RNG.normal(0, 1, 20)
        flag, empty = amplitude_resolution_test(x)
        assert flag is False
        assert empty == 0.0

    def test_constant_window_worst_resolution(self):
        x = np.ones(500)
        flag, empty = amplitude_resolution_test(x)
        assert flag is True
        assert empty == 1.0


# ---------------------------------------------------------------------------
# dropout_test
# ---------------------------------------------------------------------------
class TestDropoutTest:
    def test_clean_signal_no_dropout(self):
        x = RNG.normal(0, 1, 1000)
        soft, extreme, frac = dropout_test(x)
        assert soft is False
        assert extreme is False
        assert frac < 0.1

    def test_interior_flatline_soft_flag(self):
        x = RNG.normal(0, 1, 1000)
        x[400:520] = 0.0  # 12% stuck at the distribution centre
        soft, extreme, frac = dropout_test(x)
        assert soft is True
        assert extreme is False
        assert frac == pytest.approx(0.12, abs=0.01)

    def test_extreme_flatline_hard_flag(self):
        x = RNG.normal(0, 1, 1000)
        x[100:230] = 8.0  # 13% stuck at the top of the distribution
        soft, extreme, frac = dropout_test(x)
        assert extreme is True

    def test_fully_flat_signal(self):
        x = np.full(500, 3.3)
        soft, extreme, frac = dropout_test(x)
        assert soft is True
        assert frac == 1.0


# ---------------------------------------------------------------------------
# absolute_limits_test
# ---------------------------------------------------------------------------
class TestAbsoluteLimits:
    def test_within_range(self):
        x = RNG.normal(0, 1, 500)
        flag, n = absolute_limits_test(x, (-30.0, 30.0))
        assert flag is False
        assert n == 0

    def test_out_of_range_counted(self):
        x = RNG.normal(0, 1, 500)
        x[10] = 100.0
        x[20] = -100.0
        flag, n = absolute_limits_test(x, (-30.0, 30.0))
        assert flag is True
        assert n == 2

    def test_none_limits_skips(self):
        x = np.array([1e9, -1e9, 0.0])
        flag, n = absolute_limits_test(x, None)
        assert flag is False
        assert n == 0

    def test_nan_not_counted(self):
        x = np.array([0.0, np.nan, 1.0])
        flag, n = absolute_limits_test(x, (-5.0, 5.0))
        assert flag is False
        assert n == 0


# ---------------------------------------------------------------------------
# higher_moment_test
# ---------------------------------------------------------------------------
class TestHigherMoment:
    def test_gaussian_no_flags(self):
        x = RNG.normal(0, 1, 5000)
        skew, kurt, soft, hard = higher_moment_test(x)
        assert abs(skew) < 0.5
        assert kurt == pytest.approx(3.0, abs=0.5)
        assert soft is False
        assert hard is False

    def test_exponential_flagged(self):
        # Exponential: skewness ≈ 2, Pearson kurtosis ≈ 9 → hard flag.
        x = RNG.standard_exponential(5000)
        skew, kurt, soft, hard = higher_moment_test(x)
        assert skew > 1.0
        assert kurt > 8.0
        assert soft is True
        assert hard is True

    def test_constant_signal(self):
        x = np.full(100, 2.0)
        skew, kurt, soft, hard = higher_moment_test(x)
        assert skew == 0.0
        assert np.isnan(kurt)
        assert soft is False
        assert hard is False

    def test_too_short(self):
        skew, kurt, soft, hard = higher_moment_test(np.array([1.0, 2.0]))
        assert np.isnan(skew)
        assert np.isnan(kurt)


# ---------------------------------------------------------------------------
# vickers_mahrt_screen orchestrator
# ---------------------------------------------------------------------------
def _clean_signals(n=4000):
    return {
        "u": 5.0 + RNG.normal(0, 0.5, n),
        "v": RNG.normal(0, 0.3, n),
        "w": RNG.normal(0, 0.15, n),
        "T": 20.0 + RNG.normal(0, 0.2, n),
        "co2": 700.0 + RNG.normal(0, 5.0, n),
        "h2o": 10.0 + RNG.normal(0, 0.5, n),
    }


class TestVickersMahrtScreen:
    def test_clean_signals_no_hard_flag(self):
        flags = vickers_mahrt_screen(_clean_signals())
        assert flags["vm97_hard_flag"] is False

    def test_expected_keys_present(self):
        flags = vickers_mahrt_screen(_clean_signals())
        for var in ("u", "v", "w", "T", "co2", "h2o"):
            assert f"vm97_{var}_n_spikes" in flags
            assert f"vm97_{var}_skewness" in flags
            assert f"vm97_{var}_kurtosis" in flags
            assert f"vm97_{var}_ampres_flag" in flags
            assert f"vm97_{var}_dropout_flag" in flags
            assert f"vm97_{var}_abslim_flag" in flags
        assert "vm97_hard_flag" in flags
        assert "vm97_soft_flag" in flags
        assert "vm97_n_spikes" in flags

    def test_flag_values_are_plain_python_scalars(self):
        # Values must be JSON/Polars-friendly for results_to_dataframe.
        flags = vickers_mahrt_screen(_clean_signals())
        assert isinstance(flags["vm97_hard_flag"], bool)
        assert isinstance(flags["vm97_u_n_spikes"], int)
        assert isinstance(flags["vm97_u_ampres_flag"], bool)

    def test_disabled_returns_empty(self):
        flags = vickers_mahrt_screen(
            _clean_signals(), ScreeningConfig(enabled=False)
        )
        assert flags == {}

    def test_absolute_limit_violation_sets_hard_flag(self):
        sig = _clean_signals()
        sig["w"] = sig["w"].copy()
        sig["w"][10] = 500.0  # unphysical vertical wind
        flags = vickers_mahrt_screen(sig)
        assert flags["vm97_w_abslim_flag"] is True
        assert flags["vm97_hard_flag"] is True

    def test_total_spike_count_is_sum(self):
        sig = _clean_signals()
        sig["u"] = sig["u"].copy()
        sig["u"][100] = 100.0
        flags = vickers_mahrt_screen(sig)
        per_var = sum(flags[f"vm97_{v}_n_spikes"] for v in ("u", "v", "w", "T", "co2", "h2o"))
        assert flags["vm97_n_spikes"] == per_var
        assert flags["vm97_u_n_spikes"] >= 1


# ---------------------------------------------------------------------------
# Integration with process_interval / process_file
# ---------------------------------------------------------------------------
class TestScreeningIntegration:
    def test_process_interval_records_vm97_flags(self, full_interval_arrays, site_config):
        u, v, w, T, co2, h2o = full_interval_arrays
        res = process_interval_call(u, v, w, T, co2, h2o, site_config)
        assert "vm97_hard_flag" in res.qc_flags
        assert "vm97_u_n_spikes" in res.qc_flags
        assert res.qc_flags["vm97_hard_flag"] is False

    def test_screening_can_be_disabled(self, full_interval_arrays, site_config):
        u, v, w, T, co2, h2o = full_interval_arrays
        res = process_interval_call(
            u, v, w, T, co2, h2o, site_config,
            screening_config=ScreeningConfig(enabled=False),
        )
        assert not any(k.startswith("vm97_") for k in res.qc_flags)

    def test_spikes_in_raw_data_are_counted(self, full_interval_arrays, site_config):
        u, v, w, T, co2, h2o = [a.copy() for a in full_interval_arrays]
        u[1000] = 80.0
        u[20000] = -80.0
        res = process_interval_call(u, v, w, T, co2, h2o, site_config)
        assert res.qc_flags["vm97_u_n_spikes"] >= 2

    def test_screening_recorded_even_when_too_many_nans(self, full_interval_arrays, site_config):
        u, v, w, T, co2, h2o = [a.copy() for a in full_interval_arrays]
        u[:3000] = np.nan  # >5% NaN → too_many_nans short-circuit
        res = process_interval_call(u, v, w, T, co2, h2o, site_config)
        assert res.qc_flags.get("too_many_nans") is True
        assert "vm97_hard_flag" in res.qc_flags

    def test_screening_does_not_change_fluxes(self, full_interval_arrays, site_config):
        u, v, w, T, co2, h2o = full_interval_arrays
        with_screen = process_interval_call(u, v, w, T, co2, h2o, site_config)
        without = process_interval_call(
            u, v, w, T, co2, h2o, site_config,
            screening_config=ScreeningConfig(enabled=False),
        )
        assert with_screen.cov_wT == without.cov_wT
        assert with_screen.ustar == without.ustar
        assert np.array_equal(with_screen.freq, without.freq)


def process_interval_call(*args, **kwargs):
    """Import lazily so the module-level import matches the rest of the suite."""
    from TaylorSwift.core import process_interval

    return process_interval(*args, **kwargs)
