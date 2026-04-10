"""
Tests for eccospectra.core — spectral computation, rotation, log-binning,
and the full process_interval / process_file pipeline.
"""

import numpy as np
import pytest
from datetime import datetime

from eccospectra import SiteConfig, compute_cospectrum, compute_spectrum, rotate_wind, process_interval
from eccospectra.core import log_bin, SpectralResult


# ---------------------------------------------------------------------------
# SiteConfig
# ---------------------------------------------------------------------------

class TestSiteConfig:
    def test_defaults(self):
        cfg = SiteConfig()
        assert cfg.z_measurement == 3.0
        assert cfg.sampling_freq == 20.0
        assert cfg.averaging_period == 30.0

    def test_derived_d_and_z0(self):
        cfg = SiteConfig(z_measurement=5.0, z_canopy=1.0)
        assert cfg.d == pytest.approx(2 / 3)
        assert cfg.z0 == pytest.approx(0.1)

    def test_z_eff(self):
        cfg = SiteConfig(z_measurement=3.0, z_canopy=0.3)
        expected = 3.0 - (2 / 3) * 0.3
        assert cfg.z_eff == pytest.approx(expected)

    def test_explicit_d_z0(self):
        cfg = SiteConfig(z_measurement=5.0, z_canopy=1.0, d=0.5, z0=0.05)
        assert cfg.d == 0.5
        assert cfg.z0 == 0.05
        assert cfg.z_eff == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# rotate_wind
# ---------------------------------------------------------------------------

class TestRotateWind:
    def test_mean_v_w_are_zero_after_rotation(self, short_white_noise):
        u, v, w, *_ = short_white_noise
        u_rot, v_rot, w_rot, _ = rotate_wind(u, v, w)
        assert np.nanmean(v_rot) == pytest.approx(0.0, abs=1e-10)
        assert np.nanmean(w_rot) == pytest.approx(0.0, abs=1e-10)

    def test_wind_speed_preserved(self, short_white_noise):
        u, v, w, *_ = short_white_noise
        speed_before = np.sqrt(np.nanmean(u)**2 + np.nanmean(v)**2 + np.nanmean(w)**2)
        u_rot, v_rot, w_rot, _ = rotate_wind(u, v, w)
        speed_after = np.sqrt(np.nanmean(u_rot)**2 + np.nanmean(v_rot)**2 + np.nanmean(w_rot)**2)
        assert speed_after == pytest.approx(speed_before, rel=1e-8)

    def test_wind_dir_in_range(self, short_white_noise):
        u, v, w, *_ = short_white_noise
        _, _, _, wind_dir = rotate_wind(u, v, w)
        assert 0.0 <= wind_dir < 360.0

    def test_already_aligned_wind(self):
        n = 500
        rng = np.random.default_rng(0)
        u = 5.0 + rng.normal(0, 0.3, n)
        v = rng.normal(0, 0.01, n)   # already near-zero mean
        w = rng.normal(0, 0.01, n)
        u_rot, v_rot, w_rot, wind_dir = rotate_wind(u, v, w)
        assert np.nanmean(v_rot) == pytest.approx(0.0, abs=1e-10)
        assert np.nanmean(w_rot) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# compute_cospectrum
# ---------------------------------------------------------------------------

class TestComputeCospectrum:
    def test_output_shapes(self):
        n = 1024
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, n)
        y = rng.normal(0, 1, n)
        freq, cospec = compute_cospectrum(x, y, fs=20.0)
        assert len(freq) == len(cospec)
        assert len(freq) == n // 2   # rfft output minus DC

    def test_frequencies_positive(self):
        n = 512
        x = np.random.default_rng(2).normal(0, 1, n)
        freq, _ = compute_cospectrum(x, x, fs=10.0)
        assert np.all(freq > 0)

    def test_self_cospectrum_equals_spectrum(self):
        n = 512
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, n)
        freq_co, cospec = compute_cospectrum(x, x, fs=20.0)
        freq_sp, psd = compute_spectrum(x, fs=20.0)
        np.testing.assert_allclose(freq_co, freq_sp)
        np.testing.assert_allclose(cospec, psd, rtol=1e-12)

    def test_parseval_theorem(self):
        """Integral of PSD ≈ variance of the signal (Parseval's theorem).

        We use a loose tolerance because the Hamming window reduces the
        effective variance, and one-sided doubling is an approximation.
        """
        n = 4096
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, n)
        x -= x.mean()
        fs = 20.0
        freq, psd = compute_spectrum(x, fs=fs)
        # df is roughly constant for rfft
        df = freq[1] - freq[0]
        integral = np.sum(psd) * df
        variance = np.var(x)
        # Allow 30% error: Hamming window reduces the apparent variance
        assert integral == pytest.approx(variance, rel=0.30)

    def test_cross_spectrum_symmetry(self):
        """Co(x,y) and Co(y,x) should be equal (both are real parts)."""
        n = 256
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, n)
        y = rng.normal(0, 1, n)
        _, co_xy = compute_cospectrum(x, y, fs=20.0)
        _, co_yx = compute_cospectrum(y, x, fs=20.0)
        np.testing.assert_allclose(co_xy, co_yx, rtol=1e-12)

    def test_uncorrelated_signals_near_zero_mean(self):
        """Mean cospectrum of independent signals should be near zero."""
        n = 8192
        rng = np.random.default_rng(6)
        x = rng.normal(0, 1, n)
        y = rng.normal(0, 1, n)
        _, cospec = compute_cospectrum(x, y, fs=20.0)
        assert abs(cospec.mean()) < 0.05   # much smaller than auto-spectrum


# ---------------------------------------------------------------------------
# log_bin
# ---------------------------------------------------------------------------

class TestLogBin:
    def test_output_monotone_increasing(self):
        freq = np.logspace(-2, 1, 500)
        spec = np.ones_like(freq)
        fb, _ = log_bin(freq, spec)
        assert np.all(np.diff(fb) > 0)

    def test_bin_count_roughly_correct(self):
        freq = np.logspace(-2, 1, 500)   # 3 decades
        spec = np.ones_like(freq)
        fb, sb = log_bin(freq, spec, bins_per_decade=10)
        # 3 decades × 10 bins/decade = 30 (±a few for populated check)
        assert 20 <= len(fb) <= 35

    def test_constant_spectrum_unchanged(self):
        freq = np.logspace(-2, 1, 500)
        spec = 2.5 * np.ones_like(freq)
        _, sb = log_bin(freq, spec)
        np.testing.assert_allclose(sb, 2.5, rtol=1e-10)

    def test_empty_input(self):
        fb, sb = log_bin(np.array([]), np.array([]))
        assert len(fb) == 0
        assert len(sb) == 0

    def test_nan_values_ignored(self):
        freq = np.logspace(-2, 1, 200)
        spec = np.ones_like(freq)
        spec[10:20] = np.nan
        fb, sb = log_bin(freq, spec)
        assert np.all(np.isfinite(sb))


# ---------------------------------------------------------------------------
# process_interval
# ---------------------------------------------------------------------------

class TestProcessInterval:
    def test_returns_spectral_result(self, full_interval_arrays, site_config):
        u, v, w, T, co2, h2o = full_interval_arrays
        res = process_interval(u, v, w, T, co2, h2o, site_config)
        assert isinstance(res, SpectralResult)

    def test_freq_arrays_populated(self, full_interval_arrays, site_config):
        u, v, w, T, co2, h2o = full_interval_arrays
        res = process_interval(u, v, w, T, co2, h2o, site_config)
        assert len(res.freq) > 0
        assert len(res.freq_nd) > 0

    def test_cospectra_same_length_as_freq(self, full_interval_arrays, site_config):
        u, v, w, T, co2, h2o = full_interval_arrays
        res = process_interval(u, v, w, T, co2, h2o, site_config)
        n = len(res.freq)
        assert len(res.cosp_wT) == n
        assert len(res.cosp_wu) == n
        assert len(res.cosp_wCO2) == n
        assert len(res.cosp_wH2O) == n

    def test_turbulence_statistics_finite(self, full_interval_arrays, site_config):
        u, v, w, T, co2, h2o = full_interval_arrays
        res = process_interval(u, v, w, T, co2, h2o, site_config)
        assert np.isfinite(res.u_mean)
        assert np.isfinite(res.ustar)
        assert np.isfinite(res.T_mean)
        assert res.ustar >= 0

    def test_ustar_positive(self, full_interval_arrays, site_config):
        u, v, w, T, co2, h2o = full_interval_arrays
        res = process_interval(u, v, w, T, co2, h2o, site_config)
        assert res.ustar > 0

    def test_too_many_nans_sets_flag(self, site_config):
        """Intervals with >5% NaN in wind data should set too_many_nans."""
        rng = np.random.default_rng(7)
        n = 36000
        u = rng.normal(5, 0.5, n).astype(float)
        u[:2000] = np.nan  # ~5.6% NaN
        v = rng.normal(0, 0.3, n)
        w = rng.normal(0, 0.15, n)
        T = rng.normal(20, 0.5, n)
        co2 = rng.normal(700, 5, n)
        h2o = rng.normal(10, 0.5, n)
        res = process_interval(u, v, w, T, co2, h2o, site_config)
        assert res.qc_flags.get('too_many_nans') is True

    def test_with_nans_within_threshold(self, interval_with_nans, site_config):
        """Small NaN fraction (<5%) should not trigger too_many_nans."""
        u, v, w, T, co2, h2o = interval_with_nans
        res = process_interval(u, v, w, T, co2, h2o, site_config)
        assert not res.qc_flags.get('too_many_nans', False)

    def test_timestamps_stored(self, full_interval_arrays, site_config):
        u, v, w, T, co2, h2o = full_interval_arrays
        t0 = datetime(2023, 6, 10, 0, 0)
        t1 = datetime(2023, 6, 10, 0, 30)
        res = process_interval(u, v, w, T, co2, h2o, site_config,
                               timestamp_start=t0, timestamp_end=t1)
        assert res.timestamp_start == t0
        assert res.timestamp_end == t1

    def test_normalized_cospectra_finite_where_cov_nonzero(
        self, full_interval_arrays, site_config
    ):
        u, v, w, T, co2, h2o = full_interval_arrays
        res = process_interval(u, v, w, T, co2, h2o, site_config)
        if abs(res.cov_wT) > 1e-10:
            assert np.all(np.isfinite(res.ncosp_wT))

    def test_ogives_same_length_as_freq(self, full_interval_arrays, site_config):
        u, v, w, T, co2, h2o = full_interval_arrays
        res = process_interval(u, v, w, T, co2, h2o, site_config)
        n = len(res.freq)
        assert len(res.ogive_wT) == n
        assert len(res.ogive_wu) == n

    def test_power_spectra_non_negative(self, full_interval_arrays, site_config):
        """Auto-spectra (n·S/σ²) should be non-negative."""
        u, v, w, T, co2, h2o = full_interval_arrays
        res = process_interval(u, v, w, T, co2, h2o, site_config)
        assert np.all(res.spec_u >= 0)
        assert np.all(res.spec_w >= 0)

    def test_sensible_heat_flux_sign(self, site_config):
        """Positive w-T covariance should give positive H."""
        rng = np.random.default_rng(8)
        n = 36000
        w = rng.normal(0, 0.15, n)
        T_fluc = 0.5 * w + rng.normal(0, 0.05, n)
        u = 5.0 + rng.normal(0, 0.3, n)
        v = rng.normal(0, 0.1, n)
        T = 20.0 + T_fluc
        co2 = rng.normal(700, 5, n)
        h2o = rng.normal(10, 0.5, n)
        res = process_interval(u, v, w, T, co2, h2o, site_config)
        assert res.cov_wT > 0
        assert res.H > 0
