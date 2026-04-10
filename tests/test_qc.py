"""
Tests for eccospectra.qc — inertial slope fitting, stationarity, batch QC.
"""

import numpy as np
import pytest

from eccospectra.qc import fit_inertial_slope, classify_slope, stationarity_test, run_qc


# ---------------------------------------------------------------------------
# fit_inertial_slope
# ---------------------------------------------------------------------------

class TestFitInertialSlope:
    def _power_law(self, f, slope, amplitude=1.0):
        return amplitude * f ** slope

    def test_recovers_known_slope_cospectrum(self):
        """Should recover -4/3 slope from a clean power law."""
        f = np.logspace(-1, 1, 200)
        ncosp = self._power_law(f, slope=-4/3)
        slope, r2, _ = fit_inertial_slope(f, ncosp, f_range=(0.5, 8.0))
        assert slope == pytest.approx(-4/3, abs=0.05)
        assert r2 > 0.99

    def test_recovers_known_slope_spectrum(self):
        f = np.logspace(-1, 1, 200)
        spec = self._power_law(f, slope=-2/3)
        slope, r2, _ = fit_inertial_slope(f, spec, f_range=(0.5, 8.0))
        assert slope == pytest.approx(-2/3, abs=0.05)
        assert r2 > 0.99

    def test_returns_nans_when_too_few_points(self):
        """Empty or single-point range should return all NaN."""
        f = np.array([0.1, 0.2, 0.3])
        ncosp = np.array([1.0, 0.8, 0.6])
        # f_range outside the data range → no points
        slope, r2, intercept = fit_inertial_slope(f, ncosp, f_range=(5.0, 10.0))
        assert np.isnan(slope)
        assert np.isnan(r2)
        assert np.isnan(intercept)

    def test_negative_values_handled_via_abs(self):
        """Negative ncosp values (valid in cospectra) should be abs'd."""
        f = np.logspace(-1, 1, 100)
        # Negative power law
        ncosp = -1.0 * f ** (-4/3)
        slope, r2, _ = fit_inertial_slope(f, ncosp, f_range=(0.5, 8.0))
        assert slope == pytest.approx(-4/3, abs=0.1)

    def test_r_squared_between_zero_and_one(self):
        rng = np.random.default_rng(20)
        f = np.logspace(-1, 1, 50)
        ncosp = f ** (-4/3) + rng.normal(0, 0.1, 50)
        _, r2, _ = fit_inertial_slope(f, ncosp, f_range=(0.5, 8.0))
        assert 0.0 <= r2 <= 1.0

    def test_all_zeros_returns_nan(self):
        f = np.logspace(-1, 1, 50)
        ncosp = np.zeros(50)
        slope, r2, intercept = fit_inertial_slope(f, ncosp, f_range=(0.5, 8.0))
        assert np.isnan(slope)


# ---------------------------------------------------------------------------
# classify_slope
# ---------------------------------------------------------------------------

class TestClassifySlope:
    # Cospectrum classification (expected ≈ -4/3)
    def test_good_cospectrum(self):
        assert classify_slope(-4/3, is_cospectrum=True) == 'good'

    def test_acceptable_cospectrum_steep(self):
        assert classify_slope(-2.0, is_cospectrum=True) == 'acceptable'

    def test_suspect_cospectrum(self):
        assert classify_slope(-2.5, is_cospectrum=True) == 'suspect'

    def test_bad_cospectrum_positive(self):
        assert classify_slope(0.5, is_cospectrum=True) == 'bad'

    def test_bad_on_nan(self):
        assert classify_slope(np.nan, is_cospectrum=True) == 'bad'
        assert classify_slope(np.nan, is_cospectrum=False) == 'bad'

    # Power spectrum classification (expected ≈ -2/3)
    def test_good_spectrum(self):
        assert classify_slope(-2/3, is_cospectrum=False) == 'good'

    def test_acceptable_spectrum(self):
        assert classify_slope(-1.3, is_cospectrum=False) == 'acceptable'

    def test_bad_spectrum_far_off(self):
        assert classify_slope(-3.0, is_cospectrum=False) == 'bad'


# ---------------------------------------------------------------------------
# stationarity_test
# ---------------------------------------------------------------------------

class TestStationarityTest:
    def _make_stationary_pair(self, n=6000, seed=30):
        rng = np.random.default_rng(seed)
        w = rng.normal(0, 0.15, n)
        T_fluc = 0.4 * w + rng.normal(0, 0.05, n)
        return w, T_fluc

    def _make_nonstationarity_via_step(self, n=6000, seed=31):
        """
        Signal with a sharp step in the w-T product at the midpoint.

        Returns w and T arrays constructed so that w_det_full * T_det_full
        has a very different distribution from sub-window products — achieved
        by introducing a large mean offset in T partway through the interval.
        The global linear detrend cannot remove a step change, so
        mean(sub_cov) << cov_full.
        """
        rng = np.random.default_rng(seed)
        half = n // 2
        w = rng.normal(0, 0.2, n)
        # Large positive step change in T at mid-interval; no w-correlation
        T_base = np.concatenate([np.zeros(half), np.full(half, 20.0)])
        T = T_base + rng.normal(0, 0.01, n)
        # Sub-window detrend handles the step within each half fine,
        # but each sub-window cov(w, T) ≈ 0 since T is uncorrelated with w
        # Full-interval detrend removes only a linear ramp, leaving a large
        # residual step that creates an apparent covariance with w fluctuations
        return w, T

    def test_returns_tuple(self):
        w, T = self._make_stationary_pair()
        result = stationarity_test(w, T, fs=20.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_rel_diff_non_negative(self):
        w, T = self._make_stationary_pair()
        rel_diff, qc = stationarity_test(w, T, fs=20.0)
        assert rel_diff >= 0 or np.isnan(rel_diff)

    def test_quality_class_in_range(self):
        w, T = self._make_stationary_pair()
        _, qc = stationarity_test(w, T, fs=20.0)
        assert qc in (1, 2, 3, 4)

    def test_stationary_data_has_good_class(self):
        w, T = self._make_stationary_pair(n=36000)
        rel_diff, qc = stationarity_test(w, T, fs=20.0)
        # Stationary pair should get QC class 1 or 2
        assert qc <= 2

    def test_nonstationary_data_has_poor_class(self):
        """Step change in T mid-interval should trigger nonstationarity flag."""
        w, T = self._make_nonstationarity_via_step(n=36000)
        rel_diff, qc = stationarity_test(w, T, fs=20.0)
        assert qc >= 3

    def test_zero_covariance_returns_nan_and_class4(self):
        rng = np.random.default_rng(32)
        w = rng.normal(0, 1, 1000)
        T = np.zeros(1000)   # no covariance with w
        rel_diff, qc = stationarity_test(w, T, fs=20.0)
        assert np.isnan(rel_diff)
        assert qc == 4

    def test_custom_n_subwindows(self):
        w, T = self._make_stationary_pair(n=36000)
        rel_diff, qc = stationarity_test(w, T, fs=20.0, n_subwindows=3)
        assert qc in (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# run_qc
# ---------------------------------------------------------------------------

class TestRunQC:
    def test_adds_slope_flags(self, spectral_result_stub):
        results = run_qc([spectral_result_stub])
        res = results[0]
        for key in ('slope_wT', 'slope_wu', 'slope_wCO2', 'slope_wH2O'):
            assert key in res.qc_flags

    def test_adds_slope_class_flags(self, spectral_result_stub):
        results = run_qc([spectral_result_stub])
        res = results[0]
        for key in ('slope_class_wT', 'slope_class_wu'):
            assert key in res.qc_flags
            assert res.qc_flags[key] in ('good', 'acceptable', 'suspect', 'bad')

    def test_adds_ustar_filter(self, spectral_result_stub):
        results = run_qc([spectral_result_stub])
        res = results[0]
        assert 'ustar_filter' in res.qc_flags

    def test_ustar_filter_true_below_threshold(self, spectral_result_stub):
        spectral_result_stub.ustar = 0.05   # below 0.1 threshold
        results = run_qc([spectral_result_stub])
        assert results[0].qc_flags['ustar_filter'] is True

    def test_ustar_filter_false_above_threshold(self, spectral_result_stub):
        spectral_result_stub.ustar = 0.5
        results = run_qc([spectral_result_stub])
        assert results[0].qc_flags['ustar_filter'] is False

    def test_empty_list_returns_empty(self):
        results = run_qc([])
        assert results == []

    def test_result_with_empty_freq_skipped(self):
        from eccospectra.core import SpectralResult
        res = SpectralResult()   # empty arrays by default
        run_qc([res])            # should not raise
        assert res.qc_flags == {}

    def test_returns_same_list(self, spectral_result_stub):
        """run_qc mutates in-place and returns the same list."""
        results_in = [spectral_result_stub]
        results_out = run_qc(results_in)
        assert results_out is results_in
