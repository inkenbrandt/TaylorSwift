"""
Tests for TaylorSwift.covariance.

calc_max_covariance was rewritten for speed (Phase 1 of the roadmap); these
tests pin it to a straightforward reference implementation of the original
per-lag loop so the optimisation cannot drift from the naive semantics.
"""

import numpy as np
import pytest

from TaylorSwift.covariance import (
    build_covariance_dict,
    calc_cov,
    calc_max_covariance,
)


# ---------------------------------------------------------------------------
# Reference implementation (the pre-optimisation per-lag loop)
# ---------------------------------------------------------------------------
def _reference_max_covariance(x, y, lag=10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    candidates = []
    for k in range(-lag, lag + 1):
        if k < 0:
            xv, yv = x[:k], y[-k:]
        elif k > 0:
            xv, yv = x[k:], y[:-k]
        else:
            xv, yv = x, y
        if len(xv) == 0 or len(yv) == 0:
            continue
        candidates.append((k, calc_cov(xv, yv)))
    if not candidates:
        return []
    return [max(candidates, key=lambda item: abs(item[1]))]


def _assert_same_result(result, expected):
    assert len(result) == len(expected)
    if not expected:
        return
    (k_new, cov_new), (k_ref, cov_ref) = result[0], expected[0]
    assert k_new == k_ref
    if np.isnan(cov_ref):
        assert np.isnan(cov_new)
    else:
        assert cov_new == pytest.approx(cov_ref, rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# calc_cov
# ---------------------------------------------------------------------------
class TestCalcCov:
    def test_matches_numpy_cov(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 500)
        y = 0.5 * x + rng.normal(0, 1, 500)
        assert calc_cov(x, y) == pytest.approx(np.cov(x, y)[0, 1])

    def test_nan_pairs_excluded(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, np.nan])
        expected = np.cov([1.0, 2.0, 4.0], [2.0, 4.0, 8.0])[0, 1]
        assert calc_cov(x, y) == pytest.approx(expected)

    def test_all_nan_returns_nan(self):
        x = np.full(10, np.nan)
        assert np.isnan(calc_cov(x, x))


# ---------------------------------------------------------------------------
# calc_max_covariance equivalence with the reference loop
# ---------------------------------------------------------------------------
class TestMaxCovarianceEquivalence:
    @pytest.mark.parametrize("n", [50, 200, 3000])
    @pytest.mark.parametrize("lag", [1, 5, 10])
    def test_clean_random_data(self, n, lag):
        rng = np.random.default_rng(n + lag)
        x = rng.normal(0, 1, n)
        y = 0.4 * x + rng.normal(0, 1, n)
        _assert_same_result(
            calc_max_covariance(x, y, lag=lag),
            _reference_max_covariance(x, y, lag=lag),
        )

    @pytest.mark.parametrize("nan_frac", [0.02, 0.3])
    def test_data_with_nans(self, nan_frac):
        rng = np.random.default_rng(11)
        n = 1000
        x = rng.normal(5.0, 1.0, n)
        y = 0.4 * np.roll(x, 3) + rng.normal(0, 1, n)
        x[rng.random(n) < nan_frac] = np.nan
        y[rng.random(n) < nan_frac] = np.nan
        _assert_same_result(
            calc_max_covariance(x, y, lag=10),
            _reference_max_covariance(x, y, lag=10),
        )

    def test_large_mean_signal(self):
        """CO2-density-like series: large mean, small fluctuations."""
        rng = np.random.default_rng(21)
        n = 2000
        x = rng.normal(0, 0.15, n)
        y = 700.0 + 0.3 * x + rng.normal(0, 5.0, n)
        _assert_same_result(
            calc_max_covariance(x, y, lag=10),
            _reference_max_covariance(x, y, lag=10),
        )

    def test_known_shift_is_recovered(self):
        rng = np.random.default_rng(3)
        base = rng.normal(0, 1, 5000)
        shift = 4
        x = base[shift:]
        y = 2.0 * base[:-shift] + rng.normal(0, 0.05, 5000 - shift)
        result = calc_max_covariance(x, y, lag=10)
        assert result[0][0] == -shift or result[0][0] == shift
        _assert_same_result(result, _reference_max_covariance(x, y, lag=10))

    def test_constant_input(self):
        x = np.full(100, 3.7)
        _assert_same_result(
            calc_max_covariance(x, x, lag=5),
            _reference_max_covariance(x, x, lag=5),
        )

    def test_all_nan_input(self):
        x = np.full(50, np.nan)
        y = np.arange(50, dtype=float)
        _assert_same_result(
            calc_max_covariance(x, y, lag=5),
            _reference_max_covariance(x, y, lag=5),
        )

    @pytest.mark.parametrize("n", [0, 1, 2, 5])
    def test_tiny_arrays(self, n):
        rng = np.random.default_rng(n)
        x = rng.normal(0, 1, n)
        y = rng.normal(0, 1, n)
        _assert_same_result(
            calc_max_covariance(x, y, lag=10),
            _reference_max_covariance(x, y, lag=10),
        )

    def test_lag_zero(self):
        rng = np.random.default_rng(9)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        result = calc_max_covariance(x, y, lag=0)
        assert result[0][0] == 0
        assert result[0][1] == pytest.approx(calc_cov(x, y), rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# build_covariance_dict
# ---------------------------------------------------------------------------
class TestBuildCovarianceDict:
    def test_all_pairs_present_and_match_single_calls(self):
        rng = np.random.default_rng(5)
        n = 500
        vel = {"Ux": rng.normal(3, 1, n), "Uz": rng.normal(0, 0.2, n)}
        var = {"Ts": rng.normal(20, 0.5, n), "Q": rng.normal(0.01, 0.001, n)}
        out = build_covariance_dict(vel, var, lag=10)
        assert set(out) == {"Ux-Ts", "Ux-Q", "Uz-Ts", "Uz-Q"}
        for ik, iv in vel.items():
            for jk, jv in var.items():
                expected = _reference_max_covariance(iv, jv, lag=10)[0][1]
                assert out[f"{ik}-{jk}"] == pytest.approx(
                    expected, rel=1e-9, abs=1e-12
                )
