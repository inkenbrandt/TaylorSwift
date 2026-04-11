"""
Tests for eccospectra.ec_polars — CalcFlux class and Polars/pandas helpers.

Integration tests for runall/run_irga are intentionally omitted as they
require fully labelled, sensor-specific DataFrames.  These tests focus on
pure utility methods and construction behaviour.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from eccospectra.ec_polars import (
    CalcFlux,
    _to_pl_df,
    _to_same_type,
    _get_series,
    _assign,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestCalcFluxInit:
    def test_default_instantiation(self):
        cf = CalcFlux()
        assert cf is not None

    def test_default_meter_type(self):
        cf = CalcFlux()
        assert cf.meter_type == "IRGASON"

    def test_default_u_height(self):
        cf = CalcFlux()
        assert cf.UHeight == pytest.approx(3.52)

    def test_default_lag(self):
        cf = CalcFlux()
        assert cf.lag == 10

    def test_kwargs_override_defaults(self):
        cf = CalcFlux(UHeight=5.0, meter_type="KH20", lag=20)
        assert cf.UHeight == 5.0
        assert cf.meter_type == "KH20"
        assert cf.lag == 20

    def test_physical_constants_set(self):
        cf = CalcFlux()
        assert cf.Rd == pytest.approx(287.05)
        assert cf.Rv == pytest.approx(461.51)
        assert cf.von_karman == pytest.approx(0.41)

    def test_containers_empty_at_init(self):
        cf = CalcFlux()
        assert isinstance(cf.covar, dict)
        assert isinstance(cf.avgvals, dict)
        assert len(cf.covar) == 0

    def test_despikefields_populated(self):
        cf = CalcFlux()
        assert "Ux" in cf.despikefields
        assert "Uz" in cf.despikefields


# ---------------------------------------------------------------------------
# Temperature conversions
# ---------------------------------------------------------------------------

class TestTemperatureConversions:
    def setup_method(self):
        self.cf = CalcFlux()

    def test_ktoc_zero_celsius(self):
        assert self.cf.convert_KtoC(273.15) == pytest.approx(0.0)

    def test_ctok_zero_celsius(self):
        assert self.cf.convert_CtoK(0.0) == pytest.approx(273.15)

    def test_round_trip_scalar(self):
        T_K = 300.0
        assert self.cf.convert_CtoK(self.cf.convert_KtoC(T_K)) == pytest.approx(T_K)

    def test_array_input(self):
        T_C = np.array([0.0, 20.0, -10.0])
        T_K = self.cf.convert_CtoK(T_C)
        assert T_K.shape == T_C.shape
        np.testing.assert_allclose(T_K, T_C + 273.15)

    def test_ktoc_array(self):
        T_K = np.array([273.15, 293.15, 263.15])
        T_C = self.cf.convert_KtoC(T_K)
        np.testing.assert_allclose(T_C, [0.0, 20.0, -10.0])


# ---------------------------------------------------------------------------
# calc_cov
# ---------------------------------------------------------------------------

class TestCalcCov:
    def setup_method(self):
        self.cf = CalcFlux()

    def test_covariance_of_identical_is_variance(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        cov = self.cf.calc_cov(x, x)
        assert cov == pytest.approx(np.var(x, ddof=0), rel=1e-6)

    def test_uncorrelated_signals_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 5000)
        y = rng.normal(0, 1, 5000)
        cov = self.cf.calc_cov(x, y)
        assert abs(cov) < 0.1

    def test_perfectly_correlated_positive(self):
        x = np.arange(100, dtype=float)
        cov = self.cf.calc_cov(x, x)
        assert cov > 0

    def test_anti_correlated_negative(self):
        x = np.arange(100, dtype=float)
        cov = self.cf.calc_cov(x, -x)
        assert cov < 0


# ---------------------------------------------------------------------------
# calc_MSE
# ---------------------------------------------------------------------------

class TestCalcMSE:
    def setup_method(self):
        self.cf = CalcFlux()

    def test_zeros_gives_zero(self):
        x = np.zeros(100)
        assert self.cf.calc_MSE(x) == pytest.approx(0.0)

    def test_constant_gives_zero(self):
        x = np.full(100, 5.0)
        assert self.cf.calc_MSE(x) == pytest.approx(0.0)

    def test_non_negative(self):
        rng = np.random.default_rng(7)
        x = rng.normal(10, 2, 500)
        assert self.cf.calc_MSE(x) >= 0.0


# ---------------------------------------------------------------------------
# calc_Es (saturation vapour pressure)
# ---------------------------------------------------------------------------

class TestCalcEs:
    def setup_method(self):
        self.cf = CalcFlux()

    def test_increases_with_temperature(self):
        T_warm = 300.0
        T_cold = 270.0
        assert self.cf.calc_Es(T_warm) > self.cf.calc_Es(T_cold)

    def test_positive(self):
        T = np.array([270.0, 280.0, 290.0, 300.0])
        Es = self.cf.calc_Es(T)
        assert (Es > 0).all()

    def test_scalar_input(self):
        Es = self.cf.calc_Es(293.15)
        assert isinstance(Es, float)


# ---------------------------------------------------------------------------
# tetens (saturation vapour pressure alternative)
# ---------------------------------------------------------------------------

class TestTetens:
    def setup_method(self):
        self.cf = CalcFlux()

    def test_positive_output(self):
        result = self.cf.tetens(np.array([10.0, 20.0, 30.0]))
        assert (np.asarray(result) > 0).all()

    def test_increases_with_temperature(self):
        assert self.cf.tetens(np.array([30.0])) > self.cf.tetens(np.array([10.0]))


# ---------------------------------------------------------------------------
# shadow_correction
# ---------------------------------------------------------------------------

class TestShadowCorrection:
    def setup_method(self):
        self.cf = CalcFlux()

    def test_returns_three_arrays(self):
        n = 100
        Ux = np.random.default_rng(0).normal(2, 0.5, n)
        Uy = np.random.default_rng(1).normal(0, 0.5, n)
        Uz = np.random.default_rng(2).normal(0, 0.2, n)
        result = self.cf.shadow_correction(Ux, Uy, Uz)
        assert len(result) == 3

    def test_shapes_preserved(self):
        n = 200
        Ux = np.ones(n)
        Uy = np.zeros(n)
        Uz = np.zeros(n)
        Ux_c, Uy_c, Uz_c = self.cf.shadow_correction(Ux, Uy, Uz)
        assert Ux_c.shape == (n,)
        assert Uy_c.shape == (n,)
        assert Uz_c.shape == (n,)


# ---------------------------------------------------------------------------
# Polars/pandas compatibility helpers
# ---------------------------------------------------------------------------

class TestCompatHelpers:
    def _pd_df(self):
        return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    def _pl_df(self):
        return pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    def test_to_pl_df_from_polars(self):
        df = self._pl_df()
        result = _to_pl_df(df)
        assert isinstance(result, pl.DataFrame)

    def test_to_pl_df_from_pandas(self):
        df = self._pd_df()
        result = _to_pl_df(df)
        assert isinstance(result, pl.DataFrame)

    def test_to_pl_df_raises_on_invalid(self):
        with pytest.raises(TypeError):
            _to_pl_df([1, 2, 3])

    def test_to_same_type_returns_polars_when_original_polars(self):
        orig = self._pl_df()
        converted = _to_pl_df(orig)
        result = _to_same_type(orig, converted)
        assert isinstance(result, pl.DataFrame)

    def test_to_same_type_returns_pandas_when_original_pandas(self):
        orig = self._pd_df()
        converted = _to_pl_df(orig)
        result = _to_same_type(orig, converted)
        assert isinstance(result, pd.DataFrame)

    def test_get_series_from_polars(self):
        df = self._pl_df()
        s = _get_series(df, "a")
        assert isinstance(s, pl.Series)
        assert list(s) == [1.0, 2.0, 3.0]

    def test_get_series_from_pandas(self):
        df = self._pd_df()
        s = _get_series(df, "a")
        assert isinstance(s, pl.Series)

    def test_assign_to_polars(self):
        df = self._pl_df()
        result = _assign(df, c=np.array([7.0, 8.0, 9.0]))
        assert "c" in result.columns

    def test_assign_to_pandas(self):
        df = self._pd_df()
        result = _assign(df, c=np.array([7.0, 8.0, 9.0]))
        assert "c" in result.columns


# ---------------------------------------------------------------------------
# calc_LnKh
# ---------------------------------------------------------------------------

class TestCalcLnKh:
    def setup_method(self):
        self.cf = CalcFlux()

    def test_positive_input_returns_finite(self):
        result = self.cf.calc_LnKh(500.0)
        assert np.isfinite(result)

    def test_array_input(self):
        mvolts = np.array([100.0, 200.0, 500.0])
        result = self.cf.calc_LnKh(mvolts)
        assert np.all(np.isfinite(result))
