"""Regression coverage for API-specific behavior around shared kernels."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from TaylorSwift import constants, core, data_quality, despike, ec_spectral, thermo
from TaylorSwift.transfer_functions import tf_sensor_separation


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("values", [[], [np.nan], [2, np.nan], [1, 3, np.nan, 7]])
def test_detrend_preserves_dtype_missing_values_and_input(dtype, values):
    x = np.array(values, dtype=dtype)
    original = x.copy()
    for function, expected_dtype in [
        (core._detrend_linear, dtype),
        (data_quality._detrend_linear, np.float64),
    ]:
        result = function(x)
        assert result.dtype == expected_dtype
        np.testing.assert_array_equal(np.isnan(result), np.isnan(x))
        if np.isfinite(x).sum() >= 2:
            np.testing.assert_allclose(result[np.isfinite(x)], 0, atol=1e-6)
        else:
            np.testing.assert_array_equal(result, x)
        np.testing.assert_array_equal(x, original)


def test_quality_detrend_accepts_integers():
    np.testing.assert_allclose(data_quality._detrend_linear(np.array([1, 3, 5])), 0)


def test_mad_apis_retain_distinct_nan_policies():
    values = np.array([0.0, 1.0, 2.0, 100.0, np.nan])
    assert not despike.mad_outliers(values).any()
    np.testing.assert_array_equal(
        data_quality.OutlierDetection.mad_outliers(values),
        [False, False, False, True, False],
    )
    for function in (despike.mad_outliers, data_quality.OutlierDetection.mad_outliers):
        assert not function(np.ones(5)).any()
        assert function(values[:-1])[-1]


def test_rolling_filters_retain_window_and_zero_variance_policies():
    frame = pl.DataFrame({
        "TIMESTAMP": [datetime(2026, 1, 1) + timedelta(seconds=i) for i in range(4)],
        "Uz": [1.0, 1.0, 1.0, 10.0],
    }).reverse()
    quality = data_quality.rolling_sigma_filter(frame, sigma=1, keep_stats=True)
    spikes = despike.rolling_sigma_filter(frame, sigma=1)
    assert quality["Uz_roll_mean"][-1] == 1.0
    assert quality["Uz_filtered"][-1] == 10.0
    assert spikes["Uz_roll_mean"][-1] == 3.25
    assert spikes["Uz_filtered"][-1] is None
    assert "Uz_roll_mean" not in data_quality.rolling_sigma_filter(frame).columns
    custom = despike.rolling_sigma_filter(
        frame.rename({"TIMESTAMP": "time"}), time_col="time", closed="left",
        output_col="clean", keep_stats=False, ensure_datetime=False,
    )
    assert custom["clean"][-1] is None
    assert "Uz_roll_mean" not in custom.columns


def test_separation_apis_retain_keywords_and_cutoffs():
    freq = np.array([0.0, 1.0, 10.0], dtype=np.float32)
    expected = np.exp(-9.9 * (freq * 0.2 / 2.0) ** 1.5)
    result = tf_sensor_separation(freq, u_mean=2.0, separation=0.2)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype
    expected_float64 = np.exp(-9.9 * (freq.astype(float) * 0.2 / 2.0) ** 1.5)
    np.testing.assert_array_equal(
        ec_spectral.tf_lateral_separation(n=freq, s=0.2, u=2), expected_float64,
    )
    np.testing.assert_array_equal(tf_sensor_separation(freq, 0.1, 0.2), np.ones(3))
    assert ec_spectral.tf_lateral_separation(freq, 0.2, 0.1)[-1] < 1


def test_constants_and_thermodynamic_defaults_remain_compatible():
    assert ec_spectral.D_MOL is constants.D_MOL
    assert ec_spectral.NU_AIR == constants.NU_AIR == 1.5e-5
    assert thermo.convert_CtoK(0) == 273.15
    assert thermo.convert_KtoC(273.15) == 0
    assert thermo.calc_E(1, 300) == 461.51 * 300
    assert thermo.calc_pV(461.51 * 300, 300) == 1
