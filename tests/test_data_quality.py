"""
Tests for eccospectra.data_quality — quality flags, stationarity dataclasses,
DataQuality assessment, OutlierDetection, quality_filter, rolling_sigma_filter.
"""

import numpy as np
import pytest
import polars as pl

from eccospectra.data_quality import (
    QualityFlag,
    StabilityParameters,
    StationarityTest,
    DataQuality,
    OutlierDetection,
    quality_filter,
    rolling_sigma_filter,
)


# ---------------------------------------------------------------------------
# QualityFlag enum
# ---------------------------------------------------------------------------

class TestQualityFlag:
    def test_class_1_is_one(self):
        assert int(QualityFlag.CLASS_1) == 1

    def test_class_9_is_nine(self):
        assert int(QualityFlag.CLASS_9) == 9

    def test_ordering(self):
        assert QualityFlag.CLASS_1 < QualityFlag.CLASS_9


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------

class TestStabilityParameters:
    def test_construction(self):
        sp = StabilityParameters(
            z=3.5, L=-50.0, u_star=0.45,
            sigma_w=0.26, sigma_T=0.18,
            T_star=-0.40, latitude=40.77,
        )
        assert sp.z == 3.5
        assert sp.u_star == 0.45

    def test_all_fields_accessible(self):
        sp = StabilityParameters(
            z=5.0, L=100.0, u_star=0.3,
            sigma_w=0.2, sigma_T=0.1,
            T_star=0.1, latitude=51.5,
        )
        assert sp.L == 100.0
        assert sp.latitude == 51.5


class TestStationarityTestDataclass:
    def test_construction(self):
        st = StationarityTest(RN_uw=0.1, RN_wT=0.2, RN_wq=0.3, RN_wc=0.4)
        assert st.RN_uw == 0.1
        assert st.RN_wc == 0.4


# ---------------------------------------------------------------------------
# DataQuality
# ---------------------------------------------------------------------------

def _make_stability(z_L=-0.1):
    """Helper: StabilityParameters with controllable z/L."""
    z = 3.5
    L = z / z_L if z_L != 0 else 1e9
    return StabilityParameters(
        z=z, L=L, u_star=0.5, sigma_w=0.3,
        sigma_T=0.2, T_star=-0.3, latitude=40.0,
    )


class TestDataQuality:
    def setup_method(self):
        self.dq = DataQuality()
        self.stationary = StationarityTest(RN_uw=0.05, RN_wT=0.05, RN_wq=0.05, RN_wc=0.05)
        self.nonstationarity = StationarityTest(RN_uw=5.0, RN_wT=5.0, RN_wq=5.0, RN_wc=5.0)

    def test_assess_returns_dict(self):
        result = self.dq.assess_data_quality(
            _make_stability(), self.stationary, flux_type="momentum"
        )
        assert isinstance(result, dict)

    def test_result_has_required_keys(self):
        result = self.dq.assess_data_quality(
            _make_stability(), self.stationary, flux_type="heat"
        )
        for key in ("overall_flag", "stationarity_flag", "itc_flag", "wind_dir_flag"):
            assert key in result

    def test_overall_flag_in_range(self):
        result = self.dq.assess_data_quality(
            _make_stability(), self.stationary, flux_type="momentum"
        )
        assert 1 <= result["overall_flag"] <= 9

    def test_good_stationarity_gives_low_flag(self):
        result = self.dq.assess_data_quality(
            _make_stability(-0.1), self.stationary, flux_type="heat"
        )
        assert result["stationarity_flag"] <= 3

    def test_poor_stationarity_gives_high_flag(self):
        result = self.dq.assess_data_quality(
            _make_stability(-0.1), self.nonstationarity, flux_type="co2"
        )
        assert result["stationarity_flag"] >= 6

    def test_wind_direction_clean_sector(self):
        result = self.dq.assess_data_quality(
            _make_stability(), self.stationary,
            wind_direction=90.0, flux_type="momentum"
        )
        assert result["wind_dir_flag"] == QualityFlag.CLASS_1

    def test_wind_direction_bad_sector(self):
        result = self.dq.assess_data_quality(
            _make_stability(), self.stationary,
            wind_direction=180.0, flux_type="momentum"
        )
        assert result["wind_dir_flag"] >= QualityFlag.CLASS_7

    def test_no_wind_direction_defaults_class1(self):
        result = self.dq.assess_data_quality(
            _make_stability(), self.stationary,
            wind_direction=None, flux_type="momentum"
        )
        assert result["wind_dir_flag"] == QualityFlag.CLASS_1

    def test_overall_is_max_of_individual(self):
        result = self.dq.assess_data_quality(
            _make_stability(), self.stationary,
            wind_direction=180.0, flux_type="momentum"
        )
        assert result["overall_flag"] == max(
            result["stationarity_flag"],
            result["itc_flag"],
            result["wind_dir_flag"],
        )

    def test_disable_wind_direction_flag(self):
        dq = DataQuality(use_wind_direction=False)
        result = dq.assess_data_quality(
            _make_stability(), self.stationary,
            wind_direction=180.0, flux_type="momentum"
        )
        assert result["wind_dir_flag"] == QualityFlag.CLASS_1

    def test_get_quality_label_known(self):
        assert self.dq.get_quality_label(1) == "Highest quality"
        assert self.dq.get_quality_label(9) == "Very poor quality (discard)"

    def test_get_quality_label_unknown(self):
        assert self.dq.get_quality_label(99) == "Unknown"

    def test_itc_measured_and_modeled_in_result(self):
        result = self.dq.assess_data_quality(
            _make_stability(-0.1), self.stationary
        )
        assert "itc_measured" in result
        assert "itc_modeled" in result
        assert result["itc_measured"] > 0

    def test_stable_conditions(self):
        result = self.dq.assess_data_quality(
            _make_stability(z_L=0.5), self.stationary
        )
        assert 1 <= result["overall_flag"] <= 9

    def test_near_neutral_unstable(self):
        result = self.dq.assess_data_quality(
            _make_stability(z_L=-0.01), self.stationary
        )
        assert 1 <= result["overall_flag"] <= 9

    def test_all_flux_types(self):
        for ft in ("momentum", "heat", "moisture", "co2"):
            result = self.dq.assess_data_quality(
                _make_stability(), self.stationary, flux_type=ft
            )
            assert "overall_flag" in result


# ---------------------------------------------------------------------------
# OutlierDetection
# ---------------------------------------------------------------------------

class TestMadOutliers:
    def test_obvious_outlier_detected(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 1.0, 50)
        x = np.append(x, 100.0)  # inject obvious outlier
        mask = OutlierDetection.mad_outliers(x)
        assert mask[-1] == True

    def test_clean_data_no_outliers(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        mask = OutlierDetection.mad_outliers(x)
        assert mask.sum() < 10  # very few should be flagged

    def test_zero_mad_returns_all_false(self):
        x = np.ones(10)
        mask = OutlierDetection.mad_outliers(x)
        assert not mask.any()

    def test_output_shape_matches_input(self):
        x = np.arange(50, dtype=float)
        mask = OutlierDetection.mad_outliers(x)
        assert mask.shape == x.shape

    def test_custom_threshold(self):
        x = np.array([0.0, 0.0, 0.0, 0.0, 3.0])
        mask_strict = OutlierDetection.mad_outliers(x, threshold=2.0)
        mask_loose = OutlierDetection.mad_outliers(x, threshold=5.0)
        # stricter threshold should flag at least as many
        assert mask_strict.sum() >= mask_loose.sum()


class TestSpikeDetection:
    def test_injected_spike_detected(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 500)
        x[250] = 20.0
        mask = OutlierDetection.spike_detection(x, window_size=51, z_threshold=4.0)
        assert mask[250]

    def test_clean_signal_few_flags(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 500)
        mask = OutlierDetection.spike_detection(x, window_size=51, z_threshold=6.0)
        assert mask.sum() < 5

    def test_output_dtype_bool(self):
        x = np.random.default_rng(2).normal(0, 1, 100)
        mask = OutlierDetection.spike_detection(x)
        assert mask.dtype == bool

    def test_output_shape(self):
        x = np.zeros(200)
        mask = OutlierDetection.spike_detection(x)
        assert mask.shape == (200,)


# ---------------------------------------------------------------------------
# quality_filter
# ---------------------------------------------------------------------------

class TestQualityFilter:
    def test_values_above_threshold_become_nan(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        flags = np.array([1, 2, 4, 5])
        out = quality_filter(data, flags, min_quality=3)
        assert not np.isnan(out[0])
        assert not np.isnan(out[1])
        assert np.isnan(out[2])
        assert np.isnan(out[3])

    def test_all_pass_no_nans(self):
        data = np.array([1.0, 2.0, 3.0])
        flags = np.array([1, 1, 1])
        out = quality_filter(data, flags, min_quality=3)
        assert not np.isnan(out).any()

    def test_all_fail_all_nans(self):
        data = np.array([1.0, 2.0, 3.0])
        flags = np.array([5, 6, 7])
        out = quality_filter(data, flags, min_quality=3)
        assert np.isnan(out).all()

    def test_does_not_modify_input(self):
        data = np.array([1.0, 2.0, 3.0])
        flags = np.array([1, 5, 1])
        original = data.copy()
        quality_filter(data, flags)
        np.testing.assert_array_equal(data, original)

    def test_output_is_float(self):
        data = np.array([1, 2, 3], dtype=int)
        flags = np.array([1, 1, 1])
        out = quality_filter(data, flags)
        assert out.dtype == float


# ---------------------------------------------------------------------------
# rolling_sigma_filter
# ---------------------------------------------------------------------------

class TestRollingSigmaFilter:
    def _make_df(self, n=200, spike_idx=100, spike_val=50.0):
        rng = np.random.default_rng(42)
        ts = pl.datetime_range(
            pl.datetime(2024, 1, 1, 0, 0, 0),
            pl.datetime(2024, 1, 1, 0, 0, 0) + pl.duration(seconds=n - 1),
            interval="1s",
            eager=True,
        )
        vals = rng.normal(0, 1, n)
        if spike_idx is not None:
            vals[spike_idx] = spike_val
        return pl.DataFrame({"TIMESTAMP": ts, "Uz": vals})

    def test_returns_dataframe(self):
        df = self._make_df()
        out = rolling_sigma_filter(df)
        assert isinstance(out, pl.DataFrame)

    def test_output_has_filtered_column(self):
        df = self._make_df()
        out = rolling_sigma_filter(df, value_col="Uz", output_col="Uz_filtered")
        assert "Uz_filtered" in out.columns

    def test_spike_is_nulled(self):
        df = self._make_df(spike_idx=100, spike_val=50.0)
        out = rolling_sigma_filter(df, value_col="Uz", period="10s", sigma=3.0)
        filtered = out["Uz_filtered"]
        # The large spike should be null after filtering
        assert filtered[100] is None

    def test_clean_values_preserved(self):
        df = self._make_df(spike_idx=None)
        out = rolling_sigma_filter(df, value_col="Uz", period="10s", sigma=5.0)
        # With very high sigma, very few values should be nulled
        null_count = out["Uz_filtered"].null_count()
        assert null_count < 10

    def test_keep_stats_false_drops_columns(self):
        df = self._make_df()
        out = rolling_sigma_filter(df, value_col="Uz", keep_stats=False)
        assert "Uz_roll_mean" not in out.columns
        assert "Uz_roll_std" not in out.columns

    def test_keep_stats_true_keeps_columns(self):
        df = self._make_df()
        out = rolling_sigma_filter(df, value_col="Uz", keep_stats=True)
        assert "Uz_roll_mean" in out.columns
        assert "Uz_roll_std" in out.columns

    def test_custom_output_col_name(self):
        df = self._make_df()
        out = rolling_sigma_filter(df, value_col="Uz", output_col="my_col")
        assert "my_col" in out.columns

    def test_default_output_col_name(self):
        df = self._make_df()
        out = rolling_sigma_filter(df, value_col="Uz")
        assert "Uz_filtered" in out.columns

    def test_row_count_unchanged(self):
        df = self._make_df()
        out = rolling_sigma_filter(df)
        assert len(out) == len(df)
