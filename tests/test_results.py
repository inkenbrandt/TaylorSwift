"""
Tests for TaylorSwift.results — tabular export of SpectralResult lists.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from TaylorSwift.results import (
    SpectralResult,
    results_to_csv,
    results_to_dataframe,
    results_to_parquet,
    spectra_to_dataframe,
)


def _make_result(i: int, n_bins: int = 8) -> SpectralResult:
    start = datetime(2024, 6, 1, 0, 0) + timedelta(minutes=30 * i)
    freq = np.logspace(-2, 1, n_bins)
    cosp = 0.5 * freq / (1 + 10 * freq) ** (7 / 3)
    return SpectralResult(
        timestamp_start=start,
        timestamp_end=start + timedelta(minutes=30),
        u_mean=5.0 + i,
        wind_dir=180.0,
        T_mean=20.0,
        ustar=0.3,
        L=-50.0,
        zL=-0.05,
        H=20.0 + i,
        cov_wT=0.017,
        cov_wu=-0.09,
        cov_wCO2=-0.05,
        cov_wH2O=0.002,
        freq=freq,
        freq_nd=freq * 3.0 / 5.0,
        cosp_wT=cosp,
        ncosp_wT=cosp / 0.017,
        qc_flags={"cf_wT": 1.05 + 0.01 * i, "ustar_filter": False},
    )


class TestResultsToDataframe:
    def test_one_row_per_interval(self):
        results = [_make_result(i) for i in range(3)]
        df = results_to_dataframe(results)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3

    def test_scalar_columns_present(self):
        df = results_to_dataframe([_make_result(0)])
        for col in (
            "timestamp_start",
            "timestamp_end",
            "u_mean",
            "ustar",
            "L",
            "zL",
            "H",
            "cov_wT",
            "cov_wu",
            "cov_wCO2",
            "cov_wH2O",
        ):
            assert col in df.columns

    def test_values_roundtrip(self):
        results = [_make_result(i) for i in range(2)]
        df = results_to_dataframe(results)
        assert df["u_mean"].to_list() == [5.0, 6.0]
        assert df["H"].to_list() == [20.0, 21.0]
        assert df["timestamp_start"][0] == datetime(2024, 6, 1, 0, 0)

    def test_qc_flags_flattened(self):
        df = results_to_dataframe([_make_result(0)])
        assert "cf_wT" in df.columns
        assert "ustar_filter" in df.columns
        assert df["cf_wT"][0] == pytest.approx(1.05)

    def test_missing_qc_flag_is_null(self):
        res_with = _make_result(0)
        res_without = _make_result(1)
        res_without.qc_flags = {}
        df = results_to_dataframe([res_with, res_without])
        assert df["cf_wT"][0] == pytest.approx(1.05)
        assert df["cf_wT"][1] is None

    def test_array_qc_values_excluded(self):
        res = _make_result(0)
        res.qc_flags["some_array"] = np.arange(5)
        df = results_to_dataframe([res])
        assert "some_array" not in df.columns

    def test_include_qc_false(self):
        df = results_to_dataframe([_make_result(0)], include_qc=False)
        assert "cf_wT" not in df.columns
        assert "u_mean" in df.columns

    def test_empty_results(self):
        df = results_to_dataframe([])
        assert len(df) == 0
        assert "u_mean" in df.columns

    def test_nan_preserved_as_float(self):
        res = SpectralResult(timestamp_start=datetime(2024, 6, 1))
        df = results_to_dataframe([res])
        assert df["u_mean"].dtype == pl.Float64
        assert np.isnan(df["u_mean"][0])


class TestSpectraToDataframe:
    def test_long_format_row_count(self):
        results = [_make_result(0, n_bins=8), _make_result(1, n_bins=6)]
        df = spectra_to_dataframe(results)
        assert len(df) == 14  # 8 + 6

    def test_columns(self):
        df = spectra_to_dataframe([_make_result(0)])
        for col in ("timestamp_start", "freq", "freq_nd", "cosp_wT", "ncosp_wT"):
            assert col in df.columns

    def test_empty_freq_interval_skipped(self):
        res = SpectralResult(timestamp_start=datetime(2024, 6, 1))
        df = spectra_to_dataframe([res, _make_result(0, n_bins=5)])
        assert len(df) == 5

    def test_missing_arrays_become_nan(self):
        # _make_result fills cosp_wT but not cosp_wu — mismatched arrays
        # must appear as NaN columns rather than raising.
        df = spectra_to_dataframe([_make_result(0)])
        assert df["cosp_wu"].is_nan().all()
        assert not df["cosp_wT"].is_nan().any()

    def test_empty_results(self):
        df = spectra_to_dataframe([])
        assert len(df) == 0
        assert "freq" in df.columns


class TestWriters:
    def test_csv_roundtrip(self, tmp_path):
        results = [_make_result(i) for i in range(2)]
        path = tmp_path / "fluxes.csv"
        df = results_to_csv(results, path)
        assert path.exists()
        back = pl.read_csv(path)
        assert len(back) == len(df) == 2
        assert back["u_mean"].to_list() == [5.0, 6.0]

    def test_parquet_roundtrip(self, tmp_path):
        results = [_make_result(i) for i in range(2)]
        path = tmp_path / "fluxes.parquet"
        df = results_to_parquet(results, path)
        back = pl.read_parquet(path)
        assert back.equals(df)

    def test_parquet_with_spectra(self, tmp_path):
        results = [_make_result(0, n_bins=4)]
        path = tmp_path / "fluxes.parquet"
        spectra_path = tmp_path / "spectra.parquet"
        results_to_parquet(results, path, spectra_path=spectra_path)
        assert spectra_path.exists()
        spectra = pl.read_parquet(spectra_path)
        assert len(spectra) == 4
