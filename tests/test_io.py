"""
Tests for eccospectra.io — TOA5 file reading and multi-file compilation.

Uses synthetic in-memory TOA5 files written to a temporary directory so no
real instrument data is required.
"""

import textwrap
import numpy as np
import polars as pl
import pytest
from pathlib import Path
from datetime import datetime, timedelta

from eccospectra.io import read_toa5, scan_toa5_directory, compile_toa5


# ---------------------------------------------------------------------------
# Helpers for generating synthetic TOA5 content
# ---------------------------------------------------------------------------

def _toa5_header(station_id="TestSite", logger="CR3000"):
    return (
        f'"TOA5","{station_id}","{logger}","1234","OS32","TestProg"\n'
        '"TIMESTAMP","RECORD","Ux","Uy","Uz","T_SONIC","CO2_density","H2O_density"\n'
        '"TS","","m/s","m/s","m/s","C","mg/m^3","g/m^3"\n'
        '"","","Smp","Smp","Smp","Smp","Smp","Smp"\n'
    )


def _toa5_rows(start: datetime, n: int, fs: float = 20.0):
    """Generate *n* rows of synthetic data starting at *start*."""
    rng = np.random.default_rng(99)
    dt_us = int(1e6 / fs)
    lines = []
    ts = start
    for i in range(n):
        u  = rng.normal(5.0, 0.3)
        v  = rng.normal(0.0, 0.2)
        w  = rng.normal(0.0, 0.1)
        T  = rng.normal(20.0, 0.5)
        c  = rng.normal(700.0, 5.0)
        q  = rng.normal(10.0, 0.5)
        lines.append(
            f'"{ts.strftime("%Y-%m-%d %H:%M:%S")}",{i},{u:.4f},{v:.4f},'
            f'{w:.4f},{T:.4f},{c:.4f},{q:.4f}'
        )
        ts += timedelta(microseconds=dt_us)
    return "\n".join(lines) + "\n"


def _write_toa5(path: Path, start: datetime, n: int, fs: float = 20.0):
    content = _toa5_header() + _toa5_rows(start, n, fs)
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# read_toa5
# ---------------------------------------------------------------------------

class TestReadToa5:
    @pytest.fixture
    def toa5_file(self, tmp_path):
        fp = tmp_path / "TOA5_test.dat"
        _write_toa5(fp, start=datetime(2023, 6, 10, 0, 0, 0), n=200)
        return fp

    def test_returns_dataframe_and_metadata(self, toa5_file):
        df, meta = read_toa5(toa5_file)
        assert isinstance(df, pl.DataFrame)
        assert isinstance(meta, dict)

    def test_expected_columns_present(self, toa5_file):
        df, _ = read_toa5(toa5_file)
        for col in ('TIMESTAMP', 'Ux', 'Uy', 'Uz', 'T_SONIC',
                    'CO2_density', 'H2O_density'):
            assert col in df.columns

    def test_row_count_matches(self, toa5_file):
        df, _ = read_toa5(toa5_file)
        assert len(df) == 200

    def test_timestamp_parsed_to_datetime(self, toa5_file):
        df, _ = read_toa5(toa5_file)
        assert df['TIMESTAMP'].dtype == pl.Datetime

    def test_timestamp_not_parsed_when_disabled(self, toa5_file):
        df, _ = read_toa5(toa5_file, parse_dates=False)
        assert df['TIMESTAMP'].dtype == pl.String

    def test_metadata_has_station_id(self, toa5_file):
        _, meta = read_toa5(toa5_file)
        assert meta['station_id'] == 'TestSite'

    def test_metadata_has_units(self, toa5_file):
        _, meta = read_toa5(toa5_file)
        assert 'units' in meta
        assert 'Ux' in meta['units']

    def test_numeric_columns_are_float64(self, toa5_file):
        df, _ = read_toa5(toa5_file)
        for col in ('Ux', 'Uy', 'Uz', 'T_SONIC'):
            assert df[col].dtype == pl.Float64

    def test_diagnostic_screening_with_no_diag_column(self, toa5_file):
        """drop_diagnostics=True should not raise when diag columns are absent."""
        df, _ = read_toa5(toa5_file, drop_diagnostics=True)
        assert isinstance(df, pl.DataFrame)

    def test_file_with_diagnostics(self, tmp_path):
        """Rows with diag_sonic > 0 should null-out wind columns."""
        header = (
            '"TOA5","Site","CR3000","1","OS","Prog"\n'
            '"TIMESTAMP","RECORD","Ux","Uy","Uz","T_SONIC","CO2_density","H2O_density","diag_sonic"\n'
            '"TS","","m/s","m/s","m/s","C","mg/m^3","g/m^3",""\n'
            '"","","Smp","Smp","Smp","Smp","Smp","Smp","Smp"\n'
        )
        rows = (
            '"2023-06-10 00:00:00",1,5.0,0.1,0.0,20.0,700.0,10.0,0\n'
            '"2023-06-10 00:00:01",2,99.0,99.0,99.0,99.0,9999.0,99.0,1\n'
            '"2023-06-10 00:00:02",3,5.1,0.2,0.1,20.1,701.0,10.1,0\n'
        )
        fp = tmp_path / "diag_test.dat"
        fp.write_text(header + rows)
        df, _ = read_toa5(fp, drop_diagnostics=True, max_diag_value=0.0)
        # Row with diag=1 should have null Ux
        assert df.filter(pl.col('RECORD') == 2)['Ux'][0] is None

    def test_path_accepts_string(self, toa5_file):
        df, _ = read_toa5(str(toa5_file))
        assert len(df) > 0


# ---------------------------------------------------------------------------
# scan_toa5_directory
# ---------------------------------------------------------------------------

class TestScanToa5Directory:
    @pytest.fixture
    def data_dir(self, tmp_path):
        starts = [
            datetime(2023, 6, 10, 0, 0),
            datetime(2023, 6, 10, 1, 0),
            datetime(2023, 6, 10, 2, 0),
        ]
        for i, start in enumerate(starts):
            _write_toa5(tmp_path / f"TOA5_site_{i:02d}.dat", start=start, n=100)
        return tmp_path

    def test_returns_list(self, data_dir):
        info = scan_toa5_directory(data_dir)
        assert isinstance(info, list)

    def test_finds_correct_number_of_files(self, data_dir):
        info = scan_toa5_directory(data_dir)
        assert len(info) == 3

    def test_info_dict_has_required_keys(self, data_dir):
        info = scan_toa5_directory(data_dir)
        for entry in info:
            for key in ('path', 'filename', 'size_mb', 'n_records', 'metadata'):
                assert key in entry

    def test_sorted_by_timestamp(self, data_dir):
        info = scan_toa5_directory(data_dir)
        timestamps = [e['first_timestamp'] for e in info]
        assert timestamps == sorted(timestamps)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        info = scan_toa5_directory(tmp_path)
        assert info == []

    def test_pattern_filtering(self, tmp_path):
        _write_toa5(tmp_path / "TOA5_site_A.dat", datetime(2023, 1, 1), n=50)
        _write_toa5(tmp_path / "OTHER_file.dat", datetime(2023, 1, 2), n=50)
        info = scan_toa5_directory(tmp_path, pattern="TOA5_*.dat")
        assert len(info) == 1
        assert info[0]['filename'] == "TOA5_site_A.dat"


# ---------------------------------------------------------------------------
# compile_toa5
# ---------------------------------------------------------------------------

class TestCompileToa5:
    @pytest.fixture
    def multi_file_dir(self, tmp_path):
        """Two sequential non-overlapping files at 1 Hz (unique second-precision timestamps)."""
        _write_toa5(tmp_path / "TOA5_p1.dat",
                    start=datetime(2023, 6, 10, 0, 0, 0), n=120, fs=1.0)
        _write_toa5(tmp_path / "TOA5_p2.dat",
                    start=datetime(2023, 6, 10, 0, 3, 0), n=120, fs=1.0)
        return tmp_path

    def test_returns_dataframe_and_metadata(self, multi_file_dir):
        df, meta = compile_toa5(multi_file_dir)
        assert isinstance(df, pl.DataFrame)
        assert isinstance(meta, dict)

    def test_combined_length_sensible(self, multi_file_dir):
        df, _ = compile_toa5(multi_file_dir)
        # Two 120-row non-overlapping files → at least 120 unique rows
        assert len(df) >= 120

    def test_timestamps_sorted(self, multi_file_dir):
        df, _ = compile_toa5(multi_file_dir)
        ts = df['TIMESTAMP'].to_list()
        assert ts == sorted(ts)

    def test_no_duplicate_timestamps(self, multi_file_dir):
        df, _ = compile_toa5(multi_file_dir)
        assert df['TIMESTAMP'].n_unique() == len(df)

    def test_date_range_filter(self, multi_file_dir):
        t_start = datetime(2023, 6, 10, 0, 1, 0)
        t_end   = datetime(2023, 6, 10, 0, 2, 0)
        df, _ = compile_toa5(multi_file_dir, start_date=t_start, end_date=t_end)
        if len(df) > 0:
            assert df['TIMESTAMP'].min() >= t_start
            assert df['TIMESTAMP'].max() <= t_end

    def test_metadata_has_file_count(self, multi_file_dir):
        _, meta = compile_toa5(multi_file_dir)
        assert 'n_files' in meta
        assert meta['n_files'] == 2
