"""
Tests for TaylorSwift.core.process_file interval slicing.

The per-interval DataFrame filter was replaced with searchsorted slicing in
Phase 1 of the roadmap; these tests pin the interval semantics: half-open
[edge_i, edge_i+1) windows aligned to averaging-period boundaries, with
intervals below 90% completeness skipped.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from TaylorSwift.config import SiteConfig
from TaylorSwift.core import process_file, process_interval

FS = 2.0  # Hz — low rate keeps the tests fast
PERIOD_MIN = 1.0
N_PER_INTERVAL = int(PERIOD_MIN * 60 * FS)  # 120 samples


@pytest.fixture
def config():
    return SiteConfig(
        z_measurement=3.0,
        z_canopy=0.3,
        sampling_freq=FS,
        averaging_period=PERIOD_MIN,
    )


def _make_df(start: datetime, n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    ts = [start + timedelta(seconds=i / FS) for i in range(n)]
    return pl.DataFrame(
        {
            "TIMESTAMP": pl.Series(ts, dtype=pl.Datetime("us")),
            "Ux": 5.0 + rng.normal(0, 0.5, n),
            "Uy": 0.5 + rng.normal(0, 0.3, n),
            "Uz": rng.normal(0, 0.15, n),
            "T_SONIC": 20.0 + rng.normal(0, 0.1, n),
            "CO2_density": 700.0 + rng.normal(0, 5.0, n),
            "H2O_density": 10.0 + rng.normal(0, 0.5, n),
        }
    )


class TestProcessFileSlicing:
    def test_complete_intervals_all_processed(self, config):
        start = datetime(2024, 6, 1, 0, 0, 0)
        df = _make_df(start, 10 * N_PER_INTERVAL)
        results = process_file(df, config)
        assert len(results) == 10
        for i, res in enumerate(results):
            assert res.timestamp_start == start + timedelta(minutes=i)
            assert res.timestamp_end == start + timedelta(minutes=i + 1)

    def test_partial_leading_interval_skipped(self, config):
        # Data starts mid-interval: the first window is only half-covered.
        start = datetime(2024, 6, 1, 0, 0, 30)
        df = _make_df(start, 5 * N_PER_INTERVAL)
        results = process_file(df, config)
        starts = [r.timestamp_start for r in results]
        assert datetime(2024, 6, 1, 0, 0, 0) not in starts
        assert starts[0] == datetime(2024, 6, 1, 0, 1, 0)

    def test_gappy_interval_skipped(self, config):
        start = datetime(2024, 6, 1, 0, 0, 0)
        df = _make_df(start, 5 * N_PER_INTERVAL)
        # Remove 15% of rows inside the third interval only.
        third = (pl.col("TIMESTAMP") >= start + timedelta(minutes=2)) & (
            pl.col("TIMESTAMP") < start + timedelta(minutes=3)
        )
        rng = np.random.default_rng(1)
        drop = pl.Series(rng.random(len(df)) < 0.15)
        df = df.filter(~(third & drop))
        results = process_file(df, config)
        starts = [r.timestamp_start for r in results]
        assert start + timedelta(minutes=2) not in starts
        assert len(results) == 4

    def test_edge_sample_belongs_to_next_interval(self, config):
        """A sample exactly on an edge must count toward the interval it
        starts, not the one it ends (half-open [left, right) windows)."""
        start = datetime(2024, 6, 1, 0, 0, 0)
        # 107 samples inside [00:00, 00:01) — one short of the 90% cutoff —
        # plus one sample exactly at 00:01:00. If the edge sample were
        # wrongly included the interval would reach 108 and be processed.
        df = _make_df(start, 107)
        edge_row = _make_df(datetime(2024, 6, 1, 0, 1, 0), 1, seed=9)
        df = pl.concat([df, edge_row])
        assert process_file(df, config) == []

    def test_results_match_manual_slicing(self, config):
        start = datetime(2024, 6, 1, 0, 0, 0)
        df = _make_df(start, 3 * N_PER_INTERVAL)
        results = process_file(df, config)

        # Recompute the second interval by slicing the frame manually.
        lo, hi = N_PER_INTERVAL, 2 * N_PER_INTERVAL
        sub = df[lo:hi]
        expected = process_interval(
            u_raw=sub["Ux"].to_numpy(),
            v_raw=sub["Uy"].to_numpy(),
            w_raw=sub["Uz"].to_numpy(),
            T_sonic=sub["T_SONIC"].to_numpy(),
            co2=sub["CO2_density"].to_numpy(),
            h2o=sub["H2O_density"].to_numpy(),
            config=config,
        )
        assert results[1].cov_wT == expected.cov_wT
        assert results[1].ustar == expected.ustar
        assert np.array_equal(results[1].freq, expected.freq)

    def test_unsorted_input_and_null_timestamps(self, config):
        start = datetime(2024, 6, 1, 0, 0, 0)
        df = _make_df(start, 2 * N_PER_INTERVAL)
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=3)
        with_null = pl.concat(
            [shuffled, df[:1].with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("TIMESTAMP"))]
        )
        results = process_file(with_null, config)
        assert len(results) == 2
        assert results[0].timestamp_start == start

    def test_empty_frame(self, config):
        df = _make_df(datetime(2024, 6, 1), 5).filter(pl.lit(False))
        assert process_file(df, config) == []


class TestColumnMap:
    def test_renamed_columns_match_canonical(self, config):
        start = datetime(2024, 6, 1, 0, 0, 0)
        df = _make_df(start, 2 * N_PER_INTERVAL)
        renamed = df.rename(
            {
                "TIMESTAMP": "time",
                "Ux": "u",
                "Uy": "v",
                "Uz": "w",
                "T_SONIC": "Ts",
                "CO2_density": "co2",
                "H2O_density": "h2o",
            }
        )
        column_map = {
            "time": "TIMESTAMP",
            "u": "Ux",
            "v": "Uy",
            "w": "Uz",
            "Ts": "T_SONIC",
            "co2": "CO2_density",
            "h2o": "H2O_density",
        }
        expected = process_file(df, config)
        results = process_file(renamed, config, column_map=column_map)
        assert len(results) == len(expected) == 2
        for res, exp in zip(results, expected, strict=True):
            assert res.cov_wT == exp.cov_wT
            assert res.ustar == exp.ustar

    def test_extra_map_entries_ignored(self, config):
        start = datetime(2024, 6, 1, 0, 0, 0)
        df = _make_df(start, N_PER_INTERVAL)
        results = process_file(
            df, config, column_map={"not_a_column": "Ux", "also_missing": "Uy"}
        )
        assert len(results) == 1

    def test_missing_required_column_raises(self, config):
        start = datetime(2024, 6, 1, 0, 0, 0)
        df = _make_df(start, N_PER_INTERVAL).drop("CO2_density")
        with pytest.raises(ValueError, match="CO2_density"):
            process_file(df, config)

    def test_error_message_suggests_column_map(self, config):
        start = datetime(2024, 6, 1, 0, 0, 0)
        df = _make_df(start, N_PER_INTERVAL).rename({"Ux": "u_wind"})
        with pytest.raises(ValueError, match="column_map"):
            process_file(df, config)
