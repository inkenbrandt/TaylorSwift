"""Input ownership, structural validation and channel quality contracts."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from TaylorSwift import SiteConfig, process_interval
from TaylorSwift.core import process_file
from TaylorSwift.results import _SPECTRA_FIELDS, results_to_dataframe


def signals(n=1000):
    rng = np.random.default_rng(17)
    return [rng.normal(mean, 0.2, n) for mean in (5, 0, 0, 20, 700, 10)]


@pytest.mark.parametrize("adapter", ["numpy", "pandas", "polars"])
def test_inputs_unchanged_and_readonly_equivalent(adapter):
    arrays = signals()
    for arr in arrays:
        arr[50] = np.nan
    if adapter == "pandas":
        frame = pd.DataFrame(dict(enumerate(arrays)))
        arrays = [frame[i].to_numpy() for i in range(6)]
    elif adapter == "polars":
        frame = pl.DataFrame({str(i): a for i, a in enumerate(arrays)})
        arrays = [frame[str(i)].to_numpy() for i in range(6)]
    before = [a.copy() for a in arrays]
    result = process_interval(*arrays, SiteConfig())
    for arr, original in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(arr, original)
        arr.flags.writeable = False
    readonly = process_interval(*arrays, SiteConfig())
    writable = process_interval(*[a.copy() for a in arrays], SiteConfig())
    for name in ("freq", *_SPECTRA_FIELDS):
        np.testing.assert_array_equal(getattr(result, name), getattr(readonly, name))
        np.testing.assert_array_equal(getattr(result, name), getattr(writable, name))
    assert result.qc_flags["interval_status"] == "ok"


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_minimum_samples(n):
    with pytest.raises(ValueError, match="at least 4"):
        process_interval(*signals(n), SiteConfig())


def test_four_samples_supported():
    result = process_interval(*signals(4), SiteConfig())
    assert result.freq.size > 0


@pytest.mark.parametrize("bad", [np.ones((1000, 1)), np.ones(999), np.array(1)])
def test_bad_shapes(bad):
    arrays = signals()
    arrays[4] = bad
    with pytest.raises(ValueError, match="one-dimensional|equal length"):
        process_interval(*arrays, SiteConfig())


@pytest.mark.parametrize("channel", range(6))
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_samples_are_filled(channel, value):
    arrays = signals()
    arrays[channel][50] = value
    result = process_interval(*arrays, SiteConfig())
    key = ("u", "v", "w", "T", "co2", "h2o")[channel]
    assert result.qc_flags[f"{key}_status"] == "interpolated"
    assert np.isfinite(result.cosp_wCO2).all()
    assert np.isfinite(result.cosp_wT).all()


@pytest.mark.parametrize(
    "channel,suffix,cov",
    [
        (3, "wT", "cov_wT"),
        (4, "wCO2", "cov_wCO2"),
        (5, "wH2O", "cov_wH2O"),
    ],
)
def test_all_missing_scalar_is_explicit(channel, suffix, cov):
    arrays = signals()
    arrays[channel][:] = np.nan
    result = process_interval(*arrays, SiteConfig())
    key = ("T", "co2", "h2o")[channel - 3]
    assert result.qc_flags[f"{key}_status"] == "insufficient_finite_data"
    assert result.qc_flags["interval_status"] == "partial"
    assert np.isnan(getattr(result, cov))
    assert result.freq.size > 0
    for name in _SPECTRA_FIELDS:
        assert getattr(result, name).shape == result.freq.shape
    for prefix in ("cosp_", "ncosp_", "ogive_"):
        assert np.isnan(getattr(result, prefix + suffix)).all()
    assert np.isfinite(result.spec_w).all()
    if channel == 3:
        assert all(np.isnan(getattr(result, n)) for n in ("T_mean", "H", "L", "zL"))
        assert np.isnan(result.spec_T).all()
    exported = results_to_dataframe([result])
    assert exported[f"{key}_status"][0] == "insufficient_finite_data"


@pytest.mark.parametrize("channel", [0, 4])
@pytest.mark.parametrize("count,expected", [(20, "interpolated"), (21, "gap_too_long")])
def test_gap_duration_boundary(channel, count, expected):
    arrays = signals()
    arrays[channel][50 : 50 + count] = np.nan
    result = process_interval(*arrays, SiteConfig(sampling_freq=20))
    key = "u" if channel == 0 else "co2"
    assert result.qc_flags[f"{key}_status"] == expected
    if expected == "gap_too_long":
        assert result.qc_flags["interval_status"] == (
            "invalid_wind" if channel == 0 else "partial"
        )


@pytest.mark.parametrize("index", [0, -1])
@pytest.mark.parametrize(
    "policy,expected", [("reject", "endpoint_gap"), ("nearest", "interpolated")]
)
def test_endpoint_policy(index, policy, expected):
    arrays = signals()
    arrays[4][index] = np.nan
    result = process_interval(*arrays, SiteConfig(endpoint_policy=policy))
    assert result.qc_flags["co2_status"] == expected


def test_nearest_still_rejects_long_endpoint_gap():
    arrays = signals()
    arrays[4][:21] = np.nan
    result = process_interval(*arrays, SiteConfig(endpoint_policy="nearest"))
    assert result.qc_flags["co2_status"] == "gap_too_long"


def test_per_channel_thresholds():
    arrays = signals()
    arrays[0][10:990:10] = np.inf
    cfg = SiteConfig()
    rejected = process_interval(*arrays, cfg)
    assert rejected.freq.size == 0
    assert rejected.qc_flags["too_many_nans"] is True
    cfg.min_finite_fraction["u"] = 0.9
    accepted = process_interval(*arrays, cfg)
    assert accepted.qc_flags["u_status"] == "interpolated"
    arrays[4][10:990:10] = np.inf
    scalar_rejected = process_interval(*arrays, cfg)
    assert scalar_rejected.qc_flags["co2_status"] == "insufficient_finite_data"


@pytest.mark.parametrize("field", ["sampling_freq", "averaging_period"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_time_frequency(field, value):
    with pytest.raises(ValueError, match=field):
        SiteConfig(**{field: value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"z_measurement": 0},
        {"d": 3},
        {"d": np.inf},
        {"z_canopy": np.nan},
        {"z_canopy": -1},
        {"tau_co2": -1},
        {"max_gap_seconds": -1},
        {"max_gap_seconds": np.inf},
        {"endpoint_policy": "fill"},
        {"min_finite_fraction": {}},
        {
            "min_finite_fraction": dict.fromkeys(
                ("u", "v", "w", "T", "co2", "h2o"), np.nan
            )
        },
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SiteConfig(**kwargs)


@pytest.mark.parametrize("bins", [0, -1, np.inf, np.nan, 1.5, True])
def test_invalid_binning(bins):
    with pytest.raises(ValueError, match="bins_per_decade"):
        process_interval(*signals(), SiteConfig(), bins_per_decade=bins)


def test_mutated_config_revalidated():
    cfg = SiteConfig()
    cfg.sampling_freq = 0
    with pytest.raises(ValueError, match="sampling_freq"):
        process_interval(*signals(), cfg)


def test_file_fractional_period_and_short_windows():
    start = datetime(2026, 1, 1, 0, 0, 0, 500000)
    arrays = signals(11)
    frame = pl.DataFrame(
        {
            "TIMESTAMP": [start + timedelta(seconds=i / 20) for i in range(11)],
            **dict(
                zip(
                    ("Ux", "Uy", "Uz", "T_SONIC", "CO2_density", "H2O_density"),
                    arrays,
                    strict=True,
                )
            ),
        }
    )
    cfg = SiteConfig(averaging_period=0.5 / 60)
    results = process_file(frame, cfg)
    assert len(results) == 1  # one complete window, one single-sample tail
    assert results[0].timestamp_start == start
    with pytest.raises(ValueError, match="bins_per_decade"):
        process_file(frame.head(0), cfg, bins_per_decade=0)
    cfg.averaging_period = np.inf
    with pytest.raises(ValueError, match="averaging_period"):
        process_file(frame, cfg)


def test_zero_gap_limit_disables_interpolation():
    arrays = signals()
    arrays[4][50] = np.nan
    result = process_interval(*arrays, SiteConfig(max_gap_seconds=0))
    assert result.qc_flags["co2_status"] == "gap_too_long"
