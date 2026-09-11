"""Numerical and workflow contracts for the imported spectral diagnostics."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.signal import detrend

from TaylorSwift import ec_spectral as ecs

matplotlib.use("Agg")


@pytest.mark.parametrize("n", [511, 512])
@pytest.mark.parametrize("taper", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_covariance_and_ogive(n, taper, sign):
    rng = np.random.default_rng(44)
    w = rng.normal(size=n)
    c = sign * w + rng.normal(size=n) * 0.1
    co = ecs.cospectrum(w, c, 20, taper=taper, nbins=10 * n)
    expected = np.mean(detrend(w) * detrend(c))
    assert_allclose(np.sum(co["Co"]) * 20 / n, expected)
    assert_allclose(co["ogive"][0], 1)
    assert_allclose(co["f_bin"][-1], co["f"][-1])


@pytest.mark.parametrize("lag", [-7, 0, 6])
def test_lag_sign_and_alignment(lag):
    w = np.random.default_rng(5).normal(size=1000)
    c = -np.roll(w, lag)
    c[400] = np.nan
    found, _, r = ecs.find_lag(w, c, 20, expected_s=lag / 20)
    assert found == lag
    assert np.nanmax(np.abs(r)) <= 1 + 1e-14
    aligned = ecs.shift(c, found)
    assert_allclose(aligned[20:300], -w[20:300])


@pytest.mark.parametrize(
    "kwargs",
    [{"fs": 0}, {"d": 4}, {"flow_rate": -1}, {"scalar": "bad"}, {"detrend": "bad"}],
)
def test_invalid_system(kwargs):
    with pytest.raises(ValueError):
        ecs.ECSystem(**kwargs)


def test_invalid_signals_and_fit():
    for w, c in [(np.ones(20), np.ones(20)), ([np.nan] * 20, [1] * 20), ([1, 2], [1])]:
        with pytest.raises(ValueError):
            ecs.cospectrum(w, c, 20)
        with pytest.raises(ValueError):
            ecs.find_lag(w, c, 20)
    with pytest.raises(ValueError):
        ecs.fit_cospectrum([1, 2, 3], [0, 0, 0], 20, 1800)
    with pytest.raises(ValueError):
        ecs.shift([1, 2, 3], 3)


@pytest.mark.parametrize("fix_m", [True, False])
def test_fit_recovers_known_model(fix_m):
    f = np.geomspace(0.002, 2, 100)
    y = ecs.massman_cospectrum(f, 1.3, 0.1, 0.6)
    fit = ecs.fit_cospectrum(f, y, 20, 1800, fix_m=fix_m)
    assert_allclose(
        [fit[k] for k in ("A0", "fx", "mu", "m")], [1.3, 0.1, 0.6, 0.75], rtol=1e-4
    )
    assert fit["r2"] > 0.999999
    assert_allclose(ecs.inertial_slope(f, f ** (-4 / 3), 0.01, 1), -4 / 3)


@pytest.mark.parametrize("flow", [1, 20])
def test_closed_path_transfer_and_time_constants(flow):
    sysc = ecs.ECSystem(
        closed_path=True,
        flow_rate=flow,
        tau_scalar=0.1,
        tau_sonic=0.02,
        lateral_sep=0.1,
    )
    f = np.geomspace(0.001, 10, 100)
    comps = ecs.transfer_function_components(f, sysc, 3)
    assert any("laminar" in k if flow == 1 else "turbulent" in k for k in comps)
    assert_allclose(
        ecs.system_transfer_function(f, sysc, 3), np.prod(list(comps.values()), axis=0)
    )
    for response in comps.values():
        assert np.all((response >= 0) & (response <= 1.000001))
    for method in ("massman", "halfpower"):
        assert ecs.equivalent_time_constants(sysc, 3, method)["tau_e"] > 0


def test_correction_limits_and_grid():
    sysc = ecs.ECSystem(sonic_path=0, scalar_path=0)
    assert_allclose(
        ecs.correction_factor_integral(sysc, 3, include_lowpass=False, fx=0.1), 1
    )
    assert np.isfinite(ecs.correction_uncertainty(0.1, 600, 0))
    assert_allclose(ecs.correction_factor_analytical(1 / (2 * np.pi), 10, 0.2), 1.32)
    for grid in ([0, 1], [2, 1], [1, np.nan]):
        with pytest.raises(ValueError):
            ecs.correction_factor_integral(sysc, 3, n=grid, fx=0.1)


@pytest.mark.parametrize("method", ["analytical", "integral"])
def test_flux_table_preserves_input_and_flags_bad_winds(method):
    df = pd.DataFrame(
        {"u": [0.1, 3, 0, -1, np.nan, np.inf], "cov_wCO2": [-2.0] * 6},
        index=list("abcdef"),
    )
    original = df.copy(deep=True)
    out = ecs.correct_flux_table(
        df, ecs.ECSystem(), 0.1, cov_cols=("cov_wCO2",), method=method
    )
    pd.testing.assert_frame_equal(df, original)
    assert out.index.equals(df.index)
    assert out.F_flag.iloc[2:].all()
    assert out.F.iloc[2:].isna().all()
    assert_allclose(out.cov_wCO2_corr.iloc[:2], -2 * out.F.iloc[:2])
    if method == "integral":
        assert_allclose(
            out.F.iloc[0],
            ecs.correction_factor_integral(ecs.ECSystem(), 0.1, eta_x=0.1),
        )
    assert ecs.correct_flux_table(
        df.iloc[:0], ecs.ECSystem(), 0.1, cov_cols=("cov_wCO2",), method=method
    ).empty


def test_preprocessing(tmp_path):
    path = tmp_path / "sample.dat"
    path.write_text(
        '"TOA5","station"\n"TIMESTAMP","Ux"\n"TS","m/s"\n"",""\n"2026-01-01 00:00:00",2\n"2026-01-01 00:00:01",NAN\n'
    )
    df = ecs.load_toa5(path)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert np.isnan(df.Ux.iloc[1])
    u, v, w = ecs.double_rotation([3, 4, 5], [1, 2, 3], [0.1, 0.2, 0.3])
    assert_allclose([v.mean(), w.mean()], 0, atol=1e-14)
    x = np.ones(100)
    x[50] = 100
    assert_allclose(ecs.despike(x, window=20), 1)


def test_demo_and_notebook(tmp_path, monkeypatch):
    monkeypatch.setattr(ecs._pyplot(), "show", lambda: None)
    result = ecs.demo(str(tmp_path), ecs.ECSystem(avg_period=180))
    assert result["F_int"] >= 1
    assert (tmp_path / "ec_spectral_demo.png").stat().st_size > 1000
    notebook = (
        Path(__file__).resolve().parents[1]
        / "examples/04_ec_spectral_diagnostics.ipynb"
    )
    scope = {"__name__": "__main__"}
    for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(notebook), "exec"), scope)
