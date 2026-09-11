"""Matched H2O averaging-period comparison; run from a checkout with Python.

Outputs are written beneath examples/outputs/wellington_averaging_comparison.
No source data are changed. See the generated notebook/report for interpretation.
"""

import json
import os
import sys
from pathlib import Path

# Avoid excessive BLAS threading for thousands of small correlations/fits.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from TaylorSwift import ec_spectral as ecs  # noqa: E402 - allow checkout execution

TAPER = "--no-taper" not in sys.argv
OUTPUT = (
    ROOT
    / "examples/outputs"
    / ("wellington_averaging_comparison" if TAPER else "wellington_averaging_untapered")
)
DATA = ROOT / "data/output/Wellington_filtered.parquet"
FS = 10
CHANNELS = ["Ux", "Uy", "Uz", "H2O_density"]
MIN_COVERAGE, MAX_GAP_S = 0.85, 0.2


def longest_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int(runs.max()) if len(runs) else 0


def covariance(w, c):
    return float(np.mean((w - w.mean()) * (c - c.mean())))


def analyze(values, minutes, start, parent):
    u, v, w = ecs.double_rotation(*values[:, :3].T)
    c = values[:, 3]
    lag, lags, r = ecs.find_lag(w, c, FS, max_lag_s=0.5)
    # Trim exposed samples, retaining regular spacing.
    keep = (
        slice(None, -lag) if lag > 0 else slice(-lag, None) if lag < 0 else slice(None)
    )
    c = ecs.shift(c, lag)[keep]
    w = w[keep]
    wind = float(np.hypot(u[keep], v[keep]).mean())
    cov = covariance(w, c)
    rho = float(np.corrcoef(w, c)[0, 1])
    # Fixed five-minute chunks for all durations. Last chunk can be <0.5 s shorter.
    pieces = [(w[i : i + 3000], c[i : i + 3000]) for i in range(0, len(w), 3000)]
    short_cov = np.average(
        [covariance(x, y) for x, y in pieces], weights=[len(x) for x, _ in pieces]
    )
    stationarity = abs(cov - short_cov) / max(abs(cov), 1e-15)
    # Six equal pieces provides an alternative consistent with common stationarity tests.
    six = list(zip(np.array_split(w, 6), np.array_split(c, 6), strict=True))
    six_cov = np.average(
        [covariance(x, y) for x, y in six], weights=[len(x) for x, _ in six]
    )
    six_stationarity = abs(cov - six_cov) / max(abs(cov), 1e-15)
    t = np.linspace(-1, 1, len(w))
    wd = w - w.mean() - t * np.dot(t, w - w.mean()) / np.dot(t, t)
    cd = c - c.mean() - t * np.dot(t, c - c.mean()) / np.dot(t, t)
    linear_cov = float(np.mean(wd * cd))
    row = dict(
        parent=parent,
        start=start,
        minutes=minutes,
        wind=wind,
        covariance=cov,
        rho=rho,
        lag_s=lag / FS,
        lag_boundary=abs(lag) == 5,
        stationarity=stationarity,
        stationarity_six=six_stationarity,
        trend_sensitivity=abs(linear_cov - cov) / max(abs(cov), 1e-15),
        linear_covariance=linear_cov,
        spectral_ok=False,
        fit_ok=False,
    )
    try:
        co = ecs.cospectrum(w, c, FS, detrend="block", taper=TAPER, nbins=60)
        untapered = (
            ecs.cospectrum(w, c, FS, detrend="block", taper=False, nbins=60)
            if TAPER
            else co
        )
    except ValueError as exc:
        row["spectral_error"] = str(exc)
        return row, None
    np.testing.assert_allclose(
        co["Co"].sum() * FS / co["N"], cov, rtol=1e-9, atol=1e-12
    )
    row["spectral_ok"] = True
    # Lowest three harmonics: endpoint-shape diagnostic, not a universal QC standard.
    # Absolute bin contributions avoid hiding cancellation between slow modes.
    for suffix, spectrum in [("", co), ("_untapered", untapered)]:
        normalized = spectrum["Co"] * FS / spectrum["N"] / cov
        row["slow3_abs" + suffix] = float(np.abs(normalized[:3]).sum())
        row["slow3_signed" + suffix] = float(normalized[:3].sum())
        row["ogive_overshoot" + suffix] = float(
            max(0, spectrum["ogive"].max() - 1, -spectrum["ogive"].min())
        )
        row["slower30_abs" + suffix] = float(
            np.abs(normalized[spectrum["f"] < 1 / 1800]).sum()
        )
    system = ecs.ECSystem(
        fs=FS,
        avg_period=len(w) / FS,
        z_ref=3.52,
        d=0.2,
        sonic_path=0.10,
        scalar_path=0.154,
        tau_scalar=0.1,
        scalar="h2o",
    )
    # Common physical fit band avoids crediting a longer record for a changed fit range.
    fits = {}
    for name, fmin in [("common", 2 / 1800), ("native", 2 / system.avg_period)]:
        try:
            fit = ecs.fit_cospectrum(
                co["f_bin"], co["fCo_bin"], FS, system.avg_period, fmin=fmin, fmax=1.0
            )
            fits[name] = fit
            row.update({f"{name}_{key}": fit[key] for key in ["r2", "fx", "mu"]})
            row[f"{name}_mu_boundary"] = fit["mu"] <= 0.11 or fit["mu"] >= 2.99
        except (ValueError, RuntimeError) as exc:
            row[f"{name}_fit_error"] = str(exc)
    if "common" in fits:
        fit = fits["common"]
        row["fit_ok"] = True
        row["F_integral"] = ecs.correction_factor_integral(
            system, wind, fx=fit["fx"], mu=fit["mu"]
        )
        taus = ecs.equivalent_time_constants(system, wind)
        row["F_analytical"] = float(
            ecs.correction_factor_analytical(fit["fx"], taus["tau_b"], taus["tau_e"])
        )
    arrays = dict(
        f=co["f"], ogive=co["ogive"], f_bin=co["f_bin"], fCo_bin=co["fCo_bin"]
    )
    return row, arrays


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(DATA)
    j = parquet.schema_arrow.names.index("TIMESTAMP")
    bounds = [
        parquet.metadata.row_group(i).column(j).statistics
        for i in range(parquet.num_row_groups)
    ]
    first = pd.Timestamp(min(s.min for s in bounds)).floor("D")
    last = pd.Timestamp(max(s.max for s in bounds)).floor("D")
    rows, coverage, rejected = [], [], []
    examples = {}
    for day in pd.date_range(first, last, freq="D"):
        tomorrow = day + pd.Timedelta(days=1)
        raw = (
            pq.read_table(
                DATA,
                columns=["TIMESTAMP", *CHANNELS],
                filters=[
                    ("TIMESTAMP", ">=", day.to_pydatetime()),
                    ("TIMESTAMP", "<", tomorrow.to_pydatetime()),
                ],
            )
            .to_pandas(ignore_metadata=True)
            .set_index("TIMESTAMP")
            .sort_index()
        )
        if raw.index.has_duplicates:
            raise ValueError(f"Duplicate timestamps on {day}")
        if np.any(
            (raw.index - day) % pd.Timedelta(milliseconds=100) != pd.Timedelta(0)
        ):
            raise ValueError(f"Off-grid timestamps on {day}")
        raw = raw.replace([-999, -9999, np.inf, -np.inf], np.nan)
        for parent in pd.date_range(day, tomorrow, freq="2h", inclusive="left"):
            halves = []
            for start in pd.date_range(parent, periods=4, freq="30min"):
                grid = pd.date_range(start, periods=18000, freq="100ms")
                values = raw.reindex(grid).to_numpy(dtype=float, copy=True)
                accepted = True
                for k, channel in enumerate(CHANNELS):
                    valid = np.isfinite(values[:, k])
                    covg, gap = valid.mean(), longest_run(~valid) / FS
                    ok = covg >= MIN_COVERAGE and gap <= MAX_GAP_S + 1e-9
                    coverage.append(
                        dict(
                            start=start,
                            channel=channel,
                            coverage=covg,
                            gap_s=gap,
                            accepted=ok,
                        )
                    )
                    accepted &= ok
                    if ok:
                        values[:, k] = np.interp(
                            np.arange(len(values)),
                            np.flatnonzero(valid),
                            values[valid, k],
                        )
                halves.append(values if accepted else None)
            if any(h is None for h in halves):
                rejected.append(
                    dict(
                        parent=parent, reason="one_or_more_half_hours_failed_gap_policy"
                    )
                )
                continue
            values = np.concatenate(halves)
            for minutes in [30, 60, 120]:
                size = minutes * 60 * FS
                for offset in range(0, len(values), size):
                    start = parent + pd.Timedelta(seconds=offset / FS)
                    row, arr = analyze(
                        values[offset : offset + size], minutes, start, parent
                    )
                    rows.append(row)
                    # Save all seven spectra for the user's illustrated two-hour window.
                    if parent == pd.Timestamp("2024-06-27 14:00") and arr is not None:
                        examples[f"{minutes}_{offset // size}"] = arr
        pd.DataFrame(rows).to_csv(OUTPUT / "interval_metrics.csv", index=False)
        print(
            f"{day.date()}: {len(rows)} intervals analyzed, {len(rejected)} parent windows rejected",
            flush=True,
        )
    df = pd.DataFrame(rows)
    pd.DataFrame(coverage).to_csv(OUTPUT / "coverage.csv", index=False)
    pd.DataFrame(rejected, columns=["parent", "reason"]).to_csv(
        OUTPUT / "rejected_windows.csv", index=False
    )
    # Require all seven spectra per parent for an exactly matched spectral comparison.
    good = df.groupby("parent")["spectral_ok"].all()
    df["matched"] = df.parent.isin(good[good].index)
    # Common subset: each of the seven intervals has |r| >= 0.1.
    strong = df.groupby("parent").rho.apply(lambda x: (x.abs() >= 0.1).all())
    df["strong_parent"] = df.parent.isin(strong[strong].index) & df.matched
    df["steady30"] = df.stationarity <= 0.30
    df["steady_six30"] = df.stationarity_six <= 0.30
    df["slow3_small"] = df.slow3_abs <= 0.10
    df.to_csv(OUTPUT / "interval_metrics.csv", index=False)
    numeric = [
        "stationarity",
        "stationarity_six",
        "steady30",
        "steady_six30",
        "trend_sensitivity",
        "slow3_abs",
        "slow3_abs_untapered",
        "ogive_overshoot",
        "common_r2",
        "native_r2",
        "common_mu_boundary",
        "native_mu_boundary",
        "F_integral",
        "F_analytical",
        "lag_boundary",
    ]
    summaries = []
    for name, subset in [
        ("all_matched", df[df.matched]),
        ("strong_matched", df[df.strong_parent]),
    ]:
        for minutes, group in subset.groupby("minutes"):
            summary = dict(
                subset=name,
                minutes=minutes,
                intervals=len(group),
                parents=group.parent.nunique(),
            )
            for col in numeric:
                summary[col + "_median"] = group[col].median()
                summary[col + "_mean"] = group[col].mean()
            summaries.append(summary)
    pd.DataFrame(summaries).to_csv(OUTPUT / "summary.csv", index=False)
    settings = dict(
        data=str(DATA),
        first=str(first),
        last=str(last),
        fs=FS,
        intervals_minutes=[30, 60, 120],
        channel="H2O_density",
        units="g/m2/s",
        min_coverage=MIN_COVERAGE,
        max_gap_s=MAX_GAP_S,
        despike=False,
        taper=TAPER,
        common_fit_band_hz=[2 / 1800, 1],
        strong_rho=0.1,
        stationarity_screen=0.30,
        slow3_screen=0.10,
        valid_parents=int(df.parent.nunique()),
        matched_parents=int(good.sum()),
        strong_parents=int(df[df.strong_parent].parent.nunique()),
        rejected_parents=len(rejected),
    )
    (OUTPUT / "settings.json").write_text(json.dumps(settings, indent=2))
    if examples:
        np.savez_compressed(
            OUTPUT / "example_spectra.npz",
            **{
                f"{name}_{k}": v
                for name, arrays in examples.items()
                for k, v in arrays.items()
            },
        )
    print(
        pd.DataFrame(summaries)[
            [
                "subset",
                "minutes",
                "intervals",
                "steady30_mean",
                "common_r2_median",
                "slow3_abs_median",
                "trend_sensitivity_median",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run()
