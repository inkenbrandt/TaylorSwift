"""Summarize the saved matched comparison and generate figures and paired checks."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = (
    Path(__file__).resolve().parent
    / "outputs"
    / (
        "wellington_averaging_untapered"
        if "--no-taper" in sys.argv
        else "wellington_averaging_comparison"
    )
)


def run():
    df = pd.read_csv(OUTPUT / "interval_metrics.csv", parse_dates=["parent", "start"])
    d = df[df.matched].copy()
    counts = d.groupby(["parent", "minutes"]).size().unstack()
    assert (
        (counts[30] == 4).all() and (counts[60] == 2).all() and (counts[120] == 1).all()
    )
    assert not d.duplicated(["start", "minutes"]).any()
    assert d.spectral_ok.all()
    summaries, sensitivity, uncertainty = [], [], []
    rng = np.random.default_rng(20260911)
    for label, subset in [("All matched", d), ("Stronger signal", d[d.strong_parent])]:
        for duration, g in subset.groupby("minutes"):
            summaries.append(
                dict(
                    subset=label,
                    minutes=duration,
                    intervals=len(g),
                    parents=g.parent.nunique(),
                    median_r2=g.common_r2.median(),
                    median_native_r2=g.native_r2.median(),
                    steady_percent=100 * g.steady30.mean(),
                    steady_six_percent=100 * g.steady_six30.mean(),
                    median_stationarity_percent=100 * g.stationarity.median(),
                    median_trend_percent=100 * g.trend_sensitivity.median(),
                    median_slow3_percent=100 * g.slow3_abs.median(),
                    median_slow3_untapered_percent=100 * g.slow3_abs_untapered.median(),
                    median_overshoot_percent=100 * g.ogive_overshoot.median(),
                    fit_boundary_percent=100 * g.common_mu_boundary.mean(),
                    median_F=g.F_integral.median(),
                    median_slower30_percent=100 * g.slower30_abs.median(),
                )
            )
            for threshold in [0.1, 0.2, 0.3, 0.5, 1.0]:
                sensitivity.append(
                    dict(
                        subset=label,
                        minutes=duration,
                        threshold=threshold,
                        pass_percent=100 * (g.stationarity <= threshold).mean(),
                    )
                )
        # Equal weight per parent and per calendar day; resample days, not overlapping intervals.
        parent = subset.groupby(["parent", "minutes"])[
            ["steady30", "common_r2", "trend_sensitivity", "slow3_abs"]
        ].mean()
        for metric in parent.columns:
            wide = parent[metric].unstack()
            for duration in [60, 120]:
                delta = wide[duration] - wide[30]
                daily = delta.groupby(delta.index.floor("D")).mean().dropna().to_numpy()
                boot = rng.choice(daily, size=(5000, len(daily)), replace=True).mean(
                    axis=1
                )
                uncertainty.append(
                    dict(
                        subset=label,
                        minutes=duration,
                        metric=metric,
                        daily_mean_difference=daily.mean(),
                        ci_low=np.quantile(boot, 0.025),
                        ci_high=np.quantile(boot, 0.975),
                        days=len(daily),
                    )
                )
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "comparison_summary.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUTPUT / "threshold_sensitivity.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(OUTPUT / "paired_day_bootstrap.csv", index=False)
    covs = d.groupby(["parent", "minutes"]).covariance.mean().unstack()
    # Each column covers the identical two hours; differences include period-specific rotation/lag.
    for duration in [60, 120]:
        covs[f"delta_{duration}"] = covs[duration] - covs[30]
        covs[f"relative_delta_{duration}"] = (covs[duration] - covs[30]) / covs[
            30
        ].abs().clip(lower=1e-15)
    covs.to_csv(OUTPUT / "matched_covariances.csv")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    metrics = [
        ("median_r2", "Median spectral fit R²", "Higher fits the model better"),
        ("steady_percent", "Passing 30% stationarity screen (%)", "Higher is better"),
        (
            "median_trend_percent",
            "Median covariance change after detrending (%)",
            "Lower means less trend sensitivity",
        ),
        (
            "median_slow3_percent",
            "Median absolute lowest-three-bin contribution (%)",
            "Endpoint diagnostic; lower is better",
        ),
        (
            "fit_boundary_percent",
            "Fits near broadness parameter bounds (%)",
            "Lower is better",
        ),
        (
            "median_F",
            "Median integral correction factor",
            "Model estimate, not accuracy",
        ),
    ]
    for ax, (metric, title, note) in zip(axes.flat, metrics, strict=True):
        for name, color, style in [
            ("All matched", "#6b7280", "--"),
            ("Stronger signal", "#007c91", "-"),
        ]:
            s = summary[summary.subset == name]
            ax.plot(
                [0, 1, 2],
                s[metric],
                marker="o",
                color=color,
                linestyle=style,
                label=name,
            )
        ax.set_xticks([0, 1, 2], ["30 min", "60 min", "120 min"])
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel(note, fontsize=8, color="#555555")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Wellington H₂O: matched averaging-period comparison\n27 June–17 July 2024",
        fontsize=15,
    )
    fig.savefig(OUTPUT / "averaging_comparison.png", dpi=160)
    fig.savefig(OUTPUT / "averaging_comparison.pdf")
    plt.close(fig)
    arr = np.load(OUTPUT / "example_spectra.npz")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for j, minutes in enumerate([30, 60, 120]):
        for part in range(120 // minutes):
            prefix = f"{minutes}_{part}"
            f, y = arr[prefix + "_f_bin"], arr[prefix + "_fCo_bin"]
            start = pd.Timestamp("2024-06-27 14:00") + pd.Timedelta(
                minutes=part * minutes
            )
            axes[0, j].semilogx(f, y, marker=".", lw=0.8, label=start.strftime("%H:%M"))
            axes[1, j].semilogx(arr[prefix + "_f"], arr[prefix + "_ogive"], lw=1)
        axes[0, j].set_title(f"{minutes}-minute periods")
        axes[0, j].legend(fontsize=8)
        axes[0, j].set_ylabel("Signed f Co / covariance")
        axes[1, j].set_ylabel("High-to-low normalized ogive")
        axes[1, j].axhline(1, color="0.5", lw=0.8)
        for ax in axes[:, j]:
            ax.grid(alpha=0.2)
            ax.set_xlabel("Frequency (Hz)")
    fig.suptitle(
        "Same observations: Wellington H₂O, 27 June 2024, 14:00–16:00", fontsize=14
    )
    fig.savefig(OUTPUT / "matched_example.png", dpi=160)
    plt.close(fig)
    print(summary.round(3).to_string(index=False))
    print(
        "\nPaired daily differences:\n",
        pd.DataFrame(uncertainty).round(4).to_string(index=False),
    )
    print(
        "\nExample:\n",
        d[d.parent == pd.Timestamp("2024-06-27 14:00")][
            [
                "start",
                "minutes",
                "rho",
                "common_r2",
                "native_r2",
                "stationarity",
                "slow3_abs",
                "trend_sensitivity",
            ]
        ]
        .round(4)
        .to_string(index=False),
    )


if __name__ == "__main__":
    run()
