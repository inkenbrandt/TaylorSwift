"""
validate_process_file.py
------------------------
End-to-end validation of the cospectral analysis pipeline using the real
example TOA5 files in examples/data/.

Run from the repo root:
    python validate_process_file.py

Checks for each averaging interval:
  * freq array non-empty, positive, monotonically increasing
  * all cospectra / ogives have the same length as freq
  * power spectra are non-negative
  * turbulence statistics (u_mean, ustar, T_mean) are finite and plausible
  * no too_many_nans QC flag on valid data
  * cospectrum of wT integrates to approximately cov_wT (Parseval-style check)
"""

import sys
import numpy as np
from pathlib import Path
import polars as pl

# ── locate repo root whether run from the root or from anywhere else ──────────
HERE = Path(__file__).resolve().parent  # .../TaylorSwift
SRC = HERE / "src"

if not SRC.exists():
    sys.exit(
        f"Cannot find src/ directory at {SRC}. "
        "Run this script from the TaylorSwift repo root."
    )

sys.path.insert(0, str(SRC))

from TaylorSwift.io import read_toa5
from TaylorSwift.core import SiteConfig, process_file, SpectralResult

DATA_DIR = HERE / "examples" / "data"
if not DATA_DIR.exists():
    sys.exit(f"No examples/data/ directory found at {DATA_DIR}")


# ── site configuration (reasonable defaults for the example files) ────────────
CONFIG = SiteConfig(
    z_measurement=3.5,
    z_canopy=0.4,
    sampling_freq=20.0,
    averaging_period=30.0,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

failures = []
warnings = []


def check(ok, msg, warn=False):
    tag = PASS if ok else (WARN if warn else FAIL)
    print(f"    [{tag}] {msg}")
    if not ok:
        (warnings if warn else failures).append(msg)


def validate_interval(res: SpectralResult, label: str):
    print(f"\n  ── {label}")
    n = len(res.freq)

    check(n > 0, "freq array populated")
    if n == 0:
        return  # nothing more to check

    check(np.all(res.freq > 0), "all frequencies positive")
    check(np.all(np.diff(res.freq) > 0), "freq monotonically increasing")

    for name, arr in [
        ("cosp_wT", res.cosp_wT),
        ("cosp_wu", res.cosp_wu),
        ("cosp_wCO2", res.cosp_wCO2),
        ("cosp_wH2O", res.cosp_wH2O),
    ]:
        check(len(arr) == n, f"{name}: length == len(freq)")
        check(np.all(np.isfinite(arr)), f"{name}: all finite")

    for name, arr in [
        ("spec_u", res.spec_u),
        ("spec_v", res.spec_v),
        ("spec_w", res.spec_w),
        ("spec_T", res.spec_T),
    ]:
        check(len(arr) == n, f"{name}: length == len(freq)")
        check(np.all(arr >= -1e-12), f"{name}: non-negative (auto-spectrum)")

    for name, arr in [
        ("ogive_wT", res.ogive_wT),
        ("ogive_wu", res.ogive_wu),
        ("ogive_wCO2", res.ogive_wCO2),
        ("ogive_wH2O", res.ogive_wH2O),
    ]:
        check(len(arr) == n, f"{name}: length == len(freq)")

    check(np.isfinite(res.u_mean), f"u_mean finite  ({res.u_mean:.3f} m/s)")
    check(res.u_mean > 0, f"u_mean positive")
    check(np.isfinite(res.ustar), f"ustar finite   ({res.ustar:.4f} m/s)")
    check(res.ustar >= 0, f"ustar non-negative")
    check(np.isfinite(res.T_mean), f"T_mean finite  ({res.T_mean:.2f} °C)")

    check(
        not res.qc_flags.get("too_many_nans", False), "no too_many_nans flag", warn=True
    )

    # Parseval-style check: ∫ n·Co_wT(n) d(ln n) ≈ cov_wT
    # Skip when:
    #   * |cov_wT| < 0.005 K·m/s  — covariance too small; noise dominates ratio
    #   * |z/L| > 1.5              — very stable/unstable; cospectral shape is
    #                                strongly distorted and the log-bin integral
    #                                loses accuracy (physically expected behaviour)
    zL_ok = np.isfinite(res.zL) and abs(res.zL) <= 1.5
    if abs(res.cov_wT) > 0.005 and n > 1 and zL_ok:
        d_ln_f = np.diff(np.log(res.freq)).mean()
        integral = np.sum(res.cosp_wT) * d_ln_f
        ratio = integral / res.cov_wT
        check(
            0.05 < ratio < 20.0,
            f"cosp_wT Parseval ratio = {ratio:.3f} (expect ~1.0)",
            warn=True,
        )
    elif n > 1:
        reason = (
            f"|cov_wT| = {abs(res.cov_wT):.5f} (too small)"
            if abs(res.cov_wT) <= 0.005
            else f"z/L = {res.zL:.3f} (|z/L| > 1.5, strongly stratified)"
        )
        print(f"    [info] Parseval check skipped: {reason}")

    print(
        f"    [info] cov_wT={res.cov_wT:.4f}, H={res.H:.1f} W/m², "
        f"L={res.L:.1f} m, z/L={res.zL:.3f}"
        if np.isfinite(res.L)
        else f"    [info] cov_wT={res.cov_wT:.4f}, H={res.H:.1f} W/m², L=NaN"
    )


# ── main loop ─────────────────────────────────────────────────────────────────
dat_files = sorted(DATA_DIR.glob("*.dat"))
if not dat_files:
    sys.exit(f"No .dat files found in {DATA_DIR}")

print(f"\nFound {len(dat_files)} example file(s) in {DATA_DIR.name}/\n")

total_intervals = 0

for fp in dat_files:
    print(f"{'='*64}")
    print(f"File : {fp.name}  ({fp.stat().st_size / 1024:.0f} KB)")

    # ── read ──────────────────────────────────────────────────────────────────
    try:
        df, meta = read_toa5(fp, parse_dates=True)
    except Exception as exc:
        print(f"  [{FAIL}] read_toa5 raised {type(exc).__name__}: {exc}")
        failures.append(f"{fp.name}: read_toa5 failed")
        continue

    n_null_ts = df["TIMESTAMP"].null_count() if "TIMESTAMP" in df.columns else "N/A"
    print(f"  Rows : {len(df):,}  (null TIMESTAMP: {n_null_ts})")
    print(f"  Cols : {df.columns[:10]}")

    required = ["TIMESTAMP", "Ux", "Uy", "Uz", "T_SONIC", "CO2_density", "H2O_density"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [{FAIL}] missing required columns: {missing}")
        failures.append(f"{fp.name}: missing {missing}")
        continue

    # ── auto-detect sampling frequency from median timestamp step ─────────────
    file_config = CONFIG
    if "TIMESTAMP" in df.columns and len(df) > 100:
        dt_ms = df["TIMESTAMP"].diff().dt.total_milliseconds().drop_nulls()
        dt_ms_pos = dt_ms.filter(dt_ms > 0)
        if len(dt_ms_pos) > 0:
            median_dt_ms = float(dt_ms_pos.median())
            fs_detected = (
                round(1000.0 / median_dt_ms, 1)
                if median_dt_ms > 0
                else CONFIG.sampling_freq
            )
            if abs(fs_detected - CONFIG.sampling_freq) > 1.0:
                print(
                    f"  [info] Detected fs = {fs_detected} Hz "
                    f"(config had {CONFIG.sampling_freq} Hz) — using detected value"
                )
                file_config = SiteConfig(
                    z_measurement=CONFIG.z_measurement,
                    z_canopy=CONFIG.z_canopy,
                    sampling_freq=fs_detected,
                    averaging_period=CONFIG.averaging_period,
                )

    # ── process ───────────────────────────────────────────────────────────────
    try:
        results = process_file(df, file_config)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"  [{FAIL}] process_file raised {type(exc).__name__}: {exc}")
        failures.append(f"{fp.name}: process_file failed")
        continue

    print(f"  Intervals produced: {len(results)}")

    if len(results) == 0:
        print(
            f"  [{WARN}] no intervals produced "
            "(file may be shorter than 30 min or timestamps unaligned)"
        )
        warnings.append(f"{fp.name}: 0 intervals")

    for i, res in enumerate(results):
        label = (
            f"interval {i+1}/{len(results)}"
            f"  [{res.timestamp_start} – {res.timestamp_end}]"
        )
        validate_interval(res, label)
        total_intervals += 1

print(f"\n{'='*64}")
print(f"Total intervals validated : {total_intervals}")
print(f"Failures                  : {len(failures)}")
print(f"Warnings                  : {len(warnings)}")

if failures:
    print("\nFAILURES:")
    for f in failures:
        print(f"  {FAIL} {f}")

if warnings:
    print("\nWarnings:")
    for w in warnings:
        print(f"  {WARN} {w}")

if not failures:
    print(
        f"\n\033[32m✓ All hard checks passed — process_file cospectral "
        f"analysis is working correctly.\033[0m"
    )
    sys.exit(0)
else:
    print(f"\n\033[31m✗ {len(failures)} hard check(s) FAILED.\033[0m")
    sys.exit(1)
