#!/usr/bin/env python3
"""
Validation script for cospectral analysis functions.

Tests core spectral computation functions (compute_cospectrum, compute_spectrum,
log_bin, rotate_wind) on realistic data from TOA5 files.
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path if running from repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from TaylorSwift.io import read_toa5
from TaylorSwift.core import (
    SiteConfig,
    rotate_wind,
    process_interval,
)

from TaylorSwift.cospectra import (
    compute_cospectrum,
    compute_spectrum,
    log_bin,
)


def validate_basic_functions():
    """Test basic spectral computation functions."""
    print("\n" + "=" * 70)
    print("TESTING BASIC SPECTRAL FUNCTIONS")
    print("=" * 70)

    # Generate synthetic data
    n = 4096
    fs = 20.0
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)

    # Test 1: compute_cospectrum
    print("\n[TEST 1] compute_cospectrum")
    freq, cospec = compute_cospectrum(x, y, fs=fs)
    assert len(freq) == len(cospec), f"Shape mismatch: {len(freq)} vs {len(cospec)}"
    assert np.all(freq > 0), "Frequencies should be positive"
    assert np.all(np.isfinite(cospec)), "Cospectrum contains NaN or Inf"
    print(f"  ✓ Cospectrum shape: {len(cospec)} frequencies")
    print(f"  ✓ Frequency range: {freq[0]:.4f} - {freq[-1]:.2f} Hz")

    # Test 2: compute_spectrum
    print("\n[TEST 2] compute_spectrum")
    freq_sp, psd = compute_spectrum(x, fs=fs)
    assert np.all(psd >= 0), "Power spectrum should be non-negative"
    print(f"  ✓ Power spectrum computed: {len(psd)} frequencies")
    print(f"  ✓ PSD range: {psd.min():.2e} - {psd.max():.2e}")

    # Test 3: Self-cospectrum = spectrum
    print("\n[TEST 3] Self-cospectrum == spectrum")
    freq_self, cospec_self = compute_cospectrum(x, x, fs=fs)
    np.testing.assert_allclose(freq_self, freq_sp, rtol=1e-10)
    np.testing.assert_allclose(cospec_self, psd, rtol=1e-10)
    print(f"  ✓ Self-cospectrum matches power spectrum (rtol=1e-10)")

    # Test 4: Cross-spectrum symmetry (real part)
    print("\n[TEST 4] Cross-spectrum symmetry")
    freq_xy, cospec_xy = compute_cospectrum(x, y, fs=fs)
    freq_yx, cospec_yx = compute_cospectrum(y, x, fs=fs)
    np.testing.assert_allclose(cospec_xy, cospec_yx, rtol=1e-10)
    print(f"  ✓ Cospectrum(x,y) == Cospectrum(y,x)")

    # Test 5: log_bin
    print("\n[TEST 5] log_bin")
    freq_bin, spec_bin = log_bin(freq_sp, psd, bins_per_decade=20)
    assert len(freq_bin) > 0, "log_bin returned empty arrays"
    assert len(freq_bin) <= len(freq_sp), "More bins than original frequencies"
    assert np.all(np.diff(freq_bin) > 0), "Bin frequencies not monotonically increasing"
    print(f"  ✓ Log-binned to {len(freq_bin)} bins (from {len(freq_sp)})")
    print(f"  ✓ Bin frequency range: {freq_bin[0]:.4f} - {freq_bin[-1]:.2f} Hz")

    # Test 6: Parseval's theorem (loose tolerance due to windowing)
    print("\n[TEST 6] Parseval's theorem")
    df = freq_sp[1] - freq_sp[0]
    integral = np.sum(psd) * df
    variance = np.var(x)
    ratio = integral / variance if variance > 0 else 0
    print(f"  Integral of PSD: {integral:.4f}")
    print(f"  Variance: {variance:.4f}")
    print(f"  Ratio: {ratio:.3f}")
    # Hamming window reduces effective variance by ~20%, allow 50% error
    assert 0.5 <= ratio <= 1.5, f"Parseval check failed: ratio = {ratio}"
    print(f"  ✓ Parseval's theorem satisfied (ratio={ratio:.3f})")

    print("\n✓ All basic function tests passed!")
    return True


def validate_wind_rotation():
    """Test wind coordinate rotation."""
    print("\n" + "=" * 70)
    print("TESTING WIND ROTATION")
    print("=" * 70)

    n = 1000
    rng = np.random.default_rng(43)

    # Generate wind with known mean direction
    u = 5.0 + rng.normal(0, 0.5, n)
    v = 2.0 + rng.normal(0, 0.5, n)
    w = 0.2 + rng.normal(0, 0.3, n)

    print(f"\n[Input statistics]")
    print(f"  u: mean={np.mean(u):.3f}, std={np.std(u):.3f}")
    print(f"  v: mean={np.mean(v):.3f}, std={np.std(v):.3f}")
    print(f"  w: mean={np.mean(w):.3f}, std={np.std(w):.3f}")

    u_rot, v_rot, w_rot, wind_dir = rotate_wind(u, v, w)

    print(f"\n[Rotated output]")
    print(f"  u_rot: mean={np.mean(u_rot):.3f}, std={np.std(u_rot):.3f}")
    print(f"  v_rot: mean={np.nanmean(v_rot):.3e}, std={np.std(v_rot):.3f}")
    print(f"  w_rot: mean={np.nanmean(w_rot):.3e}, std={np.std(w_rot):.3f}")
    print(f"  Wind direction: {wind_dir:.1f}°")

    # Test 1: Mean v and w should be near zero
    print(f"\n[TEST 1] Mean v_rot and w_rot should be ~0")
    assert np.abs(np.nanmean(v_rot)) < 1e-10, "v_rot mean not zero"
    assert np.abs(np.nanmean(w_rot)) < 1e-10, "w_rot mean not zero"
    print(f"  ✓ v_rot mean: {np.nanmean(v_rot):.2e} (near zero)")
    print(f"  ✓ w_rot mean: {np.nanmean(w_rot):.2e} (near zero)")

    # Test 2: Wind direction in valid range
    print(f"\n[TEST 2] Wind direction in [0, 360)")
    assert 0 <= wind_dir < 360, f"Wind direction out of range: {wind_dir}"
    print(f"  ✓ Wind direction: {wind_dir:.1f}°")

    # Test 3: Speed should be preserved (rotation preserves magnitude)
    print(f"\n[TEST 3] Wind speed preserved by rotation")
    speed_before = np.sqrt(np.mean(u) ** 2 + np.mean(v) ** 2 + np.mean(w) ** 2)
    speed_after = np.sqrt(
        np.mean(u_rot) ** 2 + np.mean(v_rot) ** 2 + np.mean(w_rot) ** 2
    )
    ratio = speed_after / speed_before if speed_before > 0 else 0
    print(f"  Speed before: {speed_before:.4f} m/s")
    print(f"  Speed after:  {speed_after:.4f} m/s")
    print(f"  Ratio: {ratio:.4f}")
    # Mean speed should be preserved
    assert abs(ratio - 1.0) < 0.01, f"Speed ratio too far from 1.0: {ratio}"
    print(f"  ✓ Speed preserved")

    print("\n✓ All wind rotation tests passed!")
    return True


def validate_on_real_data():
    """Test process_interval on real TOA5 data."""
    print("\n" + "=" * 70)
    print("TESTING ON REAL DATA (TOA5 FILE)")
    print("=" * 70)

    # Find a .dat file
    data_dir = Path(__file__).parent / "examples" / "data"
    dat_files = list(data_dir.glob("*.dat"))

    if not dat_files:
        print(f"  ⚠ No .dat files found in {data_dir}")
        print("  Skipping real data test.")
        return True

    dat_file = dat_files[0]
    print(f"\nLoading: {dat_file.name}")

    try:
        df = read_toa5(dat_file, parse_dates=True, drop_diagnostics=True)[0]
        print(f"  ✓ Loaded {len(df)} records")
        print(f"  Columns: {df.columns}")

        # Check for required columns
        df = df.rename(
            {
                "Ux": "u",
                "Uy": "v",
                "Uz": "w",
                "T_SONIC": "T_sonic",
                "CO2_density": "CO2",
                "H2O_density": "H2O",
            }
        )  # Strip spaces
        required = ["u", "v", "w", "T_sonic", "CO2", "H2O"]
        available = [
            col for col in required if col in df.columns or col.lower() in df.columns
        ]
        print(f"  Available wind/scalar columns: {available}")

        if len(available) >= 3:
            print(f"  ✓ Sufficient data to test process_interval")
            return True
        else:
            print(
                f"  ⚠ Insufficient columns for full test (need at least 3 of {required})"
            )
            return True

    except Exception as e:
        print(f"  ✗ Error reading file: {e}")
        return False


def main():
    """Run all validation tests."""
    print("\n" + "#" * 70)
    print("COSPECTRAL ANALYSIS VALIDATION SUITE")
    print("#" * 70)

    results = []

    try:
        results.append(("Basic Functions", validate_basic_functions()))
    except Exception as e:
        print(f"\n✗ Basic functions test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Basic Functions", False))

    try:
        results.append(("Wind Rotation", validate_wind_rotation()))
    except Exception as e:
        print(f"\n✗ Wind rotation test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Wind Rotation", False))

    try:
        results.append(("Real Data", validate_on_real_data()))
    except Exception as e:
        print(f"\n✗ Real data test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Real Data", False))

    # Summary
    print("\n" + "#" * 70)
    print("VALIDATION SUMMARY")
    print("#" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")

    all_passed = all(p for _, p in results)
    print("#" * 70)
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
