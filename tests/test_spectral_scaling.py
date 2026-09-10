"""Reference checks for every positive FFT bin, including the endpoint."""

import numpy as np
import pytest
from scipy import signal

from TaylorSwift import SiteConfig, compute_cospectrum, compute_spectrum
from TaylorSwift.core import process_interval
from TaylorSwift.rotations import rotate_wind
from TaylorSwift.screening import ScreeningConfig


def reference(x, y, fs):
    options = dict(fs=fs, window=np.hamming(len(x)), detrend=False,
                   return_onesided=True, scaling="density")
    if y is x:
        f, density = signal.periodogram(x, **options)
    else:
        f, density = signal.csd(
            x, y, nperseg=len(x), noverlap=0, **options
        )
    return f[1:], density.real[1:]


@pytest.mark.parametrize("n", [4, 5, 1000, 1001])
@pytest.mark.parametrize("tone", [False, True])
def test_all_bins_match_scipy(n, tone):
    rng = np.random.default_rng(301)
    x = (np.cos(2 * np.pi * (n // 2) * np.arange(n) / n)
         if tone else rng.normal(size=n))
    y = -0.7 * x + 0.2 * rng.normal(size=n)
    for other in (x, y):
        expected_f, expected = reference(x, other, 20.0)
        f, actual = (compute_spectrum(x, 20.0) if other is x
                     else compute_cospectrum(x, other, 20.0))
        np.testing.assert_allclose(f, expected_f, rtol=1e-14)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-14)
        if tone:
            assert abs(expected[-1]) > 0.01


@pytest.mark.parametrize("n", [1000, 1001])
def test_batched_pipeline_matches_reference(monkeypatch, n):
    rng = np.random.default_rng(99)
    tone = np.cos(2 * np.pi * (n // 2) * np.arange(n) / n)
    raw = rng.normal(scale=0.2, size=(6, n)) + tone
    raw += np.array([5, 0, 0, 20, 700, 10])[:, None]
    u, v, w, _ = rotate_wind(*raw[:3])
    detrended = signal.detrend(np.array([u, v, w, *raw[3:]]), axis=1)
    u, v, w, t, c, q = detrended
    pairs = [(w, t), (w, u), (w, c), (w, q),
             (u, u), (v, v), (w, w), (t, t)]
    expected = [reference(x, y, 20.0)[1] for x, y in pairs]
    captured = []

    def capture_bins(f, density, bins_per_decade):
        captured.append(density.copy())
        return f, density

    fft_shapes = []
    original_rfft = np.fft.rfft

    def capture_fft(a, *args, **kwargs):
        fft_shapes.append(a.shape)
        return original_rfft(a, *args, **kwargs)

    monkeypatch.setattr("TaylorSwift.core.log_bin", capture_bins)
    monkeypatch.setattr(np.fft, "rfft", capture_fft)
    result = process_interval(
        *raw, SiteConfig(), screening_config=ScreeningConfig(enabled=False)
    )
    assert fft_shapes == [(6, n)]
    np.testing.assert_allclose(result.freq, reference(w, t, 20.0)[0])
    for actual, target in zip(captured[:8], expected, strict=True):
        np.testing.assert_allclose(actual, target, rtol=1e-9, atol=1e-12)
    for actual, target in zip(captured[8:], expected[:4], strict=True):
        ogive = np.cumsum(target[::-1] * 20.0 / n)[::-1]
        np.testing.assert_allclose(actual, ogive, rtol=1e-9, atol=1e-12)
