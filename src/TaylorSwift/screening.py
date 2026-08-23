"""
screening.py — Vickers & Mahrt (1997) raw-data screening for eddy covariance.

Instrument-level quality control applied to the high-frequency time series of a
single averaging interval, *before* fluxes are computed.  Each test flags a
different failure mode of the raw sonic / gas-analyser signal:

* **Spikes** — short-lived outliers from electronic noise, rain on the
  transducers, etc.  Detected with an iterative moving-window sigma test; runs
  longer than a few samples are treated as real fluctuations, not spikes.
* **Amplitude resolution** — the analog-to-digital converter or telemetry has
  too few effective levels, so the signal lands on a coarse grid and its
  empirical distribution has many empty bins.
* **Dropouts** — the instrument "sticks" at one value for a stretch of the
  record (a flat line in the trace).
* **Absolute limits** — a value falls outside the physically plausible range,
  a sign of a gross electronic glitch.
* **Higher moments** — skewness and kurtosis outside expected bounds indicate a
  distribution distorted by undetected spikes or non-turbulent artefacts.

The tests are *diagnostic*: :func:`vickers_mahrt_screen` records flags and
statistics but does not itself discard data.  Callers (e.g.
:func:`TaylorSwift.core.process_interval`) merge the returned dictionary into
``SpectralResult.qc_flags`` so the flux table can be filtered downstream.

Following Vickers & Mahrt, each test produces a *hard* flag (data very likely
unusable — absolute-limit violations, spike fraction above ~1 %, extreme
skewness/kurtosis) or a *soft* flag (suspect but potentially usable —
resolution problems, moderate dropouts, moderate higher moments).

References
----------
Vickers, D. & Mahrt, L. (1997). Quality control and flux sampling problems for
    tower and aircraft data. J. Atmos. Ocean. Technol., 14, 512–526.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ScreeningConfig",
    "vickers_mahrt_screen",
    "spike_test",
    "amplitude_resolution_test",
    "dropout_test",
    "absolute_limits_test",
    "higher_moment_test",
]

# Canonical variable keys screened by :func:`vickers_mahrt_screen`, in order.
SCREEN_VARIABLES: tuple[str, ...] = ("u", "v", "w", "T", "co2", "h2o")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _default_absolute_limits() -> dict[str, tuple[float, float]]:
    """Physically plausible ranges for each raw variable (see module units)."""
    return {
        "u": (-30.0, 30.0),  # streamwise wind [m/s]
        "v": (-30.0, 30.0),  # cross-stream wind [m/s]
        "w": (-10.0, 10.0),  # vertical wind [m/s]
        "T": (-50.0, 60.0),  # sonic temperature [°C]
        "co2": (0.0, 5000.0),  # CO₂ density [mg m⁻³]
        "h2o": (0.0, 60.0),  # H₂O density [g m⁻³]
    }


@dataclass
class ScreeningConfig:
    """
    Thresholds for the Vickers & Mahrt (1997) raw-data screening tests.

    All windows are expressed in samples so a single config works at any
    sampling rate; windows are clamped to the interval length automatically.
    Defaults follow the values recommended by Vickers & Mahrt and adopted by
    common eddy-covariance packages, but every threshold is adjustable.

    Attributes
    ----------
    enabled : bool
        Master switch.  When ``False`` :func:`vickers_mahrt_screen` returns an
        empty dictionary and the pipeline skips screening entirely.

    spike_window : int
        Width (samples) of the centred moving window for the spike test.
    spike_threshold : float
        Initial outlier threshold in standard deviations from the local mean.
    spike_threshold_increment : float
        Amount the threshold is relaxed after each pass (Vickers & Mahrt widen
        the window on successive iterations to avoid removing real structure).
    spike_max_consecutive : int
        Runs of consecutive outliers longer than this are considered genuine
        fluctuations, not spikes, and are left in place.
    spike_max_passes : int
        Maximum number of detect-and-interpolate iterations.
    spike_hard_fraction : float
        Fraction of the record flagged as spikes above which the interval gets
        a *hard* flag (default 0.01, i.e. 1 %).

    ampres_window : int
        Width (samples) of the moving window for the amplitude-resolution test.
    ampres_bins : int
        Number of histogram bins spanning each window's range.
    ampres_empty_fraction : float
        A window with more than this fraction of empty bins triggers a *soft*
        resolution flag (default 0.70).

    dropout_bins : int
        Number of histogram bins used to discretise the record for the dropout
        test.
    dropout_fraction : float
        Interior-bin dropout threshold: a *soft* flag is raised when the
        longest run of consecutive samples in a single bin exceeds this
        fraction of the record (default 0.10).
    dropout_extreme_fraction : float
        Dropout threshold for the extreme (lowest/highest) bins; runs there are
        more suspicious, so the threshold is lower and raises a *hard* flag
        (default 0.06).

    skewness_soft, skewness_hard : float
        Absolute-skewness thresholds for soft / hard flags (default 1 / 2).
    kurtosis_soft, kurtosis_hard : tuple[float, float]
        (min, max) Pearson kurtosis (normal distribution = 3) outside which a
        soft / hard flag is raised (defaults (2, 5) / (1, 8)).

    absolute_limits : dict[str, tuple[float, float]]
        Per-variable ``(min, max)`` physical bounds.  A variable absent from
        the mapping is not range-checked.
    """

    enabled: bool = True

    # --- Spike test ---
    spike_window: int = 3000  # ~2.5 min at 20 Hz
    spike_threshold: float = 3.5
    spike_threshold_increment: float = 0.1
    spike_max_consecutive: int = 3
    spike_max_passes: int = 4
    spike_hard_fraction: float = 0.01

    # --- Amplitude resolution ---
    ampres_window: int = 1000
    ampres_bins: int = 100
    ampres_empty_fraction: float = 0.70

    # --- Dropouts ---
    dropout_bins: int = 100
    dropout_fraction: float = 0.10
    dropout_extreme_fraction: float = 0.06

    # --- Higher moments ---
    skewness_soft: float = 1.0
    skewness_hard: float = 2.0
    kurtosis_soft: tuple[float, float] = (2.0, 5.0)
    kurtosis_hard: tuple[float, float] = (1.0, 8.0)

    # --- Absolute limits ---
    absolute_limits: dict[str, tuple[float, float]] = field(
        default_factory=_default_absolute_limits
    )


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _centered_rolling_stats(
    x: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    NaN-aware centred rolling mean and (population) standard deviation.

    Uses prefix sums so the whole array is computed in O(N) regardless of the
    window width.  NaNs are excluded from each window; positions whose window
    holds fewer than two finite samples return NaN std.
    """
    n = len(x)
    w = int(np.clip(window, 1, n)) if n else 1
    half = w // 2

    m = np.isfinite(x)
    x0 = np.where(m, x, 0.0)

    # Prefix sums with a leading zero → window [lo, hi) sum is cs[hi] - cs[lo].
    cs = np.concatenate(([0.0], np.cumsum(x0)))
    cs2 = np.concatenate(([0.0], np.cumsum(x0 * x0)))
    cc = np.concatenate(([0.0], np.cumsum(m.astype(np.float64))))

    idx = np.arange(n)
    lo = np.maximum(idx - half, 0)
    hi = np.minimum(idx + half + 1, n)

    count = cc[hi] - cc[lo]
    s = cs[hi] - cs[lo]
    s2 = cs2[hi] - cs2[lo]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, s / count, np.nan)
        var = s2 / count - mean * mean
    var = np.where(count > 1, np.maximum(var, 0.0), np.nan)
    return mean, np.sqrt(var)


def _limit_run_length(mask: np.ndarray, max_consecutive: int) -> np.ndarray:
    """Drop True-runs longer than ``max_consecutive`` (keep short runs only)."""
    if not mask.any():
        return mask
    out = mask.copy()
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    starts, ends = edges[0::2], edges[1::2]  # ends exclusive
    long_runs = (ends - starts) > max_consecutive
    for s, e in zip(starts[long_runs], ends[long_runs], strict=False):
        out[s:e] = False
    return out


def _interp_replace(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked samples by linear interpolation over the finite rest."""
    out = x.copy()
    to_fill = mask | ~np.isfinite(x)
    good = ~to_fill
    if good.sum() < 2 or not to_fill.any():
        return out
    out[to_fill] = np.interp(
        np.flatnonzero(to_fill), np.flatnonzero(good), out[good]
    )
    return out


def _run_lengths(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (value, length) of each maximal run of equal values in ``labels``."""
    if len(labels) == 0:
        return np.array([], dtype=labels.dtype), np.array([], dtype=int)
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    return labels[starts], ends - starts


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------
def spike_test(
    x: np.ndarray, config: ScreeningConfig | None = None
) -> tuple[int, float, np.ndarray]:
    """
    Iterative moving-window spike detection (Vickers & Mahrt 1997, §3a).

    Points deviating more than ``spike_threshold`` local standard deviations
    from the centred moving mean are flagged; runs longer than
    ``spike_max_consecutive`` are treated as real fluctuations and kept.
    Detected spikes are linearly interpolated and the pass repeats with a
    slightly relaxed threshold, up to ``spike_max_passes`` times or until no
    new spikes appear.

    Returns
    -------
    n_spikes : int
        Number of samples flagged as spikes.
    fraction : float
        ``n_spikes`` divided by the number of finite samples.
    mask : np.ndarray[bool]
        Boolean spike mask over the original array.
    """
    cfg = config or ScreeningConfig()
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    total = np.zeros(n, dtype=bool)
    n_finite = int(np.isfinite(x).sum())
    if n_finite < 3:
        return 0, 0.0, total

    work = x.copy()
    threshold = cfg.spike_threshold
    for _ in range(max(1, cfg.spike_max_passes)):
        mean, std = _centered_rolling_stats(work, cfg.spike_window)
        with np.errstate(invalid="ignore"):
            dev = np.abs(work - mean)
            flagged = (std > 0) & (dev > threshold * std)
        flagged = np.nan_to_num(flagged, nan=False).astype(bool)
        flagged = _limit_run_length(flagged, cfg.spike_max_consecutive)

        new = flagged & ~total
        if not new.any():
            break
        total |= flagged
        work = _interp_replace(work, total)
        threshold += cfg.spike_threshold_increment

    n_spikes = int(total.sum())
    fraction = n_spikes / n_finite if n_finite else 0.0
    return n_spikes, fraction, total


def amplitude_resolution_test(
    x: np.ndarray, config: ScreeningConfig | None = None
) -> tuple[bool, float]:
    """
    Amplitude-resolution test (Vickers & Mahrt 1997, §3b).

    A moving window is discretised into ``ampres_bins`` bins spanning its
    range; a coarsely resolved signal populates only a few bins and leaves many
    empty.  Returns ``(flag, max_empty_fraction)`` where ``flag`` is True when
    any window exceeds ``ampres_empty_fraction`` empty bins.
    """
    cfg = config or ScreeningConfig()
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    n = len(x)
    if len(finite) < cfg.ampres_bins:
        return False, 0.0

    w = int(np.clip(cfg.ampres_window, cfg.ampres_bins, n))
    step = max(1, w // 2)
    max_empty = 0.0
    for start in range(0, n - w + 1, step):
        seg = x[start : start + w]
        seg = seg[np.isfinite(seg)]
        if len(seg) < cfg.ampres_bins:
            continue
        lo, hi = float(seg.min()), float(seg.max())
        if hi <= lo:
            # Zero variance in the window: a single populated bin, i.e. the
            # worst possible resolution.
            max_empty = 1.0
            continue
        hist, _ = np.histogram(seg, bins=cfg.ampres_bins, range=(lo, hi))
        empty_fraction = float(np.mean(hist == 0))
        max_empty = max(max_empty, empty_fraction)

    return max_empty > cfg.ampres_empty_fraction, max_empty


def dropout_test(
    x: np.ndarray, config: ScreeningConfig | None = None
) -> tuple[bool, bool, float]:
    """
    Dropout test (Vickers & Mahrt 1997, §3c).

    The record is discretised into ``dropout_bins`` bins; the longest run of
    consecutive samples falling in one bin measures how long the signal "sticks"
    at a value.  Returns ``(soft_flag, extreme_flag, dropout_fraction)``:

    * ``soft_flag`` — longest interior-bin run exceeds ``dropout_fraction``.
    * ``extreme_flag`` — longest run sits in the lowest or highest bin and
      exceeds ``dropout_extreme_fraction`` (a stuck signal at a distribution
      tail, treated as a hard failure).
    * ``dropout_fraction`` — longest run length as a fraction of the record.
    """
    cfg = config or ScreeningConfig()
    x = np.asarray(x, dtype=np.float64)
    finite_mask = np.isfinite(x)
    finite = x[finite_mask]
    if len(finite) < 2:
        return False, False, 0.0

    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        # A perfectly flat finite signal is one giant dropout.
        return True, False, 1.0

    edges = np.linspace(lo, hi, cfg.dropout_bins + 1)
    # Bin index in [0, dropout_bins-1]; non-finite samples get a sentinel of -1
    # so they break runs rather than extending them.
    bins = np.full(len(x), -1, dtype=np.int64)
    bins[finite_mask] = np.clip(
        np.digitize(finite, edges[1:-1]), 0, cfg.dropout_bins - 1
    )

    values, lengths = _run_lengths(bins)
    real = values >= 0
    if not real.any():
        return False, False, 0.0

    n = len(x)
    longest = int(lengths[real].max())
    dropout_fraction = longest / n

    is_extreme = (values == 0) | (values == cfg.dropout_bins - 1)
    extreme_lengths = lengths[real & is_extreme]
    longest_extreme = int(extreme_lengths.max()) if extreme_lengths.size else 0

    soft_flag = dropout_fraction > cfg.dropout_fraction
    extreme_flag = (longest_extreme / n) > cfg.dropout_extreme_fraction
    return soft_flag, extreme_flag, dropout_fraction


def absolute_limits_test(
    x: np.ndarray, limits: tuple[float, float] | None
) -> tuple[bool, int]:
    """
    Absolute-limits test (Vickers & Mahrt 1997, §3d).

    Returns ``(flag, n_out_of_range)``.  A ``limits`` of ``None`` skips the
    test (returns ``(False, 0)``).
    """
    if limits is None:
        return False, 0
    x = np.asarray(x, dtype=np.float64)
    lo, hi = limits
    with np.errstate(invalid="ignore"):
        out = np.isfinite(x) & ((x < lo) | (x > hi))
    n_out = int(out.sum())
    return n_out > 0, n_out


def higher_moment_test(
    x: np.ndarray, config: ScreeningConfig | None = None
) -> tuple[float, float, bool, bool]:
    """
    Skewness / kurtosis test (Vickers & Mahrt 1997, §3f).

    Kurtosis is the Pearson definition (3 for a normal distribution).  Returns
    ``(skewness, kurtosis, soft_flag, hard_flag)``.
    """
    cfg = config or ScreeningConfig()
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, False, False

    d = x - x.mean()
    m2 = float(np.mean(d**2))
    if m2 <= 0:
        return 0.0, np.nan, False, False
    skew = float(np.mean(d**3) / m2**1.5)
    kurt = float(np.mean(d**4) / m2**2)

    k_soft_lo, k_soft_hi = cfg.kurtosis_soft
    k_hard_lo, k_hard_hi = cfg.kurtosis_hard
    hard = (abs(skew) > cfg.skewness_hard) or (kurt < k_hard_lo) or (kurt > k_hard_hi)
    soft = (abs(skew) > cfg.skewness_soft) or (kurt < k_soft_lo) or (kurt > k_soft_hi)
    return skew, kurt, soft, hard


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def vickers_mahrt_screen(
    signals: dict[str, np.ndarray],
    config: ScreeningConfig | None = None,
) -> dict[str, Any]:
    """
    Run the Vickers & Mahrt (1997) screening battery on a set of raw signals.

    Parameters
    ----------
    signals : dict[str, np.ndarray]
        Mapping of variable key → raw high-frequency array.  Recognised keys
        are ``u``, ``v``, ``w`` (winds), ``T`` (sonic temperature), ``co2`` and
        ``h2o`` (scalar densities); other keys are screened too but only get an
        absolute-limits check if present in ``config.absolute_limits``.
    config : ScreeningConfig, optional
        Thresholds; defaults to :class:`ScreeningConfig`.  If
        ``config.enabled`` is ``False`` an empty dict is returned.

    Returns
    -------
    dict[str, Any]
        Flat, JSON-friendly quality-control flags suitable for merging into
        ``SpectralResult.qc_flags``.  Per variable ``v`` it records:

        ``vm97_{v}_n_spikes``, ``vm97_{v}_spike_frac``,
        ``vm97_{v}_skewness``, ``vm97_{v}_kurtosis``,
        ``vm97_{v}_ampres_flag``, ``vm97_{v}_dropout_flag``,
        ``vm97_{v}_abslim_flag``, ``vm97_{v}_abslim_count``.

        Plus interval-level rollups ``vm97_hard_flag`` and ``vm97_soft_flag``
        (booleans) and ``vm97_n_spikes`` (total spike count across variables).
    """
    cfg = config or ScreeningConfig()
    if not cfg.enabled:
        return {}

    flags: dict[str, Any] = {}
    any_hard = False
    any_soft = False
    total_spikes = 0

    for key, arr in signals.items():
        arr = np.asarray(arr, dtype=np.float64)

        n_spikes, spike_frac, _ = spike_test(arr, cfg)
        ampres_flag, _ = amplitude_resolution_test(arr, cfg)
        dropout_soft, dropout_hard, _ = dropout_test(arr, cfg)
        abslim_flag, abslim_count = absolute_limits_test(
            arr, cfg.absolute_limits.get(key)
        )
        skew, kurt, moment_soft, moment_hard = higher_moment_test(arr, cfg)

        spike_hard = spike_frac > cfg.spike_hard_fraction

        flags[f"vm97_{key}_n_spikes"] = n_spikes
        flags[f"vm97_{key}_spike_frac"] = spike_frac
        flags[f"vm97_{key}_skewness"] = skew
        flags[f"vm97_{key}_kurtosis"] = kurt
        flags[f"vm97_{key}_ampres_flag"] = bool(ampres_flag)
        flags[f"vm97_{key}_dropout_flag"] = bool(dropout_soft or dropout_hard)
        flags[f"vm97_{key}_abslim_flag"] = bool(abslim_flag)
        flags[f"vm97_{key}_abslim_count"] = abslim_count

        total_spikes += n_spikes
        any_hard = any_hard or spike_hard or dropout_hard or abslim_flag or moment_hard
        any_soft = (
            any_soft or ampres_flag or dropout_soft or moment_soft or spike_hard
        )

    flags["vm97_n_spikes"] = total_spikes
    flags["vm97_hard_flag"] = bool(any_hard)
    flags["vm97_soft_flag"] = bool(any_soft)
    return flags
