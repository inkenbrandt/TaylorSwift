# Raw-data screening

Before any flux is computed, `TaylorSwift` runs the
[Vickers & Mahrt (1997)](../references.md) instrument-level tests on the
high-frequency signals of each averaging interval.

!!! info "Diagnostic only"
    Screening **records flags; it never discards an interval**. `process_interval`
    merges the results into `SpectralResult.qc_flags` under `vm97_*` keys and
    carries on. Filtering is your decision, made downstream.

Screening runs on the *raw* signals before gap-filling, so flags are recorded
even for intervals that later short-circuit on the NaN test.

## The tests

| Test | Failure mode it catches |
| --- | --- |
| **Spikes** | Short-lived outliers — electronic noise, rain on the transducers |
| **Amplitude resolution** | Too few effective A/D levels; signal lands on a coarse grid |
| **Dropouts** | Instrument "sticks" at one value — a flat line in the trace |
| **Absolute limits** | Value outside the physically plausible range |
| **Higher moments** | Skewness / kurtosis distorted by undetected spikes or artefacts |

Each test yields either a **hard** flag (data very likely unusable) or a
**soft** flag (suspect but potentially usable).

| Severity | Raised by |
| --- | --- |
| Hard | Absolute-limit violations, spike fraction above ~1 %, extreme skewness/kurtosis |
| Soft | Resolution problems, moderate dropouts, moderate higher moments |

## Reading the flags

Screening covers six signals: `u`, `v`, `w`, `T`, `co2`, `h2o`. Per signal you
get:

| Key | Type | Meaning |
| --- | --- | --- |
| `vm97_<sig>_n_spikes` | int | Spikes detected |
| `vm97_<sig>_spike_frac` | float | Spikes as a fraction of the record |
| `vm97_<sig>_ampres_flag` | bool | Amplitude-resolution problem |
| `vm97_<sig>_dropout_flag` | bool | Dropout problem |
| `vm97_<sig>_abslim_count` | int | Samples outside the plausible range |
| `vm97_<sig>_abslim_flag` | bool | Absolute-limit violation |
| `vm97_<sig>_skewness` | float | Skewness of the record |
| `vm97_<sig>_kurtosis` | float | Kurtosis of the record |

Plus three interval-level summaries:

| Key | Meaning |
| --- | --- |
| `vm97_hard_flag` | Any signal raised a hard flag |
| `vm97_soft_flag` | Any signal raised a soft flag |
| `vm97_n_spikes` | Total spikes across all signals |

```python
results = tswift.process_file(df, config)
res = results[0]

if res.qc_flags["vm97_hard_flag"]:
    print("interval failed hard screening")

print(res.qc_flags["vm97_w_spike_frac"])
print(res.qc_flags["vm97_T_kurtosis"])
```

### Filtering on them

```python
clean = [r for r in results if not r.qc_flags["vm97_hard_flag"]]
print(f"{len(clean)}/{len(results)} intervals passed hard screening")
```

Or, on the exported table:

```python
import polars as pl

table = tswift.results_to_dataframe(results)
clean = table.filter(~pl.col("vm97_hard_flag"))
```

## Tuning thresholds

`ScreeningConfig` carries every threshold. Defaults follow the paper.

```python
from TaylorSwift.screening import ScreeningConfig

sc = ScreeningConfig(
    spike_threshold=4.0,      # looser: fewer points called spikes
    spike_window=3000,        # ~2.5 min at 20 Hz
    spike_hard_fraction=0.02, # tolerate 2% spikes before a hard flag
)

results = tswift.process_file(df, config, screening_config=sc)
```

### Spike test

| Field | Default | Meaning |
| --- | --- | --- |
| `spike_window` | `3000` | Moving-window width [samples] |
| `spike_threshold` | `3.5` | Initial threshold in window standard deviations |
| `spike_threshold_increment` | `0.1` | Threshold loosening per pass |
| `spike_max_consecutive` | `3` | Runs longer than this are real fluctuations, not spikes |
| `spike_max_passes` | `4` | Iteration limit |
| `spike_hard_fraction` | `0.01` | Spike fraction above which the flag is hard |

The iteration is the point: each pass replaces detected spikes and loosens the
threshold slightly, so a single large excursion does not mask smaller ones
nearby.

### Amplitude resolution

| Field | Default | Meaning |
| --- | --- | --- |
| `ampres_window` | `1000` | Window width [samples] |
| `ampres_bins` | `100` | Histogram bins per window |
| `ampres_empty_fraction` | `0.70` | Empty-bin fraction that triggers the flag |

### Dropouts

| Field | Default | Meaning |
| --- | --- | --- |
| `dropout_bins` | `100` | Histogram bins |
| `dropout_fraction` | `0.10` | Consecutive-in-one-bin fraction that flags |
| `dropout_extreme_fraction` | `0.06` | Stricter limit at the distribution extremes |

### Higher moments

| Field | Default | Meaning |
| --- | --- | --- |
| `skewness_soft` | `1.0` | Soft flag beyond ±this |
| `skewness_hard` | `2.0` | Hard flag beyond ±this |
| `kurtosis_soft` | `(2.0, 5.0)` | Soft flag outside this range |
| `kurtosis_hard` | `(1.0, 8.0)` | Hard flag outside this range |

### Absolute limits

Defaults are physically plausible ranges, in the units the pipeline expects:

| Signal | Range | Units |
| --- | --- | --- |
| `u`, `v` | −30 to 30 | m s⁻¹ |
| `w` | −10 to 10 | m s⁻¹ |
| `T` | −50 to 60 | °C |
| `co2` | 0 to 5000 | mg m⁻³ |
| `h2o` | 0 to 60 | g m⁻³ |

Override per signal — a high-altitude or arid site may need a different H₂O
ceiling:

```python
sc = ScreeningConfig(
    absolute_limits={
        "u": (-30.0, 30.0),
        "v": (-30.0, 30.0),
        "w": (-10.0, 10.0),
        "T": (-40.0, 50.0),
        "co2": (0.0, 5000.0),
        "h2o": (0.0, 25.0),
    }
)
```

## Turning screening off

```python
results = tswift.process_file(df, config, screening_config=ScreeningConfig(enabled=False))
```

The `vm97_*` keys are then absent from `qc_flags`, so guard downstream code
with `.get()` if you make this configurable.

## Calling it directly

Screening works standalone on any dict of arrays:

```python
import numpy as np
from TaylorSwift.screening import vickers_mahrt_screen

flags = vickers_mahrt_screen({
    "u": u, "v": v, "w": w, "T": T, "co2": co2, "h2o": h2o,
})

print(flags["vm97_hard_flag"])
```

Individual tests are public too — `spike_test`, `amplitude_resolution_test`,
`dropout_test`, `absolute_limits_test`, and `higher_moment_test`. See the
[screening API](../api/screening.md).

## Screening vs. despiking

They solve different problems and compose well:

| | Screening | [Despiking](despiking.md) |
| --- | --- | --- |
| Acts on | Whole interval | Individual samples |
| Effect | Records a flag | Replaces values |
| Runs | Inside `process_file` | Before `process_file`, by you |
| Decides | Nothing | Which samples are bad |
