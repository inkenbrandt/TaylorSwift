# Quality control

`TaylorSwift` runs QC at two levels: [raw-data screening](screening.md) inside
`process_file`, and interval-level tests via `run_qc`.

```python
import TaylorSwift as tswift

results = tswift.process_file(df, config)      # vm97_* flags added here
results = tswift.run_qc(results)               # slope_* and ustar_filter added here
```

## What `run_qc` adds

```python
tswift.run_qc(results, f_range_cospec=(1.0, 8.0), f_range_spec=(1.0, 8.0))
```

The frequency ranges bound the inertial subrange used for slope fitting, in
dimensionless frequency. Widen them for a tall tower where the subrange starts
lower.

### Inertial-subrange slopes

In the inertial subrange, theory fixes the spectral slope. A measured slope far
from the expected value means the spectrum is contaminated — noise, aliasing,
or a failing sensor.

Per signal (`wT`, `wu`, `wCO2`, `wH2O`, `u`, `v`, `w`, `T`) you get:

| Key | Meaning |
| --- | --- |
| `slope_<sig>` | Fitted slope |
| `slope_r2_<sig>` | $R^2$ of the fit |
| `slope_class_<sig>` | `'good'`, `'acceptable'`, `'suspect'`, or `'bad'` |

The expected slope differs between cospectra and power spectra, and the
classification accounts for it:

=== "Cospectra"

    Expected $-4/3 \approx -1.33$ (Kaimal), or $-1$ under stable stratification.

    | Class | Fitted slope |
    | --- | --- |
    | `good` | −1.8 to −0.8 |
    | `acceptable` | −2.2 to −0.5 |
    | `suspect` | −3.0 to 0.0 |
    | `bad` | anything else, or NaN |

=== "Power spectra"

    Expected $-2/3 \approx -0.67$.

    | Class | Fitted slope |
    | --- | --- |
    | `good` | −1.1 to −0.3 |
    | `acceptable` | −1.5 to 0.0 |
    | `suspect` | −2.0 to 0.5 |
    | `bad` | anything else, or NaN |

Ranges are generous on purpose — a single 30-minute record gives a noisy slope
estimate, and this is a screen for gross problems, not a similarity test.

### Friction velocity filter

```python
res.qc_flags["ustar_filter"]   # True when u* < 0.1 m/s
```

Under low turbulence the eddy covariance assumptions fail and fluxes are
systematically underestimated. `True` means **filter this interval out** — the
name reads as "this interval trips the u\* filter".

!!! note "0.1 m s⁻¹ is a placeholder, not a site value"
    The proper threshold is site-specific and derived from a $u_*$ /
    night-time-flux breakpoint analysis. Published values commonly land
    between 0.1 and 0.3 m s⁻¹. Use `res.ustar` directly if you have a real
    threshold:

    ```python
    kept = [r for r in results if r.ustar >= 0.17]
    ```

## Stationarity

Foken & Wichura (1996): split the interval into sub-intervals, compute the
covariance in each, and compare the mean of those against the covariance of the
whole. A large relative difference (RN) means conditions changed mid-interval.

```python
from TaylorSwift.data_quality import stationarity_test

relative_diff, quality_class = stationarity_test(w, T, fs=20.0, n_subwindows=6)
print(f"{relative_diff:.1%} -> class {quality_class}")
```

It returns a `(relative_diff, quality_class)` tuple, where `relative_diff` is
$|1 - \overline{\text{sub covariances}} / \text{full covariance}|$ as a
fraction:

| Class | Relative difference | Reading |
| --- | --- | --- |
| 1 | < 15 % | good |
| 2 | 15–30 % | acceptable |
| 3 | 30–50 % | suspect |
| 4 | > 50 % | bad |

`n_subwindows=6` gives 5-minute sub-windows for a 30-minute interval. Pass
detrended `w` and `x`.

## Foken 9-class flags

The `DataQuality` class combines stationarity, integral turbulence
characteristics, and wind direction into the familiar 1–9 scheme:

```python
from TaylorSwift.data_quality import DataQuality, StabilityParameters, StationarityTest

dq = DataQuality(use_wind_direction=True)
assessment = dq.assess_data_quality(
    stability=stability,
    stationarity=stationarity,
    wind_direction=225.0,
    flux_type="heat",
)

print(assessment["overall_flag"])
```

The returned dict carries `overall_flag`, `stationarity_flag`, `itc_flag`,
`wind_dir_flag`, `itc_measured`, and `itc_modeled`.

Conventional grouping:

| Classes | Meaning |
| --- | --- |
| 1–3 | Suitable for fundamental research |
| 4–6 | Suitable for continuous monitoring (e.g. annual budgets) |
| 7–9 | Discard |

!!! warning "Wind direction is site-specific"
    The built-in wind-direction check flags flow through the sonic's own
    support structure using the CSAT3 boom geometry (degrading around
    151–209°). If your sonic points elsewhere, rotate your directions into that
    convention or set `use_wind_direction=False` and apply your own sector
    filter.

## Putting a filter together

```python
import polars as pl

results = tswift.run_qc(tswift.process_file(df, config))
table = tswift.results_to_dataframe(results)

clean = table.filter(
    ~pl.col("vm97_hard_flag")                      # raw-data screening
    & ~pl.col("ustar_filter")                      # low turbulence
    & pl.col("slope_class_wT").is_in(["good", "acceptable"])
)

print(f"{len(clean)}/{len(table)} intervals retained")
```

Working on the objects instead:

```python
def keep(r):
    f = r.qc_flags
    return (
        not f.get("too_many_nans", False)
        and not f.get("vm97_hard_flag", False)
        and not f.get("ustar_filter", False)
        and f.get("slope_class_wT") in ("good", "acceptable")
    )

clean = [r for r in results if keep(r)]
```

Use `.get()` with defaults — screening keys are absent when screening is
disabled, and `slope_*` keys are absent if you skipped `run_qc`.

## Masking an array

`quality_filter` blanks out elements failing a quality threshold:

```python
from TaylorSwift.data_quality import quality_filter

filtered = quality_filter(flux_array, quality_flags, min_quality=3)
```

Elements with a class above `min_quality` become NaN.

## Report what you filtered

Publication-quality work reports the retention rate and the criteria. The flags
are all in the exported table precisely so this is a one-liner:

```python
for flag in ["vm97_hard_flag", "vm97_soft_flag", "ustar_filter"]:
    print(f"{flag}: {table[flag].sum()} / {len(table)}")
```
