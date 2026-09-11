# Despiking

Despiking *repairs* individual samples, replacing them by interpolation. It is
distinct from [raw-data screening](screening.md), which only flags whole
intervals. Despiking is opt-in and runs before `process_file`.

## Which function to use

| Function | Backend | Iterates | Best for |
| --- | --- | --- | --- |
| `ukde_despike` | `KDEpy.FFTKDE` | Yes | Arrays up to ~10 000 samples |
| `polars_ukde_despike` | `KDEpy.FFTKDE`, $O(n \log n)$ | Optional; one pass by default | Full 30-min blocks at 20 Hz (~36 000 samples) |
| `despike_dataframe` | `polars_ukde_despike` per column | Optional; one pass by default | The usual entry point — many columns at once |

Use `despike_dataframe` for either pandas or Polars input; the output retains
the input frame type, row order, and unselected columns.

## The UKDE method

A sample is a spike when its kernel-density estimate falls below
`prob_threshold` times the *peak* density of the distribution. Spikes become
NaN, then are refilled by linear interpolation, and the next pass runs on the
cleaner signal. Iteration stops when no new spikes appear or `max_iter` is hit
— the DataFrame functions default to just one pass to limit repeated trimming.

This is distribution-based: rare genuine turbulence can also be flagged.
Inspect changes and process long records in blocks with comparable conditions.

!!! note "`prob_threshold` direction"
    Lower values are **more permissive** (fewer points removed); higher values
    are more aggressive. The default is `1e-4`.

## The usual call

```python
import TaylorSwift as tswift

df_clean = tswift.despike_dataframe(
    df,
    columns=["Ux", "Uy", "Uz", "T_SONIC", "CO2_density", "H2O_density"],
    prob_threshold=1e-6,  # gentler than the library default of 1e-4
    max_iter=1,
    bulk_iqr=8.0,        # broader than the library default of 4.0
    verbose=True,
)
```

`verbose=True` reports the count and percentage of changed finite samples,
plus the number of missing/non-finite samples filled. Large replacement fractions
can indicate overly aggressive settings, changing conditions, or instrument issues.
Set `max_iter=0` to bypass cleaning. Internal gaps are interpolated; boundary gaps
remain missing, without extrapolation.

See `examples/despike_demo.ipynb` for per-column overrides, a Parquet row preview,
block processing, plots, and a sensitivity comparison. To validate every row of
the local Wellington dataset with bounded memory:

```sh
python tools/validate_despiking.py data/output/Wellington_filtered.parquet
```

## Single array

```python
import numpy as np
from TaylorSwift.despike import ukde_despike

w_clean = ukde_despike(w, prob_threshold=1e-4, max_iter=10)
```

Returns an array of the same length with no NaN (unless the input was entirely
NaN); spike positions are linearly interpolated, and extrapolated at the ends.

## Single Polars column, fast path

```python
from TaylorSwift.despike import polars_ukde_despike

df = polars_ukde_despike(df, "Uz", prob_threshold=1e-4)
```

By default the KDE is fitted within 4×IQR of the median using an IQR-based
Silverman bandwidth. Set `bulk_iqr=8.0` for a broader fit, or `bulk_iqr=None`
to fit all finite observations. Samples outside the fitted grid get zero density
and are flagged regardless of how low `prob_threshold` is. Fitting all values
can preserve rare real populations but can also retain spikes.

`max_iter=1` is the default; larger values explicitly enable repeated passes.
Fewer than four finite values or zero IQR skips spike detection.

## Other methods

The module also carries the simpler filters used by the legacy `CalcFlux`
pipelines:

| Function | Method |
| --- | --- |
| `despike` | Global mean ± `nstd`·σ, interpolated |
| `despike_ewma_fb` | Forward-backward EWMA, threshold on `delta` |
| `despike_med_mod` | Centred rolling median |
| `despike_quart_filter` | Rolling interquartile range |
| `mad_outliers` | Median absolute deviation |
| `spike_detection` | Local moving-window *z*-score |
| `rolling_sigma_filter` | Rolling standard-deviation filter |

`spike_detection` and `mad_outliers` return **boolean masks** rather than
cleaned arrays, which is what you want when you would rather inspect than
replace:

```python
from TaylorSwift.despike import spike_detection

mask = spike_detection(w, window_size=100, z_threshold=4.0)
print(f"{mask.sum()} spikes flagged ({100 * mask.mean():.2f}%)")
```

!!! warning
    `spike_detection` assumes finite, non-NaN input. Pre-filter or impute
    first.

## Should you despike at all?

Despiking is not free — it changes your data, and an over-aggressive threshold
removes real turbulence from the high-frequency tail, biasing fluxes low.

A defensible default:

1. Run `process_file` **without** despiking.
2. Inspect `vm97_*_spike_frac` from [screening](screening.md).
3. Despike only if spike fractions are materially above zero, and only the
   affected columns.
4. Re-run and compare fluxes. A large change is a signal to investigate the
   instrument, not to accept the new number.

See [Quality control](quality-control.md) for reading the flags in step 2.
