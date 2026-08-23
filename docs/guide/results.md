# Exporting results

`process_file` returns a `list[SpectralResult]` — convenient in Python, awkward
for analysis. Two helpers flatten it into tidy Polars tables.

## One row per interval

```python
import TaylorSwift as tswift

table = tswift.results_to_dataframe(results, include_qc=True)
print(table.head())
```

Holds the scalar statistics — `u_mean`, `wind_dir`, `T_mean`, `ustar`, `L`,
`zL`, `H`, the raw covariances, the mean densities — bracketed by
`timestamp_start` and `timestamp_end`.

With `include_qc=True` (the default) every **scalar** entry of `qc_flags`
becomes a column too: QC flags, correction factors (`cf_*`), corrected
covariances, and WPL fluxes. Intervals missing a given flag get a null rather
than breaking the schema, so you can mix screened and unscreened runs.

Spectra are excluded — they are arrays, and belong in the long table.

```python
table = tswift.results_to_dataframe(results, include_qc=False)  # scalars only
```

## One row per (interval, frequency bin)

```python
long = tswift.spectra_to_dataframe(results)
```

Columns: `timestamp_start`, `freq`, `freq_nd`, and every spectral array —
`cosp_*`, `ncosp_*`, `spec_*`, `ogive_*`.

Intervals with empty frequency arrays — those short-circuited by the NaN test —
contribute no rows, so the table is already clean of skipped intervals.

## Writing to disk

```python
tswift.results_to_csv(results, "fluxes.csv", include_qc=True)
tswift.results_to_parquet(results, "fluxes.parquet", include_qc=True)
```

Both write the interval-level table and **return** it, so you can keep working:

```python
table = tswift.results_to_csv(results, "fluxes.csv")
print(f"wrote {len(table)} intervals")
```

Prefer Parquet for anything large or long-lived — it preserves dtypes
(including the datetime columns and the boolean flags, which CSV flattens to
strings) and is far smaller.

## Getting to pandas

The tables are Polars. Convert if you need to:

```python
df = tswift.results_to_dataframe(results).to_pandas()
```

## Typical analyses

=== "Daily flux totals"

    ```python
    import polars as pl

    daily = (
        table
        .filter(~pl.col("ustar_filter"))
        .group_by(pl.col("timestamp_start").dt.date().alias("date"))
        .agg(
            pl.col("H").mean().alias("H_mean"),
            pl.col("ustar").mean().alias("ustar_mean"),
            pl.len().alias("n_intervals"),
        )
        .sort("date")
    )
    ```

=== "Binning by stability"

    ```python
    stability = (
        table
        .with_columns(
            pl.when(pl.col("zL") < -0.1).then(pl.lit("unstable"))
             .when(pl.col("zL") > 0.1).then(pl.lit("stable"))
             .otherwise(pl.lit("neutral"))
             .alias("regime")
        )
        .group_by("regime")
        .agg(pl.col("ustar").mean(), pl.len().alias("n"))
    )
    ```

=== "Median cospectrum by stability"

    ```python
    long = tswift.spectra_to_dataframe(results)

    med = (
        long
        .join(table.select("timestamp_start", "zL"), on="timestamp_start")
        .filter(pl.col("zL").is_between(-2.0, -0.1))
        .group_by(pl.col("freq_nd").log10().round(1))
        .agg(pl.col("ncosp_wT").median())
        .sort("freq_nd")
    )
    ```

    Joining the two tables on `timestamp_start` is the intended way to carry
    interval-level scalars onto the spectral rows.

## Result containers

| Class | Produced by |
| --- | --- |
| [`SpectralResult`](../api/results.md) | `process_interval` / `process_file` |
| [`FluxResult`](../api/results.md) | The legacy [`CalcFlux` pipelines](pipelines.md) |

`SpectralResult` is a plain dataclass, so `dataclasses.asdict` and normal
attribute access both work if you would rather not go through the tables.
