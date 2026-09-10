# Reading data

`TaylorSwift` reads Campbell Scientific **TOA5** files natively, using Polars
for the parse. Any other source works too — the spectral stack only needs a
DataFrame with the right column names.

## A single file

```python
import TaylorSwift as tswift

df, meta = tswift.read_toa5("TOA5_mysite_2023_06_10.dat")
```

`read_toa5` returns a `(polars.DataFrame, dict)` pair. The metadata dict comes
from the four-line TOA5 header: station id, logger model, program name, table
name, and the per-column units and aggregation types.

```python
print(meta["station_id"])
print(meta["units"]["Ux"])
```

### Options

| Argument | Default | Effect |
| --- | --- | --- |
| `parse_dates` | `True` | Parse `TIMESTAMP` into a datetime column |
| `drop_diagnostics` | `False` | Drop rows whose diagnostic word exceeds `max_diag_value` |
| `max_diag_value` | `0.0` | Threshold for the above |

Campbell loggers write a diagnostic word per record; anything above zero means
the sonic or IRGA flagged that sample. Dropping them here is cheaper than
finding them later:

```python
df, meta = tswift.read_toa5("TOA5_mysite.dat", drop_diagnostics=True)
```

## Many files

```python
from datetime import datetime

df, meta = tswift.compile_toa5(
    "/data/raw/2023-06-10/",
    pattern="TOA5_*.dat",
    start_date=datetime(2023, 6, 10),
    end_date=datetime(2023, 6, 11),
    drop_diagnostics=True,
    recursive=False,
    verbose=True,
)

print(f"{len(df):,} rows from {meta['n_files']} files")
```

Timestamps retain **microsecond precision** (`Datetime('us')`), without a
timezone. Both the reader and scalar scanner accept whole seconds or 1–6
fractional digits, including mixed precision in one file. Invalid or missing
timestamps and more than six fractional digits are rejected, never truncated.
The scanner warns and skips files with invalid first timestamps.

`compile_toa5` supports one station per call and rejects mixed station IDs;
compile and process each station separately. Sample identity is station ID
plus the full timestamp. Exact repeated parsed rows, including `RECORD`, are
removed. Different values at the same timestamp raise `ValueError` by default.
To explicitly resolve these conflicts, use `conflict_policy="first"` or
`"last"`. These select by input file order, then row order; directory inputs
are ordered by first timestamp, with sorted paths breaking ties. Comparison
happens before diagnostic screening. The output is sorted by timestamp.

Metadata reports total removals in `n_duplicates_removed`, exact repetitions
in `n_exact_duplicates_removed`, and additional distinct rows sharing a
timestamp in `n_conflicting_records`, plus the selected `conflict_policy` and
`timestamp_precision` (`"us"`).

`start_date` /
`end_date` are applied to the concatenated frame, so a file straddling the
boundary is trimmed rather than dropped. The window is half-open —
`start_date <= t < end_date` — so consecutive days chain without
double-counting midnight.

Both accept a `datetime` or an ISO-8601 string:

```python
df, meta = tswift.compile_toa5("/data/raw/", start_date="2023-06-10", end_date="2023-06-11")
```

Set `recursive=True` to walk subdirectories.

## Inspecting before reading

`scan_toa5_directory` reads only the headers, which is fast enough to run over
a season of data:

```python
files = tswift.scan_toa5_directory("/data/raw/", pattern="TOA5_*.dat", recursive=True)

for f in files[:5]:
    print(f["filename"], f["first_timestamp"], "->", f["last_timestamp"],
          f"{f['n_records']} records, {f['size_mb']:.1f} MB")
```

Each entry is a dict with `path`, `filename`, `size_mb`, `first_timestamp`,
`last_timestamp`, `n_records`, and a nested `metadata` dict holding
`station_id` and `logger_model`. The list comes back sorted by
`first_timestamp`.

Use it to spot a gap, a duplicated logger deployment, or a mid-season table
change before it becomes a confusing concatenation error.

## Required columns

`process_file` expects these canonical names:

| Column | Meaning | Units |
| --- | --- | --- |
| `TIMESTAMP` | Record timestamp | datetime |
| `Ux` | Streamwise wind | m s⁻¹ |
| `Uy` | Cross-stream wind | m s⁻¹ |
| `Uz` | Vertical wind | m s⁻¹ |
| `T_SONIC` | Sonic temperature | °C |
| `CO2_density` | CO₂ density | mg m⁻³ |
| `H2O_density` | H₂O density | g m⁻³ |

If a column is missing, `process_file` raises `ValueError` listing both what it
needed and what your frame actually has.

## Renaming non-Campbell columns

Pass `column_map`, mapping *your* names to the canonical ones:

```python
results = tswift.process_file(
    df,
    config,
    column_map={
        "u": "Ux",
        "v": "Uy",
        "w": "Uz",
        "Ts": "T_SONIC",
        "co2": "CO2_density",
        "h2o": "H2O_density",
    },
)
```

Names absent from the frame are ignored, so one shared map can cover several
logger programs.

## Using pandas instead

`process_file` accepts a `pandas.DataFrame` and converts internally. The FFT
pipeline runs on NumPy arrays either way, so results are identical:

```python
import pandas as pd

df_pd = pd.read_csv("my_data.csv", parse_dates=["TIMESTAMP"])
results = tswift.process_file(df_pd, config)
```

Polars is the faster path for large files, since it avoids the conversion copy.
