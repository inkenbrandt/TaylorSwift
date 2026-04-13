"""
io.py — Data readers for eddy-covariance time-series files.

Currently supports Campbell Scientific TOA5 format (the default output from
LoggerNet / EasyFlux DL).  The TOA5 header structure is:

  Row 0 : file metadata (station name, logger model, etc.)
  Row 1 : column names
  Row 2 : units
  Row 3 : data type / aggregation (e.g. "Smp", "Avg")
  Row 4+: data

Includes a multi-file compiler for building long time series from directories
of raw files (handles overlapping timestamps from multiple files per day).

All public functions return ``polars.DataFrame`` objects.  Polars' columnar
engine and lazy execution model give 5–50× faster CSV parsing than pandas for
typical IRGASON file sizes, and the vectorised sort / unique / diff operations
scale well to multi-day compiled time series.
"""

import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime
import re


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_toa5_timestamp(ts_str: str) -> datetime:
    """Parse a TOA5 timestamp string to a Python :class:`datetime`.

    Strips surrounding quotes and any sub-second fraction that
    :func:`datetime.strptime` cannot handle directly.

    Parameters
    ----------
    ts_str : str
        Raw timestamp field from a TOA5 file, e.g. ``'"2023-06-10 00:00:00"``
        or ``'"2023-06-10 00:00:00.05"``'.

    Returns
    -------
    datetime
    """
    ts_str = ts_str.strip().strip('"')
    ts_str = re.sub(r'\.\d+$', '', ts_str)   # drop sub-second fraction
    return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')


def _format_duration(seconds: float) -> str:
    """Return a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_toa5(
    filepath,
    parse_dates: bool = True,
    drop_diagnostics: bool = False,
    max_diag_value: float = 0.0,
):
    """
    Read a Campbell Scientific TOA5 time-series file.

    Uses ``polars.read_csv`` for fast columnar parsing, then applies
    timestamp parsing and optional diagnostic screening entirely via
    Polars expressions (no Python-level row loops).

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.dat`` / ``.csv`` file.
    parse_dates : bool
        Convert the TIMESTAMP column to ``pl.Datetime`` (default ``True``).
    drop_diagnostics : bool
        If ``True``, set wind / scalar values to ``null`` when the
        corresponding diagnostic flag is non-zero (``diag_sonic``,
        ``diag_irga``).
    max_diag_value : float
        Maximum acceptable diagnostic value; rows exceeding this are flagged
        (default ``0`` — any non-zero flag is bad).

    Returns
    -------
    pl.DataFrame
        Data with original column names from the TOA5 header.
    dict
        Metadata extracted from the first header row (station info, etc.).
    """
    filepath = Path(filepath)

    # --- Parse the four-line header (plain Python I/O) ---------------------
    with open(filepath, 'r') as fh:
        meta_line = fh.readline().strip().strip('"')
        col_line  = fh.readline()
        unit_line = fh.readline()
        _agg_line = fh.readline()   # aggregation row — not stored

    meta_parts = [s.strip().strip('"') for s in meta_line.split(',')]
    metadata = {
        'file_type':    meta_parts[0] if len(meta_parts) > 0 else '',
        'station_id':   meta_parts[1] if len(meta_parts) > 1 else '',
        'logger_model': meta_parts[2] if len(meta_parts) > 2 else '',
        'serial':       meta_parts[3] if len(meta_parts) > 3 else '',
        'os_version':   meta_parts[4] if len(meta_parts) > 4 else '',
        'program':      meta_parts[5] if len(meta_parts) > 5 else '',
    }

    col_names = [s.strip().strip('"') for s in col_line.split(',')]
    units     = [s.strip().strip('"') for s in unit_line.split(',')]
    metadata['units'] = dict(zip(col_names, units))

    # Force TIMESTAMP to be read as String so we can clean it ourselves
    schema_overrides = ({'TIMESTAMP': pl.String}
                        if 'TIMESTAMP' in col_names else {})

    # --- Read the data block with Polars -----------------------------------
    df = pl.read_csv(
        filepath,
        skip_rows=4,
        has_header=False,
        new_columns=col_names,
        null_values=['NAN', '"NAN"', 'NaN', ''],
        infer_schema_length=10_000,
        schema_overrides=schema_overrides,
        ignore_errors=True,
    )

    # --- TIMESTAMP: strip quotes, optionally parse to Datetime -------------
    if 'TIMESTAMP' in df.columns:
        df = df.with_columns(
            pl.col('TIMESTAMP').str.strip_chars('"').alias('TIMESTAMP')
        )
        if parse_dates:
            # High-frequency (20 Hz) TOA5 files often carry sub-second
            # timestamps such as "2023-08-29 00:00:00.05".  Polars'
            # format=None inference can silently produce all-null when it
            # cannot uniquely determine the format from the sample rows, so
            # we strip the fractional-second part first (matching the
            # behaviour of _parse_toa5_timestamp) and then use an explicit
            # format string.  The regex r'\.\d+$' removes ".05", ".000",
            # ".5000000" etc. and is a no-op on second-precision strings.
            df = df.with_columns(
                pl.col('TIMESTAMP')
                .str.replace(r'\.\d+$', '', literal=False)
                .str.to_datetime(format='%Y-%m-%d %H:%M:%S', strict=False)
                .alias('TIMESTAMP')
            )

    # --- Cast all non-timestamp columns to Float64 in one pass ------------
    numeric_cols = [c for c in df.columns if c not in ('TIMESTAMP', 'RECORD')]
    cast_exprs = [
        pl.col(c).cast(pl.Float64, strict=False)
        for c in numeric_cols
        if df.schema[c] != pl.Float64
    ]
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    # --- Optional diagnostic screening ------------------------------------
    if drop_diagnostics:
        if 'diag_sonic' in df.columns:
            sonic_cols = [c for c in ('Ux', 'Uy', 'Uz', 'T_SONIC', 'T_SONIC_corr')
                          if c in df.columns]
            df = df.with_columns([
                pl.when(pl.col('diag_sonic') > max_diag_value)
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in sonic_cols
            ])

        if 'diag_irga' in df.columns:
            irga_cols = [c for c in ('CO2_density', 'H2O_density',
                                     'CO2_density_fast_tmpr')
                         if c in df.columns]
            df = df.with_columns([
                pl.when(pl.col('diag_irga') > max_diag_value)
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in irga_cols
            ])

    return df, metadata


# ---------------------------------------------------------------------------
# Multi-file compilation
# ---------------------------------------------------------------------------

def scan_toa5_directory(
    directory,
    pattern: str = "TOA5_*.dat",
    recursive: bool = False,
):
    """
    Scan a directory for TOA5 files and return sorted file info.

    Uses lightweight Python file I/O (no DataFrame construction) so it is
    fast even over large directories.

    Parameters
    ----------
    directory : str or Path
        Directory to scan.
    pattern : str
        Glob pattern for matching files (default ``"TOA5_*.dat"``).
    recursive : bool
        If ``True``, search subdirectories recursively.

    Returns
    -------
    list[dict]
        List of dicts with keys: ``'path'``, ``'filename'``, ``'size_mb'``,
        ``'first_timestamp'``, ``'last_timestamp'``, ``'n_records'``,
        ``'metadata'``.  Sorted by ``first_timestamp``.
    """
    directory = Path(directory)
    files = sorted(directory.rglob(pattern) if recursive
                   else directory.glob(pattern))

    file_info = []
    for fp in files:
        try:
            # Quick peek: grab first data line for timestamp range
            with open(fp, 'r') as fh:
                for _ in range(4):
                    fh.readline()
                first_line = fh.readline().strip()

            if not first_line:
                continue

            first_ts = _parse_toa5_timestamp(first_line.split(',')[0])

            # Record count (cheap byte scan)
            with open(fp, 'rb') as fh:
                n_lines = sum(1 for _ in fh)
            n_records = n_lines - 4

            # Last timestamp from file tail
            last_ts = None
            with open(fp, 'rb') as fh:
                fh.seek(0, 2)
                fsize = fh.tell()
                fh.seek(max(0, fsize - 2048))
                tail = fh.read().decode('utf-8', errors='ignore')
                lines = [ln for ln in tail.strip().split('\n') if ln]
                if lines:
                    try:
                        last_ts = _parse_toa5_timestamp(lines[-1].split(',')[0])
                    except Exception:
                        pass

            # Metadata from header row
            with open(fp, 'r') as fh:
                meta_line = fh.readline().strip()
            meta_parts = [s.strip().strip('"') for s in meta_line.split(',')]

            file_info.append({
                'path':            fp,
                'filename':        fp.name,
                'size_mb':         fp.stat().st_size / (1024 * 1024),
                'first_timestamp': first_ts,
                'last_timestamp':  last_ts,
                'n_records':       n_records,
                'metadata': {
                    'station_id':   meta_parts[1] if len(meta_parts) > 1 else '',
                    'logger_model': meta_parts[2] if len(meta_parts) > 2 else '',
                },
            })

        except Exception as e:
            print(f"  Warning: Could not scan {fp.name}: {e}")
            continue

    file_info.sort(key=lambda x: x['first_timestamp'])
    return file_info


def compile_toa5(
    source,
    pattern: str = "TOA5_*.dat",
    start_date=None,
    end_date=None,
    drop_diagnostics: bool = True,
    max_diag_value: float = 0.0,
    recursive: bool = False,
    verbose: bool = True,
):
    """
    Compile multiple TOA5 files into a single DataFrame for long time series.

    Handles overlapping timestamps (common when files span arbitrary periods)
    by sorting chronologically and then keeping the first occurrence of each
    duplicate timestamp.  All heavy operations (concat, sort, unique, gap
    detection) run inside Polars' multi-threaded engine.

    Parameters
    ----------
    source : str, Path, or list
        Either a directory path to scan, or an explicit list of file paths.
    pattern : str
        Glob pattern when *source* is a directory (default ``"TOA5_*.dat"``).
    start_date : str or datetime, optional
        Only include data on or after this date (e.g. ``"2023-06-01"``).
    end_date : str or datetime, optional
        Only include data before this date (e.g. ``"2023-07-01"``).
    drop_diagnostics : bool
        Screen bad diagnostic flags (default ``True``).
    max_diag_value : float
        Diagnostic threshold (default ``0``).
    recursive : bool
        Search subdirectories (default ``False``).
    verbose : bool
        Print progress information.

    Returns
    -------
    pl.DataFrame
        Concatenated, deduplicated, time-sorted DataFrame.
    dict
        Compilation metadata (file count, time range, records, gaps).
    """
    # --- Resolve file list -------------------------------------------------
    if isinstance(source, (str, Path)):
        source = Path(source)
        filepaths = ([fi['path'] for fi in scan_toa5_directory(source, pattern, recursive)]
                     if source.is_dir() else [source])
    elif isinstance(source, (list, tuple)):
        filepaths = [Path(f) for f in source]
    else:
        raise ValueError(
            f"source must be a directory, file path, or list; got {type(source)}")

    if not filepaths:
        raise FileNotFoundError(
            f"No TOA5 files found matching '{pattern}' in {source}")

    if verbose:
        print(f"  Found {len(filepaths)} TOA5 file(s) to compile")

    # --- Read all files ----------------------------------------------------
    frames = []
    all_metadata = None
    total_raw_records = 0

    for i, fp in enumerate(filepaths):
        if verbose:
            print(f"  [{i+1}/{len(filepaths)}] Reading {fp.name} "
                  f"({fp.stat().st_size / 1e6:.1f} MB) ...", end='')
        try:
            df_i, meta_i = read_toa5(
                fp,
                parse_dates=True,
                drop_diagnostics=drop_diagnostics,
                max_diag_value=max_diag_value,
            )
            if all_metadata is None:
                all_metadata = meta_i
            total_raw_records += len(df_i)
            frames.append(df_i)
            if verbose:
                print(f" {len(df_i):,} records")
        except Exception as e:
            if verbose:
                print(f" FAILED: {e}")
            continue

    if not frames:
        raise RuntimeError("No files could be read successfully")

    # --- Concatenate, sort, deduplicate ------------------------------------
    if verbose:
        print(f"  Concatenating {len(frames)} DataFrames ...")

    # 'diagonal' fills any schema mismatches with null (safe for mixed setups)
    df = pl.concat(frames, how='diagonal')
    df = df.sort('TIMESTAMP')

    n_before = len(df)
    df = df.unique(subset=['TIMESTAMP'], keep='first', maintain_order=True)
    n_dupes = n_before - len(df)

    # --- Date range filter -------------------------------------------------
    if start_date is not None:
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        df = df.filter(pl.col('TIMESTAMP') >= start_date)
    if end_date is not None:
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        df = df.filter(pl.col('TIMESTAMP') < end_date)

    # --- Detect gaps (fully vectorised with Polars diff) -------------------
    gaps = []
    if len(df) > 1:
        dt_ms = (
            df.select(pl.col('TIMESTAMP').diff().dt.total_milliseconds())
            ['TIMESTAMP']           # column retains original name after select
            .to_numpy()
            .astype(float)
        )
        dt_sec = dt_ms / 1000.0

        valid_dt = dt_sec[np.isfinite(dt_sec) & (dt_sec > 0)]
        median_dt = float(np.median(valid_dt)) if len(valid_dt) > 0 else 1.0

        for idx in np.where(dt_sec > 5 * median_dt)[0]:
            gaps.append({
                'start':        df['TIMESTAMP'][int(idx) - 1],
                'end':          df['TIMESTAMP'][int(idx)],
                'duration_sec': dt_sec[idx],
                'duration_str': _format_duration(dt_sec[idx]),
            })

    compile_meta = {
        'n_files':              len(frames),
        'n_raw_records':        total_raw_records,
        'n_duplicates_removed': n_dupes,
        'n_final_records':      len(df),
        'time_start':           df['TIMESTAMP'][0]  if len(df) > 0 else None,
        'time_end':             df['TIMESTAMP'][-1] if len(df) > 0 else None,
        'n_gaps':               len(gaps),
        'gaps':                 gaps,
        'file_metadata':        all_metadata,
    }

    if verbose:
        print(f"  Compiled: {len(df):,} records")
        print(f"  Time range: {compile_meta['time_start']} → "
              f"{compile_meta['time_end']}")
        print(f"  Duplicates removed: {n_dupes:,}")
        print(f"  Data gaps detected: {len(gaps)}")
        for g in gaps[:10]:
            print(f"    {g['start']} → {g['end']} ({g['duration_str']})")
        if len(gaps) > 10:
            print(f"    ... and {len(gaps) - 10} more")

    return df, compile_meta
