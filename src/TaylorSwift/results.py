"""
results.py — Result containers and tabular export.

:class:`SpectralResult` holds everything computed for one averaging interval
by the spectral stack (:func:`TaylorSwift.core.process_interval`);
:class:`FluxResult` is the legacy CalcFlux-pipeline record.

The export helpers turn a ``list[SpectralResult]`` into tidy tables:

* :func:`results_to_dataframe` — one row per interval of scalar statistics
  (u*, L, z/L, H, covariances, QC flags, correction factors).
* :func:`spectra_to_dataframe` — long-format spectra/cospectra table with
  one row per (interval, frequency bin).
* :func:`results_to_csv` / :func:`results_to_parquet` — convenience writers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

__all__ = [
    "FluxResult",
    "SpectralResult",
    "results_to_dataframe",
    "spectra_to_dataframe",
    "results_to_csv",
    "results_to_parquet",
]


@dataclass
class FluxResult:
    """Legacy CalcFlux-pipeline record of mean fluxes and diagnostics."""

    Ta: float
    Td: float
    D: float
    Ustr: float
    zeta: float
    H: float
    StDevUz: float
    StDevTa: float
    direction: float
    exchange: float
    lambdaE: float
    ET: float
    Uxy: float

    def to_series(self) -> pd.Series:
        return pd.Series(self.__dict__)


# ---------------------------------------------------------------------------
# Spectral results (one per averaging interval)
# ---------------------------------------------------------------------------
@dataclass
class SpectralResult:
    """
    Container for results from one averaging interval.

    Attributes
    ----------
    timestamp_start
        Start of the averaging interval.
    timestamp_end
        End of the averaging interval.
    u_mean
        Mean streamwise wind [m/s].
    wind_dir
        Horizontal wind direction [degrees relative to sonic x-axis].
    T_mean
        Mean sonic temperature [°C].
    ustar
        Friction velocity [m/s].
    L
        Monin-Obukhov length [m].
    zL
        Stability parameter z/L (dimensionless).
    H
        Sensible heat flux [W/m²].
    cov_wT
        Raw covariance of vertical wind and temperature.
    cov_wu
        Raw covariance of vertical wind and streamwise wind.
    cov_wCO2
        Raw covariance of vertical wind and CO₂ density.
    cov_wH2O
        Raw covariance of vertical wind and H₂O density.
    freq
        Bin-centre frequencies [Hz].
    freq_nd
        Dimensionless frequency f = n*z/U.
    cosp_wT
        Area-preserving cospectrum n·Co_wT(n).
    cosp_wu
        Area-preserving cospectrum n·Co_wu(n).
    cosp_wCO2
        Area-preserving cospectrum n·Co_wCO2(n).
    cosp_wH2O
        Area-preserving cospectrum n·Co_wH2O(n).
    ncosp_wT
        Normalized cospectrum n·Co_wT(n) / cov(w'T').
    ncosp_wu
        Normalized cospectrum n·Co_wu(n) / cov(w'u').
    ncosp_wCO2
        Normalized cospectrum n·Co_wCO2(n) / cov(w'CO2').
    ncosp_wH2O
        Normalized cospectrum n·Co_wH2O(n) / cov(w'H2O').
    spec_u
        Normalized power spectrum n·S_u(n) / σ_u².
    spec_v
        Normalized power spectrum n·S_v(n) / σ_v².
    spec_w
        Normalized power spectrum n·S_w(n) / σ_w².
    spec_T
        Normalized power spectrum n·S_T(n) / σ_T².
    ogive_wT
        Cumulative cospectrum for w'T' (high to low frequency).
    ogive_wu
        Cumulative cospectrum for w'u' (high to low frequency).
    ogive_wCO2
        Cumulative cospectrum for w'CO2' (high to low frequency).
    ogive_wH2O
        Cumulative cospectrum for w'H2O' (high to low frequency).
    co2_mean
        Mean CO₂ density [mg m⁻³], populated by enrich_results_with_means().
    h2o_mean
        Mean H₂O density [g m⁻³], populated by enrich_results_with_means().
    P_mean
        Mean atmospheric pressure [kPa], populated by enrich_results_with_means().
    qc_flags
        Dictionary of quality control flags and intermediate results.
    """

    timestamp_start: datetime | None = None
    timestamp_end: datetime | None = None

    # Mean meteorological quantities
    u_mean: float = np.nan
    wind_dir: float = np.nan
    T_mean: float = np.nan
    ustar: float = np.nan
    L: float = np.nan
    zL: float = np.nan
    H: float = np.nan

    # Raw covariances
    cov_wT: float = np.nan
    cov_wu: float = np.nan
    cov_wCO2: float = np.nan
    cov_wH2O: float = np.nan

    # Frequency arrays (after log binning)
    freq: np.ndarray = field(default_factory=lambda: np.array([]))
    freq_nd: np.ndarray = field(default_factory=lambda: np.array([]))

    # Cospectra  (n * Co_xy)
    cosp_wT: np.ndarray = field(default_factory=lambda: np.array([]))
    cosp_wu: np.ndarray = field(default_factory=lambda: np.array([]))
    cosp_wCO2: np.ndarray = field(default_factory=lambda: np.array([]))
    cosp_wH2O: np.ndarray = field(default_factory=lambda: np.array([]))

    # Normalized cospectra  (n * Co_xy / cov_xy)
    ncosp_wT: np.ndarray = field(default_factory=lambda: np.array([]))
    ncosp_wu: np.ndarray = field(default_factory=lambda: np.array([]))
    ncosp_wCO2: np.ndarray = field(default_factory=lambda: np.array([]))
    ncosp_wH2O: np.ndarray = field(default_factory=lambda: np.array([]))

    # Power spectra  (n * S_x / var_x)
    spec_u: np.ndarray = field(default_factory=lambda: np.array([]))
    spec_v: np.ndarray = field(default_factory=lambda: np.array([]))
    spec_w: np.ndarray = field(default_factory=lambda: np.array([]))
    spec_T: np.ndarray = field(default_factory=lambda: np.array([]))

    # Ogives (cumulative cospectra from high to low frequency)
    ogive_wT: np.ndarray = field(default_factory=lambda: np.array([]))
    ogive_wu: np.ndarray = field(default_factory=lambda: np.array([]))
    ogive_wCO2: np.ndarray = field(default_factory=lambda: np.array([]))
    ogive_wH2O: np.ndarray = field(default_factory=lambda: np.array([]))

    # Mean scalar densities (filled by enrich_results_with_means, needed for WPL)
    co2_mean: float = np.nan
    h2o_mean: float = np.nan
    P_mean: float = np.nan

    # Quality flags (filled by qc module)
    qc_flags: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tabular export
# ---------------------------------------------------------------------------
_SCALAR_FIELDS: tuple[str, ...] = (
    "u_mean",
    "wind_dir",
    "T_mean",
    "ustar",
    "L",
    "zL",
    "H",
    "cov_wT",
    "cov_wu",
    "cov_wCO2",
    "cov_wH2O",
    "co2_mean",
    "h2o_mean",
    "P_mean",
)

_SPECTRA_FIELDS: tuple[str, ...] = (
    "cosp_wT",
    "cosp_wu",
    "cosp_wCO2",
    "cosp_wH2O",
    "ncosp_wT",
    "ncosp_wu",
    "ncosp_wCO2",
    "ncosp_wH2O",
    "spec_u",
    "spec_v",
    "spec_w",
    "spec_T",
    "ogive_wT",
    "ogive_wu",
    "ogive_wCO2",
    "ogive_wH2O",
)


def _scalar_qc_keys(results: Iterable[SpectralResult]) -> list[str]:
    """Union of qc_flags keys holding scalar values, in first-seen order."""
    keys: dict[str, None] = {}
    for res in results:
        for key, value in res.qc_flags.items():
            if isinstance(value, (bool, int, float, str, np.bool_, np.number)):
                keys.setdefault(key, None)
    return list(keys)


def results_to_dataframe(
    results: Sequence[SpectralResult],
    include_qc: bool = True,
) -> pl.DataFrame:
    """
    Flatten a list of SpectralResults into a tidy one-row-per-interval table.

    The table holds the scalar statistics (u*, L, z/L, H, raw covariances,
    mean densities) plus — when ``include_qc`` is True — every scalar entry
    of ``qc_flags`` (QC flags, correction factors ``cf_*``, corrected
    covariances, WPL fluxes).  Intervals missing a given flag get a null.
    Spectra are excluded; use :func:`spectra_to_dataframe` for those.

    Parameters
    ----------
    results : list[SpectralResult]
        Output of :func:`TaylorSwift.core.process_file` (optionally after
        :func:`TaylorSwift.corrections.apply_spectral_corrections` /
        ``run_qc``).
    include_qc : bool
        Include scalar ``qc_flags`` entries as columns (default True).

    Returns
    -------
    pl.DataFrame
        One row per interval, sorted as given. Call ``.to_pandas()`` for a
        pandas frame.
    """
    qc_keys = _scalar_qc_keys(results) if include_qc else []

    rows: list[dict[str, Any]] = []
    for res in results:
        row: dict[str, Any] = {
            "timestamp_start": res.timestamp_start,
            "timestamp_end": res.timestamp_end,
        }
        for name in _SCALAR_FIELDS:
            value = getattr(res, name)
            row[name] = float(value) if value is not None else None
        for key in qc_keys:
            value = res.qc_flags.get(key)
            if isinstance(value, (np.bool_, np.number)):
                value = value.item()
            row[key] = value
        rows.append(row)

    schema: dict[str, Any] = {
        "timestamp_start": pl.Datetime,
        "timestamp_end": pl.Datetime,
    }
    schema.update({name: pl.Float64 for name in _SCALAR_FIELDS})
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema_overrides=schema)


def spectra_to_dataframe(results: Sequence[SpectralResult]) -> pl.DataFrame:
    """
    Long-format spectra table: one row per (interval, frequency bin).

    Columns: ``timestamp_start``, ``freq``, ``freq_nd``, plus every
    (co)spectral array of :class:`SpectralResult` (``cosp_*``, ``ncosp_*``,
    ``spec_*``, ``ogive_*``).  Intervals with empty frequency arrays (e.g.
    skipped by QC) contribute no rows.

    Parameters
    ----------
    results : list[SpectralResult]
        Output of :func:`TaylorSwift.core.process_file` (optionally after
        :func:`TaylorSwift.corrections.apply_spectral_corrections` /
        ``run_qc``).

    Returns
    -------
    pl.DataFrame
    """
    frames: list[pl.DataFrame] = []
    for res in results:
        n = len(res.freq)
        if n == 0:
            continue
        data: dict[str, Any] = {
            "timestamp_start": [res.timestamp_start] * n,
            "freq": np.asarray(res.freq, dtype=np.float64),
            "freq_nd": _column_or_nan(res.freq_nd, n),
        }
        for name in _SPECTRA_FIELDS:
            data[name] = _column_or_nan(getattr(res, name), n)
        frames.append(
            pl.DataFrame(data, schema_overrides={"timestamp_start": pl.Datetime})
        )

    if not frames:
        schema: dict[str, Any] = {"timestamp_start": pl.Datetime}
        schema.update(
            {name: pl.Float64 for name in ("freq", "freq_nd", *_SPECTRA_FIELDS)}
        )
        return pl.DataFrame(schema=schema)
    return pl.concat(frames)


def _column_or_nan(arr: np.ndarray, n: int) -> np.ndarray:
    """Return ``arr`` as float64, or a NaN column if its length is not n."""
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) != n:
        return np.full(n, np.nan)
    return arr


def results_to_csv(
    results: Sequence[SpectralResult],
    path: str | Path,
    include_qc: bool = True,
) -> pl.DataFrame:
    """Write :func:`results_to_dataframe` to CSV; returns the frame.

    Parameters
    ----------
    results : list[SpectralResult]
        Output of :func:`TaylorSwift.core.process_file` (optionally after
        :func:`TaylorSwift.corrections.apply_spectral_corrections` /
        ``run_qc``).
    path : str or Path
        Path to the output CSV file.
    include_qc : bool, optional
        If True, include quality control information in the output.

    Returns
    -------
    pl.DataFrame
        The resulting DataFrame.
    """
    df = results_to_dataframe(results, include_qc=include_qc)
    df.write_csv(str(path))
    return df


def results_to_parquet(
    results: Sequence[SpectralResult],
    path: str | Path,
    include_qc: bool = True,
    spectra_path: str | Path | None = None,
) -> pl.DataFrame:
    """
    Write :func:`results_to_dataframe` to Parquet; returns the frame.

    If ``spectra_path`` is given, the long-format spectra table is written
    there as a second Parquet file.

    Parameters
    ----------
    results : list[SpectralResult]
        Output of :func:`TaylorSwift.core.process_file` (optionally after
        :func:`TaylorSwift.corrections.apply_spectral_corrections` /
        ``run_qc``).
    path : str or Path
        Path to the output Parquet file.
    include_qc : bool, optional
        If True, include quality control information in the output.
    spectra_path : str or Path or None
        If given, the path to the output Parquet file for the long-format
        spectra table.

    Returns
    -------
    pl.DataFrame
        The resulting DataFrame.
    """
    df = results_to_dataframe(results, include_qc=include_qc)
    df.write_parquet(str(path))
    if spectra_path is not None:
        spectra_to_dataframe(results).write_parquet(str(spectra_path))
    return df
