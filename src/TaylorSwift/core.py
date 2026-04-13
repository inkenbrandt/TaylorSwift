"""
core.py — FFT-based (co)spectral computation for eddy covariance data.

Implements the standard micrometeorological workflow:
  1. Extract averaging windows (default 30 min)
  2. Linear detrend each window
  3. Double-rotation coordinate alignment (mean v = 0, mean w = 0)
  4. Compute cross-spectral density via FFT
  5. Logarithmic frequency binning (block averaging in log-space)
  6. Normalize following Kaimal et al. (1972) conventions

References
----------
Kaimal, J.C. et al. (1972). Spectral characteristics of surface-layer
    turbulence. Quart. J. Roy. Meteor. Soc., 98, 563–589.
Moraes, O.L.L. et al. (2008). Comparing spectra and cospectra of turbulence
    over different surface boundary conditions. Physica A, 387, 4927–4939.
Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology. Kluwer.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import timedelta

from .constants import K_VON_KARMAN, G0


# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------
@dataclass
class SiteConfig:
    """Metadata for an eddy-covariance station."""
    z_measurement: float = 3.0        # measurement height [m]
    z_canopy: float = 0.3             # canopy height [m]
    d: Optional[float] = None         # displacement height [m]  (default: 2/3 * z_canopy)
    z0: Optional[float] = None        # roughness length [m]     (default: 0.1 * z_canopy)
    latitude: float = 0.0             # site latitude [deg]
    longitude: float = 0.0            # site longitude [deg]
    sampling_freq: float = 20.0       # Hz
    averaging_period: float = 30.0    # minutes

    def __post_init__(self):
        if self.d is None:
            self.d = (2.0 / 3.0) * self.z_canopy
        if self.z0 is None:
            self.z0 = 0.1 * self.z_canopy

    @property
    def z_eff(self) -> float:
        """Effective measurement height above displacement height."""
        return self.z_measurement - self.d


# ---------------------------------------------------------------------------
# Dataclass for spectral results
# ---------------------------------------------------------------------------
@dataclass
class SpectralResult:
    """Container for results from one averaging interval."""
    timestamp_start: object = None
    timestamp_end: object = None

    # Mean meteorological quantities
    u_mean: float = np.nan            # mean streamwise wind [m/s]
    wind_dir: float = np.nan          # wind direction [deg]
    T_mean: float = np.nan            # mean sonic temperature [°C]
    ustar: float = np.nan             # friction velocity [m/s]
    L: float = np.nan                 # Obukhov length [m]
    zL: float = np.nan                # stability parameter z/L
    H: float = np.nan                 # sensible heat flux [W/m²]

    # Raw covariances
    cov_wT: float = np.nan
    cov_wu: float = np.nan
    cov_wCO2: float = np.nan
    cov_wH2O: float = np.nan

    # Frequency arrays (after log binning)
    freq: np.ndarray = field(default_factory=lambda: np.array([]))
    freq_nd: np.ndarray = field(default_factory=lambda: np.array([]))  # dimensionless f = n*z/U

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

    # Quality flags (filled by qc module)
    qc_flags: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Wind coordinate rotation (double rotation)
# ---------------------------------------------------------------------------
def rotate_wind(u_raw: np.ndarray, v_raw: np.ndarray, w_raw: np.ndarray):
    """
    Apply double rotation so that mean(v_rot) = 0 and mean(w_rot) = 0.

    This aligns the x-axis with the mean horizontal wind vector and tilts
    the coordinate system so the mean vertical velocity vanishes — the
    standard approach in eddy-covariance processing.

    Parameters
    ----------
    u_raw, v_raw, w_raw : array-like
        Raw wind components from the sonic anemometer.

    Returns
    -------
    u_rot, v_rot, w_rot : np.ndarray
        Rotated wind components.
    wind_dir : float
        Horizontal wind direction [degrees from north] — NB: only meaningful
        if the sonic x-axis orientation is known; here it gives the angle
        of the mean wind relative to the sonic x-axis.
    """
    u = np.asarray(u_raw, dtype=np.float64)
    v = np.asarray(v_raw, dtype=np.float64)
    w = np.asarray(w_raw, dtype=np.float64)

    u_bar = np.nanmean(u)
    v_bar = np.nanmean(v)
    w_bar = np.nanmean(w)

    # First rotation: align u with horizontal wind vector
    alpha = np.arctan2(v_bar, u_bar)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)

    u1 = u * cos_a + v * sin_a
    v1 = -u * sin_a + v * cos_a
    w1 = w.copy()

    # Second rotation: tilt to make mean(w) = 0
    u1_bar = np.nanmean(u1)
    w1_bar = np.nanmean(w1)
    beta = np.arctan2(w1_bar, u1_bar)
    cos_b, sin_b = np.cos(beta), np.sin(beta)

    u2 = u1 * cos_b + w1 * sin_b
    v2 = v1.copy()
    w2 = -u1 * sin_b + w1 * cos_b

    wind_dir = np.degrees(alpha) % 360.0

    return u2, v2, w2, wind_dir


# ---------------------------------------------------------------------------
# Core spectral computation
# ---------------------------------------------------------------------------
def _detrend_linear(x: np.ndarray) -> np.ndarray:
    """Remove a linear trend from *x* (handles NaNs via polyfit)."""
    n = len(x)
    t = np.arange(n, dtype=np.float64)
    mask = np.isfinite(x)
    if mask.sum() < 2:
        return x - np.nanmean(x)
    coeffs = np.polyfit(t[mask], x[mask], 1)
    return x - np.polyval(coeffs, t)


def compute_cospectrum(x: np.ndarray, y: np.ndarray, fs: float):
    """
    Compute the one-sided cospectrum of two real signals.

    The cospectrum Co_xy(n) is the real part of the cross-spectral density.
    Its integral over all frequencies equals the covariance cov(x', y').

    Parameters
    ----------
    x, y : np.ndarray
        Detrended time series of equal length.
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    freq : np.ndarray
        Frequency array [Hz] (positive only, excluding DC).
    cospec : np.ndarray
        One-sided cospectral density [units of x * y / Hz].
    """
    N = len(x)
    # Apply Hamming window to reduce spectral leakage
    window = np.hamming(N)
    # Window correction factor for energy preservation
    S1 = np.sum(window)
    S2 = np.sum(window ** 2)

    xw = x * window
    yw = y * window

    X = np.fft.rfft(xw)
    Y = np.fft.rfft(yw)

    # Cross-spectral density (two-sided -> one-sided)
    Sxy = X * np.conj(Y)

    # Normalise: divide by fs * S2 to get spectral density
    # (S2 corrects for the window energy)
    Sxy = Sxy / (fs * S2)

    # One-sided: double all except DC and Nyquist
    Sxy[1:-1] *= 2.0

    # Cospectrum = real part
    cospec = np.real(Sxy)

    # Frequencies
    freq = np.fft.rfftfreq(N, d=1.0 / fs)

    # Drop DC component
    return freq[1:], cospec[1:]


def compute_spectrum(x: np.ndarray, fs: float):
    """
    Compute the one-sided power spectrum of a real signal.

    Parameters
    ----------
    x : np.ndarray
        Detrended time series.
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    freq : np.ndarray
        Frequency array [Hz].
    psd : np.ndarray
        One-sided power spectral density [units²/Hz].
    """
    freq, cospec = compute_cospectrum(x, x, fs)
    return freq, cospec


# ---------------------------------------------------------------------------
# Logarithmic frequency binning
# ---------------------------------------------------------------------------
def log_bin(freq: np.ndarray, spec: np.ndarray, bins_per_decade: int = 20):
    """
    Average spectral estimates into logarithmically spaced bins.

    This is the standard approach in the micromet community for producing
    smooth spectral curves (e.g., Kaimal et al. 1972; Moraes et al. 2008).

    Parameters
    ----------
    freq : np.ndarray
        Frequency array [Hz] (positive, no DC).
    spec : np.ndarray
        Spectral or cospectral density at each frequency.
    bins_per_decade : int
        Number of bins per frequency decade (default 20).

    Returns
    -------
    freq_bin : np.ndarray
        Bin-centre frequencies.
    spec_bin : np.ndarray
        Bin-averaged spectral density.
    """
    if len(freq) == 0:
        return np.array([]), np.array([])

    log_f = np.log10(freq)
    f_min, f_max = log_f.min(), log_f.max()
    n_bins = max(int((f_max - f_min) * bins_per_decade), 1)
    bin_edges = np.linspace(f_min, f_max, n_bins + 1)

    bin_centers = 10.0 ** (0.5 * (bin_edges[:-1] + bin_edges[1:]))

    # Assign each frequency to a 0-indexed bin; clip so the rightmost
    # value (= f_max = bin_edges[-1]) lands in the last bin rather than
    # spilling to n_bins.
    bin_idx = np.clip(np.digitize(log_f, bin_edges[1:]), 0, n_bins - 1)

    # Vectorised per-bin mean, handling NaN in spec without a Python loop
    valid     = np.isfinite(spec)
    spec_safe = np.where(valid, spec, 0.0)
    bin_sum = np.bincount(bin_idx, weights=spec_safe,          minlength=n_bins)
    bin_cnt = np.bincount(bin_idx, weights=valid.astype(float), minlength=n_bins)

    populated = bin_cnt > 0
    return bin_centers[populated], bin_sum[populated] / bin_cnt[populated]


# ---------------------------------------------------------------------------
# Process a single averaging interval
# ---------------------------------------------------------------------------
def process_interval(
    u_raw: np.ndarray,
    v_raw: np.ndarray,
    w_raw: np.ndarray,
    T_sonic: np.ndarray,
    co2: np.ndarray,
    h2o: np.ndarray,
    config: SiteConfig,
    timestamp_start=None,
    timestamp_end=None,
    bins_per_decade: int = 20,
) -> SpectralResult:
    """
    Full cospectral analysis for one averaging interval.

    Steps: quality screen → double rotation → detrend → FFT cospectra →
    log-bin → normalise → compute turbulence statistics.

    Parameters
    ----------
    u_raw, v_raw, w_raw : array-like
        Wind components [m/s].
    T_sonic : array-like
        Sonic temperature [°C].
    co2 : array-like
        CO₂ density [mg m⁻³].
    h2o : array-like
        H₂O density [g m⁻³].
    config : SiteConfig
        Station metadata.
    timestamp_start, timestamp_end : optional
        Timestamps bounding the interval.
    bins_per_decade : int
        Log-binning resolution.

    Returns
    -------
    SpectralResult
    """
    fs = config.sampling_freq
    z = config.z_eff
    res = SpectralResult(timestamp_start=timestamp_start,
                         timestamp_end=timestamp_end)

    # --- Convert to arrays and screen for NaN runs -------------------------
    arrs = [np.asarray(a, dtype=np.float64) for a in
            [u_raw, v_raw, w_raw, T_sonic, co2, h2o]]
    u_r, v_r, w_r, Ts, c, q = arrs

    # Drop intervals where wind data is >5% NaN
    wind_nan_frac = np.mean(np.isnan(u_r) | np.isnan(v_r) | np.isnan(w_r))
    if wind_nan_frac > 0.05:
        res.qc_flags['too_many_nans'] = True
        return res

    # Simple gap-fill: linear interpolation for short gaps
    for arr in [u_r, v_r, w_r, Ts, c, q]:
        nans = np.isnan(arr)
        if nans.any() and (~nans).sum() > 2:
            arr[nans] = np.interp(
                np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans]
            )

    # --- Double rotation ---------------------------------------------------
    u, v, w, wind_dir = rotate_wind(u_r, v_r, w_r)
    res.wind_dir = wind_dir
    res.u_mean = float(np.nanmean(u))

    # --- Batch detrend all 6 signals in one matrix operation ---------------
    # After gap-filling the data is essentially NaN-free; fall back to the
    # scalar _detrend_linear only if residual NaNs remain.
    N = len(u)
    t_idx = np.arange(N, dtype=np.float64)
    t_c   = t_idx - t_idx.mean()
    t_var = float(np.dot(t_c, t_c))

    signals = np.array([u, v, w, Ts, c, q])   # shape (6, N)
    if np.any(~np.isfinite(signals)):
        # Rare fall-back: NaN-safe scalar detrend
        u_p, v_p, w_p, T_p, c_p, q_p = [_detrend_linear(s) for s in signals]
    else:
        slopes     = (signals @ t_c) / t_var          # (6,) — one dot-product each
        intercepts = signals.mean(axis=1) - slopes * t_idx.mean()
        detrended  = signals - (slopes[:, None] * t_idx + intercepts[:, None])
        u_p, v_p, w_p, T_p, c_p, q_p = detrended

    # --- Turbulence statistics ---------------------------------------------
    res.T_mean = float(np.nanmean(Ts))
    T_K = res.T_mean + 273.15  # approximate potential temperature

    res.cov_wT   = float(np.nanmean(w_p * T_p))
    res.cov_wu   = float(np.nanmean(w_p * u_p))
    res.cov_wCO2 = float(np.nanmean(w_p * c_p))
    res.cov_wH2O = float(np.nanmean(w_p * q_p))

    res.ustar = float((res.cov_wu**2 + np.nanmean(w_p * v_p)**2) ** 0.25)

    if abs(res.cov_wT) > 1e-10 and res.ustar > 0.01:
        res.L = -(res.ustar ** 3 * T_K) / (K_VON_KARMAN * G0 * res.cov_wT)
    else:
        res.L = np.nan

    res.zL = z / res.L if np.isfinite(res.L) and abs(res.L) > 0.1 else np.nan
    res.H  = 1200.0 * res.cov_wT

    # --- Compute all 6 FFTs in one batched call ----------------------------
    # Build the Hamming window once; reuse for every cross-spectrum.
    window = np.hamming(N)
    S2     = float(np.dot(window, window))
    norm   = 1.0 / (fs * S2)

    # shape (6, N//2+1) — one rfft per signal, zero extra copies
    W_f, U_f, V_f, T_f, C_f, Q_f = np.fft.rfft(
        np.array([w_p, u_p, v_p, T_p, c_p, q_p]) * window, axis=1
    )

    freq_all = np.fft.rfftfreq(N, d=1.0 / fs)
    freq     = freq_all[1:]           # positive frequencies, no DC

    def _csd(X, Y):
        """One-sided cospectral density from pre-computed FFTs."""
        Sxy = X * np.conj(Y) * norm
        Sxy[1:-1] *= 2.0
        return np.real(Sxy)[1:]       # drop DC component

    # All cross-spectra from the same 6 FFT arrays — no redundant transforms
    cosp_wT_raw   = _csd(W_f, T_f)
    cosp_wu_raw   = _csd(W_f, U_f)
    cosp_wCO2_raw = _csd(W_f, C_f)
    cosp_wH2O_raw = _csd(W_f, Q_f)
    spec_u_raw    = _csd(U_f, U_f)
    spec_v_raw    = _csd(V_f, V_f)
    spec_w_raw    = _csd(W_f, W_f)
    spec_T_raw    = _csd(T_f, T_f)

    # --- Log-bin -----------------------------------------------------------
    freq_bin, cosp_wT   = log_bin(freq, cosp_wT_raw,   bins_per_decade)
    _,        cosp_wu   = log_bin(freq, cosp_wu_raw,   bins_per_decade)
    _,        cosp_wCO2 = log_bin(freq, cosp_wCO2_raw, bins_per_decade)
    _,        cosp_wH2O = log_bin(freq, cosp_wH2O_raw, bins_per_decade)
    _,        spec_u    = log_bin(freq, spec_u_raw,    bins_per_decade)
    _,        spec_v    = log_bin(freq, spec_v_raw,    bins_per_decade)
    _,        spec_w    = log_bin(freq, spec_w_raw,    bins_per_decade)
    _,        spec_T    = log_bin(freq, spec_T_raw,    bins_per_decade)

    res.freq   = freq_bin
    res.freq_nd = (freq_bin * z / res.u_mean
                   if res.u_mean > 0.5 else freq_bin * np.nan)

    # n * Co(n)  — area-preserving form
    res.cosp_wT   = freq_bin * cosp_wT
    res.cosp_wu   = freq_bin * cosp_wu
    res.cosp_wCO2 = freq_bin * cosp_wCO2
    res.cosp_wH2O = freq_bin * cosp_wH2O
    res.spec_u    = freq_bin * spec_u
    res.spec_v    = freq_bin * spec_v
    res.spec_w    = freq_bin * spec_w
    res.spec_T    = freq_bin * spec_T

    # Normalised cospectra: n Co(n) / cov(w'x')
    def _safe_norm(arr, cov):
        return arr / cov if abs(cov) > 1e-12 else arr * np.nan

    res.ncosp_wT   = _safe_norm(res.cosp_wT,   res.cov_wT)
    res.ncosp_wu   = _safe_norm(res.cosp_wu,   res.cov_wu)
    res.ncosp_wCO2 = _safe_norm(res.cosp_wCO2, res.cov_wCO2)
    res.ncosp_wH2O = _safe_norm(res.cosp_wH2O, res.cov_wH2O)

    # --- Ogives (cumulative cospectrum, high → low) -------------------------
    # Reuse the raw cospectra already computed above — no extra FFT calls.
    d_freq = float(freq[1] - freq[0]) if len(freq) > 1 else 1.0

    def _ogive(cosp_raw):
        cum = np.cumsum(cosp_raw[::-1] * d_freq)[::-1]
        _, ogive_bin = log_bin(freq, cum, bins_per_decade)
        return ogive_bin

    res.ogive_wT   = _ogive(cosp_wT_raw)
    res.ogive_wu   = _ogive(cosp_wu_raw)
    res.ogive_wCO2 = _ogive(cosp_wCO2_raw)
    res.ogive_wH2O = _ogive(cosp_wH2O_raw)

    return res


# ---------------------------------------------------------------------------
# Batch-process an entire file
# ---------------------------------------------------------------------------
def process_file(df, config: SiteConfig, bins_per_decade: int = 20):
    """
    Process all averaging intervals in a DataFrame.

    Accepts either a ``polars.DataFrame`` (preferred, returned by
    :func:`eccospectra.io.read_toa5`) or a legacy ``pandas.DataFrame``.
    The actual FFT pipeline operates entirely on NumPy arrays, so both
    input types produce identical results.

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        Must contain columns: ``TIMESTAMP``, ``Ux``, ``Uy``, ``Uz``,
        ``T_SONIC``, ``CO2_density``, ``H2O_density``.
    config : SiteConfig
        Station metadata.
    bins_per_decade : int
        Log-binning resolution.

    Returns
    -------
    list[SpectralResult]
        One result per averaging interval.
    """
    import polars as pl

    # Normalise to Polars (cheap no-op if already Polars)
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    df = df.sort('TIMESTAMP')

    # Drop rows where the timestamp failed to parse (Polars sort puts nulls
    # first with the default nulls_last=False, so df['TIMESTAMP'][0] would
    # be None for any file that has even one unparseable timestamp row).
    df = df.filter(pl.col('TIMESTAMP').is_not_null())
    if len(df) == 0:
        return []

    # --- Build interval edge list without pandas ---------------------------
    period_sec = int(config.averaging_period * 60)
    td = timedelta(seconds=period_sec)

    t0_raw = df['TIMESTAMP'][0]   # Python datetime from Polars Datetime column
    t_end  = df['TIMESTAMP'][-1]

    # Floor t0 to the nearest averaging-period boundary within its day
    day_secs = t0_raw.hour * 3600 + t0_raw.minute * 60 + t0_raw.second
    floored  = (day_secs // period_sec) * period_sec
    t_start  = t0_raw.replace(
        hour        = floored // 3600,
        minute      = (floored % 3600) // 60,
        second      = floored % 60,
        microsecond = 0,
    )

    edges = []
    t = t_start
    while t <= t_end:
        edges.append(t)
        t += td
    edges.append(t)   # sentinel past the last record

    # --- Warn if the declared sampling_freq looks wrong for this file ----------
    # Infer fs from the median inter-sample interval across the first 10 000
    # rows (cheap and avoids scanning the whole file).  If it deviates by
    # more than 20 % from config.sampling_freq the caller probably passed the
    # wrong SiteConfig — skip every interval silently would be the only
    # outcome, so a clear warning is more helpful than silence.
    if len(df) > 10:
        dt_ms_series = (
            df[:10_000]['TIMESTAMP']
            .diff()
            .dt.total_milliseconds()
            .drop_nulls()
        )
        dt_ms_pos = dt_ms_series.filter(dt_ms_series > 0)
        if len(dt_ms_pos) > 0:
            fs_data = 1000.0 / float(dt_ms_pos.median())
            if abs(fs_data - config.sampling_freq) / config.sampling_freq > 0.20:
                import warnings
                warnings.warn(
                    f"process_file: data appears to be sampled at {fs_data:.1f} Hz "
                    f"but config.sampling_freq = {config.sampling_freq} Hz.  "
                    f"Intervals requiring {int(config.averaging_period * 60 * config.sampling_freq):,} "
                    f"samples will be skipped.  Pass the correct sampling_freq to SiteConfig.",
                    stacklevel=2,
                )

    # --- Process each interval ---------------------------------------------
    n_expected = int(config.averaging_period * 60 * config.sampling_freq)
    results = []

    for i in range(len(edges) - 1):
        sub = df.filter(
            (pl.col('TIMESTAMP') >= edges[i]) &
            (pl.col('TIMESTAMP') <  edges[i + 1])
        )

        if len(sub) < 0.9 * n_expected:
            continue   # skip intervals with > 10 % missing records

        res = process_interval(
            u_raw   = sub['Ux'].to_numpy(),
            v_raw   = sub['Uy'].to_numpy(),
            w_raw   = sub['Uz'].to_numpy(),
            T_sonic = sub['T_SONIC'].to_numpy(),
            co2     = sub['CO2_density'].to_numpy(),
            h2o     = sub['H2O_density'].to_numpy(),
            config          = config,
            timestamp_start = edges[i],
            timestamp_end   = edges[i + 1],
            bins_per_decade = bins_per_decade,
        )
        results.append(res)

    return results
