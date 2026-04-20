import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SiteConfig

# ===================================================================
# Transfer functions
# ===================================================================


def tf_block_average(freq: np.ndarray, averaging_period: float) -> np.ndarray:
    """
    Transfer function for block (running mean) averaging.

    Describes flux loss at low frequencies due to the finite averaging window.
    From Kaimal et al. (1968) Eq. 32 and Moore (1986).

    H(n) = [1 - sin(π n τ) / (π n τ)]²

    where τ is the averaging period in seconds.

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    averaging_period : float
        Averaging period [minutes].

    Returns
    -------
    np.ndarray
        Transfer function values (0 to 1).
    """
    tau = averaging_period * 60.0  # convert to seconds
    x = np.pi * freq * tau
    # Avoid division by zero at freq=0
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(x > 1e-10, (1.0 - np.sin(x) / x) ** 2, 0.0)
    return tf


def tf_linear_detrend(freq: np.ndarray, averaging_period: float) -> np.ndarray:
    """
    Transfer function for linear detrending.

    Rannik & Vesala (1999) showed that linear detrending has a slightly
    different low-frequency response than block averaging. This uses the
    analytical form from Rannik & Vesala (1999).

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    averaging_period : float
        Averaging period [minutes].

    Returns
    -------
    np.ndarray
        Transfer function values (0 to 1).
    """
    tau = averaging_period * 60.0
    x = np.pi * freq * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = np.where(x > 1e-10, np.sin(x) / x, 1.0)
        # Moncrieff et al. (2004) robust form
        tf = 1.0 - sinc**2
    return np.clip(tf, 0.0, 1.0)


def tf_first_order_response(freq: np.ndarray, tau: float) -> np.ndarray:
    """
    Transfer function for a first-order sensor response.

    T_s(n) = 1 / (1 + (2π n τ)²)

    where τ is the sensor time constant [s].  This describes the
    attenuation caused by a sensor that cannot follow rapid fluctuations
    (Moore 1986, Eq. 3).

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    tau : float
        First-order time constant [s].

    Returns
    -------
    np.ndarray
        Transfer function values (0 to 1).
    """
    if tau <= 0:
        return np.ones_like(freq)
    return 1.0 / (1.0 + (2.0 * np.pi * freq * tau) ** 2)


def tf_sonic_line_averaging(
    freq: np.ndarray,
    u_mean: float,
    path_length: float,
) -> np.ndarray:
    """
    Transfer function for sonic anemometer line-averaging.

    The sonic samples the wind averaged over its path length l. This
    attenuates fluctuations at wavelengths comparable to l. From
    Kaimal et al. (1968) Eq. 15:

    T(k₁l) = sin²(k₁l/2) / (k₁l/2)²

    Converted to frequency via Taylor's hypothesis: k₁ = 2πn/U.

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    u_mean : float
        Mean wind speed [m/s].
    path_length : float
        Sonic path length [m].

    Returns
    -------
    np.ndarray
        Transfer function values (0 to 1).
    """
    if u_mean <= 0.1 or path_length <= 0:
        return np.ones_like(freq)

    k1_l = 2.0 * np.pi * freq * path_length / u_mean
    half_k1l = k1_l / 2.0

    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(half_k1l > 1e-10, (np.sin(half_k1l) / half_k1l) ** 2, 1.0)
    return tf


def tf_scalar_path_averaging(
    freq: np.ndarray,
    u_mean: float,
    path_length: float,
) -> np.ndarray:
    """
    Transfer function for scalar (IRGA) path averaging.

    For a cylindrical optical path of length l, the transfer function
    has the same sinc² form as line averaging (Moore 1986):

    T(n) = sin²(πnl/U) / (πnl/U)²

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    u_mean : float
        Mean wind speed [m/s].
    path_length : float
        IRGA optical path length [m].

    Returns
    -------
    np.ndarray
        Transfer function values (0 to 1).
    """
    # Same functional form as sonic line averaging
    return tf_sonic_line_averaging(freq, u_mean, path_length)


def tf_sensor_separation(
    freq: np.ndarray,
    u_mean: float,
    separation: float,
) -> np.ndarray:
    """
    Transfer function for lateral sensor separation.

    When the sonic and gas analyser are separated by distance d
    perpendicular to the mean wind, eddies passing one sensor may
    not be fully sampled by the other. From Moore (1986):

    T(n) = exp(-9.9 (nd/U)^1.5)

    For IRGASON (integrated sensor), separation ≈ 0 and this returns 1.

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    u_mean : float
        Mean wind speed [m/s].
    separation : float
        Total sensor separation distance [m].

    Returns
    -------
    np.ndarray
        Transfer function values (0 to 1).
    """
    if separation <= 0.001 or u_mean <= 0.1:
        return np.ones_like(freq)

    nd_U = freq * separation / u_mean
    return np.exp(-9.9 * nd_U**1.5)


# ===================================================================
# Combined transfer function (Massman 2000 approach)
# ===================================================================
_SCALAR_FLUXES = frozenset({"wT", "wCO2", "wH2O"})

_KAIMAL_PARAMS = {
    "wT": (12.92, 26.7),
    "wCO2": (12.92, 26.7),
    "wH2O": (12.92, 26.7),
    "wu": (9.6, 14.0),
}


def _sensor_tau_for_flux(flux_type: str, instrument: "SiteConfig") -> float:
    if flux_type == "wT":
        return instrument.tau_T
    if flux_type == "wCO2":
        return instrument.tau_co2
    if flux_type == "wH2O":
        return instrument.tau_h2o
    return 0.0


def _validate_flux_type(flux_type: str) -> None:
    if flux_type not in _KAIMAL_PARAMS:
        raise ValueError(f"Unknown flux_type: {flux_type}")


def _calc_alph_x(UHeight: float, L: float) -> tuple[float, float]:
    """Massman (2000) α, X stability parameters."""
    if not np.isfinite(L) or (UHeight / L) <= 0:
        return 0.925, 0.085
    return 1.0, 2.0 - 1.915 / (1.0 + 0.5 * UHeight / L)


def _correct_spectral(B: float, alpha: float, V: float) -> float:
    """Massman (2000, 2001) spectral transfer factor."""
    B_a = B**alpha
    V_a = V**alpha
    return (B_a / (B_a + 1.0)) * (B_a / (B_a + V_a)) * (1.0 / (V_a + 1.0))


def combined_transfer_function(
    freq: np.ndarray,
    u_mean: float,
    instrument: "SiteConfig",
    averaging_period: float = 30.0,
    flux_type: str = "wT",
) -> np.ndarray:
    """
    Compute the combined spectral transfer function for a given flux.

    Multiplies all applicable individual transfer functions following
    Massman (2000). The total transfer function describes the fraction
    of the true cospectral density that is actually measured at each
    frequency.

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    u_mean : float
        Mean horizontal wind speed [m/s].
    instrument : SiteConfig
        Instrument parameters.
    averaging_period : float
        Averaging period [minutes].
    flux_type : str
        Which flux: 'wT', 'wu', 'wCO2', 'wH2O'.

    Returns
    -------
    np.ndarray
        Combined transfer function T(n) at each frequency.
    """
    _validate_flux_type(flux_type)

    # 1. Low-frequency: block averaging
    T_low = tf_block_average(freq, averaging_period)

    # 2. Sonic line-averaging (affects w always, and u for momentum flux)
    T_sonic_w = tf_sonic_line_averaging(freq, u_mean, instrument.sonic_path_length)

    # 3. High-frequency: sensor-specific
    if flux_type == "wT":
        T_sensor = tf_first_order_response(
            freq, _sensor_tau_for_flux(flux_type, instrument)
        )
        T_path_scalar = np.ones_like(freq)  # sonic T has same path as w
        T_sep = np.ones_like(freq)

    elif flux_type == "wu":
        # Momentum: both components are from the sonic
        T_sensor = np.ones_like(freq)  # no extra time constant
        T_path_scalar = tf_sonic_line_averaging(
            freq, u_mean, instrument.sonic_path_length
        )
        T_sep = np.ones_like(freq)

    elif flux_type in {"wCO2", "wH2O"}:
        T_sensor = tf_first_order_response(
            freq, _sensor_tau_for_flux(flux_type, instrument)
        )
        T_path_scalar = tf_scalar_path_averaging(
            freq, u_mean, instrument.irga_path_length
        )
        T_sep = tf_sensor_separation(freq, u_mean, instrument.sensor_separation_total)
    else:
        raise ValueError(f"Unknown flux_type: {flux_type}")

    # Combined: product of all transfer functions
    T_combined = T_low * T_sonic_w * T_sensor * T_path_scalar * T_sep

    return np.clip(T_combined, 1e-10, 1.0)


# ===================================================================
# Correction factor computation
# ===================================================================


def kaimal_cospec_model(f_nd, flux_type="wT"):
    """
    Kaimal et al. (1972) / Massman (2000) model cospectrum.

    Returns n Co(n) / cov(w'x') as a function of dimensionless
    frequency f = nz/U.  Used as the "true" cospectral shape for
    computing correction factors.
    """
    if flux_type not in _KAIMAL_PARAMS:
        flux_type = "wT"
    a, b = _KAIMAL_PARAMS[flux_type]
    return a * f_nd / (1.0 + b * f_nd) ** (7.0 / 4.0)


def compute_spectral_correction_factor(
    u_mean: float,
    z_eff: float,
    instrument: "SiteConfig",
    averaging_period: float = 30.0,
    flux_type: str = "wT",
    n_freqs: int = 10000,
    f_nd_range: tuple = (1e-4, 1e3),
) -> float:
    """
    Compute the multiplicative correction factor for a flux.

    The correction factor CF is:
        CF = ∫ Co_model(f) d(ln f)  /  ∫ T(f) · Co_model(f) d(ln f)

    where Co_model is the Kaimal (1972) model cospectrum and T(f) is the
    combined transfer function.  The corrected flux is:
        F_corrected = CF × F_measured

    This follows Massman (2000) and is equivalent to the approach used in
    EddyPro and other standard EC processing software.

    Parameters
    ----------
    u_mean : float
        Mean wind speed [m/s].
    z_eff : float
        Effective measurement height (z - d) [m].
    instrument : SiteConfig
        Instrument parameters.
    averaging_period : float
        Averaging period [minutes].
    flux_type : str
        'wT', 'wu', 'wCO2', or 'wH2O'.
    n_freqs : int
        Number of frequencies for numerical integration.
    f_nd_range : tuple
        Range of dimensionless frequencies for integration.

    Returns
    -------
    float
        Correction factor (≥ 1.0). Multiply measured flux by this value.
    """
    if u_mean < 0.5:
        return np.nan

    # Dimensionless frequency grid
    f_nd = np.logspace(np.log10(f_nd_range[0]), np.log10(f_nd_range[1]), n_freqs)

    # Convert to natural frequency: n = f_nd * U / z
    freq = f_nd * u_mean / z_eff

    # Model cospectrum (the "true" shape)
    Co_model = kaimal_cospec_model(f_nd, flux_type)

    # Combined transfer function
    T = combined_transfer_function(
        freq, u_mean, instrument, averaging_period, flux_type
    )

    # Integration in log-frequency space: ∫ Co d(ln f)
    # Numerator: integral of true cospectrum
    num = np.trapezoid(Co_model, np.log(f_nd))

    # Denominator: integral of attenuated cospectrum
    den = np.trapezoid(T * Co_model, np.log(f_nd))

    if den > 1e-12:
        cf = num / den
    else:
        cf = np.nan

    return max(cf, 1.0)  # correction factor should always be ≥ 1


def horst_analytical_correction(
    u_mean: float,
    z_eff: float,
    tau_eff: float,
    flux_type: str = "wT",
) -> float:
    """
    Horst (1997) analytical correction factor.

    A simple closed-form approximation that avoids numerical integration.
    Good for quick estimates; less accurate than the full Massman approach
    for complex instrument configurations.

    CF = 1 / (1 + (2π f_m τ_eff)^α)

    where f_m is the dimensionless frequency of the cospectral peak,
    τ_eff is the effective time constant, and α depends on the cospectral
    model (α ≈ 1 for the Kaimal model).

    Parameters
    ----------
    u_mean : float
        Mean wind speed [m/s].
    z_eff : float
        Effective measurement height [m].
    tau_eff : float
        Effective combined time constant [s] (from all high-freq sources).
    flux_type : str
        'wT', 'wu', 'wCO2', or 'wH2O'.

    Returns
    -------
    float
        Correction factor (≥ 1.0).
    """
    if u_mean < 0.5 or tau_eff <= 0:
        return 1.0

    # Cospectral peak frequency (dimensionless) — neutral stability
    if flux_type == "wu":
        f_peak = 0.085  # Kaimal (1972) momentum
    elif flux_type in _SCALAR_FLUXES:
        f_peak = 0.065  # Kaimal (1972) scalars
    else:
        f_peak = 0.065

    # Convert to natural frequency
    n_peak = f_peak * u_mean / z_eff

    # Horst (1997) Eq. 9: α ≈ 7/8 for Kaimal cospectrum
    alpha = 7.0 / 8.0
    cf = 1.0 / (1.0 - (2.0 * np.pi * n_peak * tau_eff) ** alpha)

    return max(cf, 1.0)


# ===================================================================
# Apply corrections to SpectralResult objects
# ===================================================================
# ---------------------------------------------------------------------------
# Dataclass for spectral results
# ---------------------------------------------------------------------------
@dataclass
class SpectralResult:
    """
    Container for results from one averaging interval.

    Attributes:
        timestamp_start: Start of the averaging interval.
        timestamp_end: End of the averaging interval.
        u_mean: Mean streamwise wind [m/s].
        wind_dir: Horizontal wind direction [degrees relative to sonic x-axis].
        T_mean: Mean sonic temperature [°C].
        ustar: Friction velocity [m/s].
        L: Monin-Obukhov length [m].
        zL: Stability parameter z/L (dimensionless).
        H: Sensible heat flux [W/m²].
        cov_wT: Raw covariance of vertical wind and temperature.
        cov_wu: Raw covariance of vertical wind and streamwise wind.
        cov_wCO2: Raw covariance of vertical wind and CO₂ density.
        cov_wH2O: Raw covariance of vertical wind and H₂O density.
        freq: Bin-centre frequencies [Hz].
        freq_nd: Dimensionless frequency f = n*z/U.
        cosp_wT: Area-preserving cospectrum n·Co_wT(n).
        cosp_wu: Area-preserving cospectrum n·Co_wu(n).
        cosp_wCO2: Area-preserving cospectrum n·Co_wCO2(n).
        cosp_wH2O: Area-preserving cospectrum n·Co_wH2O(n).
        ncosp_wT: Normalized cospectrum n·Co_wT(n) / cov(w'T').
        ncosp_wu: Normalized cospectrum n·Co_wu(n) / cov(w'u').
        ncosp_wCO2: Normalized cospectrum n·Co_wCO2(n) / cov(w'CO2').
        ncosp_wH2O: Normalized cospectrum n·Co_wH2O(n) / cov(w'H2O').
        spec_u: Normalized power spectrum n·S_u(n) / σ_u².
        spec_v: Normalized power spectrum n·S_v(n) / σ_v².
        spec_w: Normalized power spectrum n·S_w(n) / σ_w².
        spec_T: Normalized power spectrum n·S_T(n) / σ_T².
        ogive_wT: Cumulative cospectrum for w'T' (high to low frequency).
        ogive_wu: Cumulative cospectrum for w'u' (high to low frequency).
        ogive_wCO2: Cumulative cospectrum for w'CO2' (high to low frequency).
        ogive_wH2O: Cumulative cospectrum for w'H2O' (high to low frequency).
        qc_flags: Dictionary of quality control flags and intermediate results.
    """

    timestamp_start: object = None
    timestamp_end: object = None

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

    # Quality flags (filled by qc module)
    qc_flags: dict = field(default_factory=dict)


def apply_spectral_corrections(
    results,
    site_config: "SiteConfig",
    instrument: "SiteConfig",
    apply_high_freq: bool = True,
    apply_low_freq: bool = True,
    apply_wpl: bool = True,
    method: str = "massman",
    verbose: bool = False,
):
    """
    Apply all spectral and density corrections to a list of SpectralResults.

    Parameters
    ----------
    results : list[SpectralResult]
        Output from process_file().
    site_config : SiteConfig
        Station configuration.
    instrument : SiteConfig
        Instrument parameters.
    apply_high_freq : bool
        Apply high-frequency spectral corrections (default True).
    apply_low_freq : bool
        Apply low-frequency corrections (default True).
    apply_wpl : bool
        Apply WPL density correction for open-path CO₂/H₂O (default True).
        Only applied if instrument.irga_type == 'open_path'.
    method : str
        'massman' for full numerical integration (Massman 2000),
        'horst' for the analytical approximation (Horst 1997).
    verbose : bool
        Print correction factors.

    Returns
    -------
    list[SpectralResult]
        The same results list, with corrections applied in-place.
        New attributes added to qc_flags:
          'cf_wT', 'cf_wu', 'cf_wCO2', 'cf_wH2O' — correction factors
          'cov_wT_corrected', etc. — corrected covariances
          'wpl_Fc', 'wpl_Fe' — WPL-corrected fluxes
    """
    z_eff = site_config.z_eff

    # Imported lazily to avoid requiring correction-stack dependencies
    # when only spectral utilities are used.
    from .corrections import wpl_correction

    for res in results:
        if not np.isfinite(res.u_mean) or res.u_mean < 0.5:
            continue

        # ---------------------------------------------------------------
        # Spectral correction factors
        # ---------------------------------------------------------------
        for flux_type in ["wT", "wu", "wCO2", "wH2O"]:
            if method == "massman":
                cf = compute_spectral_correction_factor(
                    u_mean=res.u_mean,
                    z_eff=z_eff,
                    instrument=instrument,
                    averaging_period=site_config.averaging_period,
                    flux_type=flux_type,
                )
            elif method == "horst":
                # Compute effective time constant for this flux
                tau_eff = _sensor_tau_for_flux(flux_type, instrument)

                # Add path-averaging equivalent time constant
                # τ_path ≈ l / (2π U) for a path of length l
                if res.u_mean > 0.5:
                    tau_path = instrument.irga_path_length / (2.0 * np.pi * res.u_mean)
                    tau_eff = np.sqrt(tau_eff**2 + tau_path**2)

                cf = horst_analytical_correction(res.u_mean, z_eff, tau_eff, flux_type)
            else:
                raise ValueError(f"Unknown method: {method}")

            res.qc_flags[f"cf_{flux_type}"] = cf

            # Apply correction factor to covariances
            cov_attr = f"cov_{flux_type}"
            cov_raw = getattr(res, cov_attr)
            if np.isfinite(cf) and np.isfinite(cov_raw):
                res.qc_flags[f"{cov_attr}_corrected"] = cov_raw * cf

            # Also correct the cospectral arrays by dividing by T(f)
            # at each frequency bin (spectral correction)
            if apply_high_freq and len(res.freq) > 0 and np.isfinite(cf):
                T_f = combined_transfer_function(
                    res.freq,
                    res.u_mean,
                    instrument,
                    site_config.averaging_period,
                    flux_type,
                )

                cosp_attr = f"cosp_{flux_type}"
                cosp = getattr(res, cosp_attr)
                if len(cosp) > 0:
                    cosp_corrected = cosp / T_f
                    setattr(res, cosp_attr, cosp_corrected)

                    # Update normalised version
                    ncosp_attr = f"ncosp_{flux_type}"
                    cov_val = getattr(res, cov_attr)
                    if abs(cov_val) > 1e-12:
                        setattr(res, ncosp_attr, cosp_corrected / cov_val)

        # ---------------------------------------------------------------
        # WPL density correction (open-path only)
        # ---------------------------------------------------------------
        if apply_wpl and instrument.irga_type == "open_path":
            # Need pressure — check if available, otherwise estimate
            P_mean = getattr(res, "P_mean", None)
            if P_mean is None or not np.isfinite(P_mean):
                P_mean = 101.3  # standard atmosphere [kPa]

            # Get mean scalar densities from raw covariances context
            co2_mean = getattr(res, "co2_mean", None)
            h2o_mean = getattr(res, "h2o_mean", None)

            if (
                co2_mean is not None
                and h2o_mean is not None
                and np.isfinite(co2_mean)
                and np.isfinite(h2o_mean)
            ):

                # Use spectrally-corrected covariances if available
                cov_wCO2 = res.qc_flags.get("cov_wCO2_corrected", res.cov_wCO2)
                cov_wH2O = res.qc_flags.get("cov_wH2O_corrected", res.cov_wH2O)
                H_corr = res.qc_flags.get("cov_wT_corrected", res.cov_wT) * 1200.0

                wpl = wpl_correction(
                    Fc_raw=cov_wCO2,
                    Fe_raw=cov_wH2O,
                    H=H_corr,
                    T_mean=res.T_mean,
                    P_mean=P_mean,
                    co2_mean=co2_mean,
                    h2o_mean=h2o_mean,
                )
                res.qc_flags["wpl_Fc"] = wpl["Fc_wpl"]
                res.qc_flags["wpl_Fe"] = wpl["Fe_wpl"]
                res.qc_flags["wpl_Fc_correction"] = wpl["Fc_correction"]
                res.qc_flags["wpl_Fe_correction"] = wpl["Fe_correction"]

        if verbose:
            ts = res.timestamp_start
            tstr = ts.strftime("%H:%M") if ts is not None else "??"
            cfs = [
                f"{res.qc_flags.get(f'cf_{ft}', np.nan):.3f}"
                for ft in ["wT", "wu", "wCO2", "wH2O"]
            ]
            print(
                f"  {tstr}  CF: wT={cfs[0]} wu={cfs[1]} " f"wCO2={cfs[2]} wH2O={cfs[3]}"
            )

    return results


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
    S2 = np.sum(window**2)

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
    valid = np.isfinite(spec)
    spec_safe = np.where(valid, spec, 0.0)
    bin_sum = np.bincount(bin_idx, weights=spec_safe, minlength=n_bins)
    bin_cnt = np.bincount(bin_idx, weights=valid.astype(float), minlength=n_bins)

    populated = bin_cnt > 0
    return bin_centers[populated], bin_sum[populated] / bin_cnt[populated]
