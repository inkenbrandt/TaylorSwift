"""
corrections.py — Flux corrections for eddy covariance data.

This is the single home for every correction applied to computed fluxes;
both the spectral stack (:mod:`TaylorSwift.core`) and the legacy CalcFlux
pipelines (:mod:`TaylorSwift.pipelines`) call the functions defined here.

SPECTRAL (frequency-response) CORRECTIONS:
  * :func:`compute_spectral_correction_factor` — full numerical integration
    of the Kaimal model cospectrum against the combined transfer function
    (Massman 2000).  Transfer functions live in
    :mod:`TaylorSwift.transfer_functions`.
  * :func:`horst_analytical_correction` — closed-form approximation
    (Horst 1997).
  * :func:`apply_spectral_corrections` — applies either method (plus WPL)
    to a list of :class:`~TaylorSwift.results.SpectralResult`.

DENSITY CORRECTIONS:
  * :func:`wpl_correction` — Webb-Pearman-Leuning (1980) correction of
    open-path CO₂/H₂O density fluxes (Fc, Fe).
  * :func:`webb_pearman_leuning` — the Campbell EasyFlux formulation of the
    same WPL correction for latent heat, coupled with the sonic-temperature
    humidity correction.  Used by the CalcFlux pipelines where H and LE are
    solved together.

WIND CORRECTIONS:
  * :func:`shadow_correction` — CSAT3 transducer-shadow correction
    (Horst, Wilczak & Cook 2015).

References
----------
Massman, W.J. (2000). A simple method for estimating frequency response
    corrections for eddy covariance systems. Agric. For. Meteorol., 104,
    185–198.
Horst, T.W. (1997). A simple formula for attenuation of eddy fluxes
    measured with first-order-response scalar sensors. Boundary-Layer
    Meteorol., 82, 219–233.
Webb, E.K., Pearman, G.I. & Leuning, R. (1980). Correction of flux
    measurements for density effects due to heat and water vapour transfer.
    Quart. J. Roy. Meteor. Soc., 106, 85–100.
Leuning, R. (2007). The correct formula for the WPL correction.
    Boundary-Layer Meteorol., 126, 263–272.
Horst, T.W., Wilczak, J.M. & Cook, D. (2015). Correction of a non-orthogonal,
    three-component sonic anemometer for flow distortion by transducer
    shadowing. Boundary-Layer Meteorol., 155, 371–395.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from .constants import CP_DRY_AIR, MOLAR_MASS, R_SPECIFIC, T_ZERO_C
from .transfer_functions import (
    _SCALAR_FLUXES,
    _sensor_tau_for_flux,
    _trapezoid,
    combined_transfer_function,
    kaimal_cospec_model,
)

if TYPE_CHECKING:
    from .config import SiteConfig
    from .results import SpectralResult


# ---------------------------------------------------------------------------
# Micrometeorological helpers (ported from legacy CalcFlux class)
# ---------------------------------------------------------------------------
def shadow_correction(
    Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray, n_iter: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSAT3 transducer-shadow correction (Horst, Wilczak & Cook 2015).
    Parameters
    ----------
    Ux : np.ndarray
        x-component of wind velocity.
    Uy : np.ndarray
        y-component of wind velocity.
    Uz : np.ndarray
        z-component of wind velocity.
    n_iter : int
        Number of iterations for the correction (default 4).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Corrected (Ux, Uy, Uz) wind components.
    """
    h = np.array(
        [
            [0.25, 0.4330127018922193, 0.8660254037844386],
            [-0.5, 0.0, 0.8660254037844386],
            [0.25, -0.4330127018922193, 0.8660254037844386],
        ]
    )
    hinv = np.array(
        [
            [0.6666666666666666, -1.3333333333333333, 0.6666666666666666],
            [1.1547005383792517, 0.0, -1.1547005383792517],
            [0.38490017945975047, 0.38490017945975047, 0.38490017945975047],
        ]
    )
    Ux = np.asarray(Ux, dtype=float).copy()
    Uy = np.asarray(Uy, dtype=float).copy()
    Uz = np.asarray(Uz, dtype=float).copy()

    for _ in range(n_iter):
        Uxh = h[0, 0] * Ux + h[0, 1] * Uy + h[0, 2] * Uz
        Uyh = h[1, 0] * Ux + h[1, 1] * Uy + h[1, 2] * Uz
        Uzh = h[2, 0] * Ux + h[2, 1] * Uy + h[2, 2] * Uz

        scalar = np.sqrt(Ux**2 + Uy**2 + Uz**2)
        with np.errstate(invalid="ignore", divide="ignore"):
            Theta1 = np.arccos(np.clip(np.abs(Uxh) / scalar, 0.0, 1.0))
            Theta2 = np.arccos(np.clip(np.abs(Uyh) / scalar, 0.0, 1.0))
            Theta3 = np.arccos(np.clip(np.abs(Uzh) / scalar, 0.0, 1.0))

        Uxa = Uxh / (0.84 + 0.16 * np.sin(Theta1))
        Uya = Uyh / (0.84 + 0.16 * np.sin(Theta2))
        Uza = Uzh / (0.84 + 0.16 * np.sin(Theta3))

        Ux = hinv[0, 0] * Uxa + hinv[0, 1] * Uya + hinv[0, 2] * Uza
        Uy = hinv[1, 0] * Uxa + hinv[1, 1] * Uya + hinv[1, 2] * Uza
        Uz = hinv[2, 0] * Uxa + hinv[2, 1] * Uya + hinv[2, 2] * Uza

    return Ux, Uy, Uz


# ===================================================================
# WPL (Webb-Pearman-Leuning) density correction
# ===================================================================


def webb_pearman_leuning(
    lamb: float,
    Tsa: float,
    pVavg: float,
    Uz_Ta: float,
    Uz_pV: float,
    p: float,
    Cp: float,
    pD: float,
) -> float:
    """
    WPL-corrected latent heat flux [W m⁻²] — Campbell EasyFlux formulation.

    Solves the Webb et al. (1980) density correction for LE together with
    the sonic-temperature humidity correction (the coupled H/LE system),
    which is why it takes kinematic covariances rather than raw fluxes.
    For the standard open-path Fc/Fe correction use :func:`wpl_correction`.

    Parameters
    ----------
    lamb : float
        Latent heat of vaporisation [J kg⁻¹].
    Tsa : float
        Mean sonic-derived absolute air temperature [K].
    pVavg : float
        Mean water-vapour density [kg m⁻³].
    Uz_Ta : float
        Kinematic sensible-heat covariance w'T' [K m s⁻¹].
    Uz_pV : float
        Kinematic vapour covariance w'ρv' [kg m⁻² s⁻¹].
    p : float
        Moist-air density [kg m⁻³].
    Cp : float
        Moist-air specific heat [J kg⁻¹ K⁻¹].
    pD : float
        Dry-air density [kg m⁻³].

    Returns
    -------
    float
        WPL-corrected latent heat flux [W m⁻²].
    """
    pCpTsa = p * Cp * Tsa
    pRatio = 1.0 + 1.6129 * (pVavg / pD)
    return (
        lamb
        * pCpTsa
        * pRatio
        * (Uz_pV + (pVavg / Tsa) * Uz_Ta)
        / (pCpTsa + lamb * pRatio * pVavg * 0.07)
    )


def wpl_correction(
    Fc_raw: float,
    Fe_raw: float,
    H: float,
    T_mean: float,
    P_mean: float,
    co2_mean: float,
    h2o_mean: float,
) -> dict[str, float]:
    """
    Webb-Pearman-Leuning (1980) density correction for open-path fluxes.

    Parameters
    ----------
    Fc_raw : float
        Uncorrected CO₂ flux (w'ρc') [mg m⁻² s⁻¹].
    Fe_raw : float
        Uncorrected H₂O flux (w'ρv') [g m⁻² s⁻¹].
    H : float
        Sensible heat flux [W m⁻²].
    T_mean : float
        Mean air temperature [°C].
    P_mean : float
        Mean atmospheric pressure [kPa].
    co2_mean : float
        Mean CO₂ density [mg m⁻³].
    h2o_mean : float
        Mean H₂O density [g m⁻³].

    Returns
    -------
    dict with keys 'Fc_wpl', 'Fe_wpl', 'Fc_correction', 'Fe_correction',
    'mu', and 'sigma'.
    """
    Md = MOLAR_MASS["air_dry"]
    Mv = MOLAR_MASS["h2o"]
    mu = Md / Mv
    Rd = R_SPECIFIC["dry_air"]
    cp = CP_DRY_AIR

    T_K = T_mean + T_ZERO_C
    P_Pa = P_mean * 1000.0

    rho_v = h2o_mean * 1e-3
    rho_d = (P_Pa / (Rd * T_K)) - rho_v
    sigma = rho_v / rho_d if rho_d > 0 else 0.0

    Fe_wpl = (1.0 + mu * sigma) * (Fe_raw + (h2o_mean / T_K) * (H / (rho_d * cp)))

    Fe_raw_kg = Fe_raw * 1e-3
    Fc_wpl = (
        Fc_raw
        + mu * (co2_mean * 1e-6 / rho_d) * Fe_raw_kg * 1e6
        + (1.0 + mu * sigma) * (co2_mean / T_K) * (H / (rho_d * cp))
    )

    return {
        "Fc_wpl": Fc_wpl,
        "Fe_wpl": Fe_wpl,
        "Fc_correction": Fc_wpl - Fc_raw,
        "Fe_correction": Fe_wpl - Fe_raw,
        "mu": mu,
        "sigma": sigma,
    }


# ===================================================================
# Spectral correction factors
# ===================================================================


def compute_spectral_correction_factor(
    u_mean: float,
    z_eff: float,
    instrument: SiteConfig,
    averaging_period: float = 30.0,
    flux_type: str = "wT",
    n_freqs: int = 10000,
    f_nd_range: tuple[float, float] = (1e-4, 1e3),
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
    num = _trapezoid(Co_model, np.log(f_nd))

    # Denominator: integral of attenuated cospectrum
    den = _trapezoid(T * Co_model, np.log(f_nd))

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

    Horst (1997) gives the measured-to-true flux ratio as

        F_meas / F ≈ 1 / (1 + (2π n_m τ_eff)^α)

    so the multiplicative correction factor is

        CF = F / F_meas = 1 + (2π n_m τ_eff)^α

    where n_m is the natural frequency of the cospectral peak, τ_eff is
    the effective time constant, and α ≈ 7/8 for the Kaimal cospectrum
    (unstable / neutral stratification).

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

    # Horst (1997): α ≈ 7/8 for Kaimal cospectrum (unstable/neutral)
    alpha = 7.0 / 8.0
    cf = 1.0 + (2.0 * np.pi * n_peak * tau_eff) ** alpha

    return max(cf, 1.0)


# ===================================================================
# Apply corrections to SpectralResult objects
# ===================================================================


def apply_spectral_corrections(
    results: list[SpectralResult],
    site_config: SiteConfig,
    instrument: SiteConfig,
    apply_high_freq: bool = True,
    apply_low_freq: bool = True,
    apply_wpl: bool = True,
    method: str = "massman",
    verbose: bool = False,
) -> list[SpectralResult]:
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
            print(f"  {tstr}  CF: wT={cfs[0]} wu={cfs[1]} wCO2={cfs[2]} wH2O={cfs[3]}")

    return results


# ===================================================================
# Convenience: store mean scalars during processing
# ===================================================================


def enrich_results_with_means(
    results: list[SpectralResult],
    df: Any,
    site_config: SiteConfig,
) -> list[SpectralResult]:
    """
    Add mean scalar values (CO₂, H₂O, pressure) to SpectralResult objects.

    These are needed for WPL corrections but are not computed during the
    core spectral processing. Call this after process_file() and before
    apply_spectral_corrections().

    Parameters
    ----------
    results : list[SpectralResult]
    df : pd.DataFrame
        The original high-frequency DataFrame with columns like
        CO2_density, H2O_density, PA, etc.
    site_config : SiteConfig

    Returns
    -------
    list[SpectralResult]
        The same results list, with co2_mean, h2o_mean, and P_mean
        attributes added to each SpectralResult.
    """
    # Normalise to Polars (no-op if already a pl.DataFrame)
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    for res in results:
        if res.timestamp_start is None or res.timestamp_end is None:
            continue

        sub = df.filter(
            (pl.col("TIMESTAMP") >= res.timestamp_start)
            & (pl.col("TIMESTAMP") < res.timestamp_end)
        )

        if len(sub) == 0:
            continue

        if "CO2_density" in sub.columns:
            val = sub["CO2_density"].mean()
            if val is not None:
                res.co2_mean = float(val)
        if "H2O_density" in sub.columns:
            val = sub["H2O_density"].mean()
            if val is not None:
                res.h2o_mean = float(val)
        if "PA" in sub.columns:
            val = sub["PA"].mean()
            if val is not None:
                res.P_mean = float(val)

    return results
