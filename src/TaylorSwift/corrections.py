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
    *,
    apply_low_freq: bool = True,
    apply_high_freq: bool = True,
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
    apply_low_freq, apply_high_freq : bool
        Include block averaging and instrument response, independently.
        Both False returns exactly 1, including for unavailable wind.

    Returns
    -------
    float
        Correction factor (≥ 1.0). Multiply measured flux by this value.
    """
    if not apply_low_freq and not apply_high_freq:
        return 1.0
    if not np.isfinite(u_mean) or u_mean < 0.5:
        return np.nan

    # Dimensionless frequency grid
    f_nd = np.logspace(np.log10(f_nd_range[0]), np.log10(f_nd_range[1]), n_freqs)

    # Convert to natural frequency: n = f_nd * U / z
    freq = f_nd * u_mean / z_eff

    # Model cospectrum (the "true" shape)
    Co_model = kaimal_cospec_model(f_nd, flux_type)

    # Combined transfer function
    T = combined_transfer_function(
        freq, u_mean, instrument, averaging_period, flux_type,
        apply_low_freq=apply_low_freq, apply_high_freq=apply_high_freq,
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
        Include block averaging (default True). Linear detrending is not included.
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
        The same results list, with derived products rebuilt from raw data.
        Original arrays, covariances, H and stability fields remain raw.
        ``corrected_spectra`` contains deconvolved ``cosp_*``, ``ogive_*``
        and ``ncosp_*``; normalization uses the log-frequency integral of
        the deconvolved binned array. WPL does not modify these arrays.
        New entries in qc_flags:
          'cf_wT', 'cf_wu', 'cf_wCO2', 'cf_wH2O' — correction factors
          'cov_wT_corrected', etc. — raw covariance times model factor
          'cov_wT_deconvolved', etc. — integrated deconvolved binned arrays
          'H_corrected', 'H_deconvolved' — respective wT covariance times 1200
          'wpl_Fc', 'wpl_Fe' — WPL-corrected fluxes
        Horst uses analytical high-frequency factors times numerical block-only
        factors, a separable approximation. Array deconvolution uses the same
        selected physical transfer functions for either method; it is a different
        estimator from model scaling. Transfer functions are floored at 1e-10.
    """
    if method not in {"massman", "horst"}:
        raise ValueError(f"Unknown method: {method}")
    z_eff = site_config.z_eff

    for res in results:
        flags = res.qc_flags
        owned = {f"{prefix}{flux}{suffix}"
                 for flux in ("wT", "wu", "wCO2", "wH2O")
                 for prefix, suffix in (("cf_", ""), ("cov_", "_corrected"),
                                        ("cov_", "_deconvolved"))}
        owned.update(("H_corrected", "H_deconvolved"))
        for key in list(flags):
            if key in owned or key.startswith(("wpl_", "spectral_")):
                del flags[key]
        res.corrected_spectra.clear()
        enabled = apply_low_freq or apply_high_freq
        valid_wind = np.isfinite(res.u_mean) and res.u_mean >= 0.5
        flags.update(
            spectral_method=method,
            spectral_low_freq=apply_low_freq,
            spectral_high_freq=apply_high_freq,
            spectral_low_response="block_average" if apply_low_freq else "none",
            spectral_status=("skipped_invalid_wind" if enabled and not valid_wind
                             else "applied" if enabled else "disabled"),
        )
        for flux_type in ["wT", "wu", "wCO2", "wH2O"]:
            if not enabled:
                cf = 1.0
            elif not valid_wind:
                cf = np.nan
            elif method == "massman":
                cf = compute_spectral_correction_factor(
                    res.u_mean, z_eff, instrument, site_config.averaging_period,
                    flux_type, apply_low_freq=apply_low_freq,
                    apply_high_freq=apply_high_freq,
                )
            else:
                # Separable approximation: analytical HF times numerical LF.
                cf = 1.0
                if apply_high_freq:
                    tau_eff = _sensor_tau_for_flux(flux_type, instrument)
                    tau_path = instrument.irga_path_length / (2.0 * np.pi * res.u_mean)
                    tau_eff = np.sqrt(tau_eff**2 + tau_path**2)
                    cf = horst_analytical_correction(res.u_mean, z_eff, tau_eff, flux_type)
                if apply_low_freq:
                    cf *= compute_spectral_correction_factor(
                        res.u_mean, z_eff, instrument, site_config.averaging_period,
                        flux_type, apply_low_freq=True, apply_high_freq=False,
                    )
            flags[f"cf_{flux_type}"] = cf
            flags[f"cov_{flux_type}_corrected"] = getattr(res, f"cov_{flux_type}") * cf
            flags[f"cov_{flux_type}_deconvolved"] = np.nan
            cosp = getattr(res, f"cosp_{flux_type}")
            if not np.isfinite(cf) or len(cosp) == 0:
                continue
            if (len(cosp) != len(res.freq) or np.any(~np.isfinite(res.freq))
                    or np.any(res.freq <= 0) or np.any(np.diff(res.freq) <= 0)):
                raise ValueError("Correction frequencies must be positive, increasing and match cospectra")
            transfer = combined_transfer_function(
                res.freq, res.u_mean, instrument, site_config.averaging_period,
                flux_type, apply_low_freq=apply_low_freq,
                apply_high_freq=apply_high_freq,
            )
            corrected = cosp / transfer
            # Integrate binned n Co(n) over ln(n), independently of model CF.
            widths = np.diff(np.log(res.freq))
            areas = 0.5 * (corrected[:-1] + corrected[1:]) * widths
            ogive = np.concatenate((np.cumsum(areas[::-1])[::-1], [0.0]))
            cov = float(ogive[0]) if len(corrected) > 1 else np.nan
            flags[f"cov_{flux_type}_deconvolved"] = cov
            res.corrected_spectra[f"cosp_{flux_type}"] = corrected
            res.corrected_spectra[f"ogive_{flux_type}"] = ogive
            res.corrected_spectra[f"ncosp_{flux_type}"] = (
                corrected / cov if np.isfinite(cov) and abs(cov) > 1e-12
                else np.full_like(corrected, np.nan)
            )

        # Same fixed volumetric heat capacity as core.process_interval.
        flags["H_corrected"] = flags["cov_wT_corrected"] * 1200.0
        flags["H_deconvolved"] = flags["cov_wT_deconvolved"] * 1200.0
        flags["wpl_status"] = "disabled"
        flags["wpl_pressure_source"] = "not_used"
        flags["wpl_missing_prerequisites"] = ""
        if apply_wpl and instrument.irga_type != "open_path":
            flags["wpl_status"] = "not_applicable"
        elif apply_wpl:
            prerequisites = {
                "co2_mean": res.co2_mean, "h2o_mean": res.h2o_mean,
                "T_mean": res.T_mean, "cov_wCO2_corrected": flags["cov_wCO2_corrected"],
                "cov_wH2O_corrected": flags["cov_wH2O_corrected"],
                "H_corrected": flags["H_corrected"],
            }
            missing = [k for k, v in prerequisites.items() if v is None or not np.isfinite(v)]
            if missing:
                flags["wpl_status"] = "missing_prerequisites"
                flags["wpl_missing_prerequisites"] = ",".join(missing)
            else:
                pressure = res.P_mean
                if pressure is None or not np.isfinite(pressure):
                    pressure = 101.3
                    flags["wpl_pressure_source"] = "standard_atmosphere_fallback"
                else:
                    flags["wpl_pressure_source"] = "measured"
                flags["wpl_pressure_kpa"] = pressure
                if (res.T_mean + T_ZERO_C <= 0 or pressure <= 0
                        or res.h2o_mean < 0 or res.co2_mean < 0
                        or pressure * 1000 / (R_SPECIFIC["dry_air"] * (res.T_mean + T_ZERO_C))
                        <= res.h2o_mean * 1e-3):
                    flags["wpl_status"] = "invalid_prerequisites"
                else:
                    wpl = wpl_correction(
                        flags["cov_wCO2_corrected"], flags["cov_wH2O_corrected"],
                        flags["H_corrected"], res.T_mean, pressure, res.co2_mean, res.h2o_mean,
                    )
                    flags["wpl_status"] = "applied"
                    flags["wpl_Fc"] = wpl["Fc_wpl"]
                    flags["wpl_Fe"] = wpl["Fe_wpl"]
                    flags["wpl_Fc_correction"] = wpl["Fc_correction"]
                    flags["wpl_Fe_correction"] = wpl["Fe_correction"]

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
