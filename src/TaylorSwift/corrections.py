"""
corrections.py — Spectral and flux corrections for eddy covariance data.

Implements the standard suite of corrections used in the micromet community:

DESPIKING (pre-processing, applied before spectral computation):
  0. Iterative UKDE despiking — kernel density estimation identifies and
     removes statistical outliers from the raw time series before FFTs are
     computed (Metzger et al. 2012).
     • ukde_despike          – scipy Gaussian KDE, iterative (small arrays)
     • polars_ukde_despike   – FFT-KDE via KDEpy, single-pass (large arrays)
     • despike_dataframe     – multi-column wrapper for pandas DataFrames

HIGH-FREQUENCY CORRECTIONS (attenuate flux at high frequencies):
  1. Sensor frequency response — first-order time constant (Moore 1986)
  2. Path averaging / line averaging along sonic paths (Kaimal et al. 1968)
  3. Scalar path averaging for IRGA optical path (Moore 1986)
  4. Sensor separation — lateral/longitudinal displacement between sonic
     and gas analyser (Moore 1986).  For IRGASON this is ~0 (integrated).
  5. Combined transfer function approach (Massman 2000)
  6. Analytical correction factor (Horst 1997)

LOW-FREQUENCY CORRECTIONS (attenuate flux at low frequencies):
  7. Block-average (finite averaging window) transfer function
  8. Linear detrend transfer function

DENSITY CORRECTIONS:
  9. Webb-Pearman-Leuning (WPL 1980) correction for open-path CO₂/H₂O
     fluxes measured as density.

References
----------
Moore, C.J. (1986). Frequency response corrections for eddy correlation
    systems. Boundary-Layer Meteorol., 37, 17–35.
Massman, W.J. (2000). A simple method for estimating frequency response
    corrections for eddy covariance systems. Agric. For. Meteorol., 104,
    185–198.
Massman, W.J. (2001). Reply to comment by Rannik on "A simple method..."
    Agric. For. Meteorol., 107, 247–251.
Horst, T.W. (1997). A simple formula for attenuation of eddy fluxes
    measured with first-order-response scalar sensors. Boundary-Layer
    Meteorol., 82, 219–233.
Kaimal, J.C. et al. (1968). Deriving power spectra from a three-component
    sonic anemometer. J. Appl. Meteorol., 7, 827–837.
Webb, E.K., Pearman, G.I. & Leuning, R. (1980). Correction of flux
    measurements for density effects due to heat and water vapour transfer.
    Quart. J. Roy. Meteor. Soc., 106, 85–100.
Leuning, R. (2007). The correct formula for the WPL correction.
    Boundary-Layer Meteorol., 126, 263–272.
Metzger, S., Junkermann, W., Mauder, M., Beyrich, F., Butterbach-Bahl, K.,
    Schmid, H. P., & Foken, T. (2012). Eddy-covariance flux measurements
    with a weight-shift microlight aircraft. Atmospheric Measurement
    Techniques, 5, 1699–1717.
"""

import numpy as np
from dataclasses import dataclass
import polars as pl
import pandas as pd
from KDEpy import FFTKDE
from scipy.interpolate import interp1d

from .constants import MOLAR_MASS, R_SPECIFIC, CP_DRY_AIR, T_ZERO_C


# ---------------------------------------------------------------------------
# Micrometeorological helpers (ported from legacy CalcFlux class)
# ---------------------------------------------------------------------------
def shadow_correction(Ux, Uy, Uz, n_iter: int = 4):
    """CSAT3 transducer-shadow correction (Horst, Wilczak & Cook 2015)."""
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
) -> dict:
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
# Convenience: store mean scalars during processing
# ===================================================================


def enrich_results_with_means(results, df, site_config):
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
