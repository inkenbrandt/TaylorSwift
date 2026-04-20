"""
pipelines.py — End-to-end eddy-covariance flux pipelines.

Two pipelines are provided, one per common open-path deployment:

* :func:`run_irga`  – integrated IRGASON / LI-7500 + CSAT3 (water-vapour
  density measured directly in g m⁻³).
* :func:`run_kh20`  – KH-20 krypton hygrometer + CSAT3 (water-vapour
  density derived from the hygrometer signal).

Both routines apply despiking, CSAT3 shadow correction, double coordinate
rotation, lag-optimised covariances, Massman (2000, 2001) spectral
corrections, the Webb–Pearman–Leuning (1980) density correction, and
return a 13-element :class:`pandas.Series` of mean fluxes / diagnostics
for the averaging period represented by *df*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import covariance, despike, thermo
from .constants import _calc_L
from .config import FluxConfig
from .corrections import webb_pearman_leuning, shadow_correction
from .cospectra import _correct_spectral, _calc_alph_x
from .frame_utils import normalize_input_frame
from .rotations import (
    determine_wind_dir,
    coord_rotation,
    rotate_velocities,
    rotate_covariances,
)


# ---------------------------------------------------------------------------
# Column aliases from common datalogger exports → canonical names used here
# ---------------------------------------------------------------------------
_DEFAULT_RENAME_MAP = {
    "T_SONIC": "Ts",
    "TA_1_1_1": "Ta",
    "amb_press": "Pr",
    "PA": "Pr",
    "H2O_density": "pV",
    "RH_1_1_1": "Rh",
    "t_hmp": "Ta",
    "e_hmp": "Ea",
    "kh": "volt_KH20",
    "q": "Q",
}

_OUTPUT_COLUMNS = [
    "Ta",
    "Td",
    "D",
    "Ustr",
    "zeta",
    "H",
    "StDevUz",
    "StDevTa",
    "direction",
    "exchange",
    "lambdaE",
    "ET",
    "Uxy",
]


def _correct_kh20_oxygen(Uz_Ta: float, P: float, T: float, config: FluxConfig) -> float:
    """Oren et al. (1998) O2 cross-sensitivity correction for KH-20."""
    return (
        ((config.Co * config.Mo * P) / (config.Ru * T**2))
        * (config.Ko / config.Kw)
        * Uz_Ta
    )


def _max_cov_value(x, y, lag: int) -> float:
    result = covariance.calc_max_covariance(x, y, lag=lag)
    if result:
        return float(result[0][1])
    return float(covariance.calc_cov(x, y))


def _duration_days(df: pd.DataFrame) -> float:
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
        span = df.index[-1] - df.index[0]
        return span.total_seconds() / 86400.0
    for name in ("TIMESTAMP", "timestamp", "time", "datetime"):
        if name in df.columns:
            col = pd.to_datetime(df[name], errors="coerce").dropna()
            if len(col) > 1:
                return (col.iloc[-1] - col.iloc[0]).total_seconds() / 86400.0
            break
    return 30.0 / (60.0 * 24.0)  # default to 30-min averaging period


def _apply_rename(df: pd.DataFrame, rename_map: dict | None) -> pd.DataFrame:
    mapping = dict(_DEFAULT_RENAME_MAP)
    if rename_map:
        mapping.update(rename_map)
    return df.rename(columns=mapping)


def _despike_columns(df: pd.DataFrame, columns: list[str], suffix: str) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col + suffix] = despike.despike_med_mod(df[col])
    return df


# ---------------------------------------------------------------------------
# Core flux computation (shared by both pipelines once df has canonical cols)
# ---------------------------------------------------------------------------
def _compute_fluxes(df: pd.DataFrame, config: FluxConfig) -> pd.Series:
    # Required columns after harmonisation: Ux, Uy, Uz, Ts (K), pV (kg m-3),
    # Pr (Pa), E, Q, Tsa, Sd.
    Ux_raw = df["Ux"].to_numpy(dtype=float)
    Uy_raw = df["Uy"].to_numpy(dtype=float)
    Uz_raw = df["Uz"].to_numpy(dtype=float)

    # 1. Remove CSAT shadow effect.
    Ux, Uy, Uz = shadow_correction(Ux_raw, Uy_raw, Uz_raw)
    df["Ux"], df["Uy"], df["Uz"] = Ux, Uy, Uz

    Ts = df["Ts"].to_numpy(dtype=float)
    pV = df["pV"].to_numpy(dtype=float)
    Q = df["Q"].to_numpy(dtype=float)
    Sd = df["Sd"].to_numpy(dtype=float)

    Ts_mean = float(np.nanmean(Ts))
    Pr_mean = float(np.nanmean(df["Pr"].to_numpy(dtype=float)))
    pV_mean = float(np.nanmean(pV))
    E_mean = float(np.nanmean(df["E"].to_numpy(dtype=float)))
    Q_mean = float(np.nanmean(Q))

    # 2. Raw covariances (lag-optimised via max-covariance search).
    lag = config.lag
    variables = {"Ux": Ux, "Uy": Uy, "Uz": Uz, "Ts": Ts, "pV": pV, "Q": Q, "Sd": Sd}
    covar: dict[str, float] = {}
    for vel in ("Ux", "Uy", "Uz"):
        for name, arr in variables.items():
            covar[f"{vel}-{name}"] = _max_cov_value(variables[vel], arr, lag)
    covar["Ts-Ts"] = covariance.calc_cov(Ts, Ts)
    covar["Ts-Q"] = _max_cov_value(Ts, Q, lag)
    covar["Ux-Uy"] = _max_cov_value(Ux, Uy, lag)

    # 3. Coordinate rotation + rotated-frame statistics.
    cosv, sinv, sinTheta, cosTheta, Uxy, Uxyz = coord_rotation(Ux, Uy, Uz)
    Uxr, Uyr, Uzr = rotate_velocities(Ux, Uy, Uz, cosv, sinv, sinTheta, cosTheta)
    errvals = {
        "Ux": covariance.calc_MSE(Uxr),
        "Uy": covariance.calc_MSE(Uyr),
        "Uz": covariance.calc_MSE(Uzr),
        "Q": covariance.calc_MSE(Q),
    }

    covar = rotate_covariances(covar, errvals, cosv, sinv, sinTheta, cosTheta)
    Uxy_Uz = covar["Uxy-Uz"]
    Ustr = float(np.sqrt(Uxy_Uz)) if Uxy_Uz > 0 else 0.0

    # 4. Sonic-adjusted air temperature (Wallace & Hobbs / Campbell EasyFlux).
    Tsa = thermo.calc_Tsa_sonic_temp(Ts_mean, Pr_mean, pV_mean, Rv=config.Rv)
    lamb = 2500800.0 - 2366.8 * thermo.convert_KtoC(Tsa)

    # Std of air temperature from sonic-temperature variance.
    StDevTa = float(
        np.sqrt(
            np.abs(
                covar["Ts-Ts"]
                - 1.02 * Ts_mean * covar["Ts-Q"]
                - 0.2601 * errvals["Q"] * Ts_mean**2
            )
        )
    )

    # 5. Moist-air heat capacity and densities.
    Cp = config.Cpd * (1.0 + 0.84 * Q_mean)
    pD = (Pr_mean - E_mean) / (config.Rd * Tsa)
    p_rho = pD + pV_mean

    # 6. Sensible-heat kinematic flux corrected for latent-heat contribution.
    Uz_Ta = covar["Uz-Ts"] - 0.07 * lamb * covar["Uz-pV"] / (p_rho * Cp)

    # 7. Dew point, VPD, Clausius-Clapeyron slope.
    Td = thermo.calc_Td_dewpoint(E_mean)
    D = thermo.calc_Es(Tsa) - E_mean
    S = (
        thermo.calc_Q(Pr_mean, thermo.calc_Es(Tsa + 1.0))
        - thermo.calc_Q(Pr_mean, thermo.calc_Es(Tsa - 1.0))
    ) / 2.0

    # 8. Wind direction — computed from raw (shadow-corrected) means.
    pathlen, direction = determine_wind_dir(
        float(np.nanmean(Ux)),
        float(np.nanmean(Uy)),
        config.sonic_dir,
        config.PathDist_U,
    )
    StDevUz = float(np.nanstd(Uzr, ddof=1))
    UMean = max(float(np.nanmean(Uxr)), 1e-6)

    # 9. Massman spectral corrections.
    tauB = 3600.0 / 2.8
    tauEKH20 = np.sqrt((0.01 / (4.0 * UMean)) ** 2 + (pathlen / (1.1 * UMean)) ** 2)
    tauETs = np.sqrt((0.1 / (8.4 * UMean)) ** 2)
    tauEMom = np.sqrt((0.1 / (5.7 * UMean)) ** 2 + (0.1 / (2.8 * UMean)) ** 2)

    L = _calc_L(Ustr, Tsa, Uz_Ta, config.g, config.von_karman)
    alpha, X = _calc_alph_x(config.UHeight, L)
    fX = X * UMean / config.UHeight
    B_par = 2.0 * np.pi * fX * tauB
    Ts_corr = _correct_spectral(B_par, alpha, 2.0 * np.pi * fX * tauETs)
    Mom_corr = _correct_spectral(B_par, alpha, 2.0 * np.pi * fX * tauEMom)
    KH_corr = _correct_spectral(B_par, alpha, 2.0 * np.pi * fX * tauEKH20)

    if Mom_corr > 0:
        Uxy_Uz /= Mom_corr
        Ustr = float(np.sqrt(Uxy_Uz)) if Uxy_Uz > 0 else Ustr

    # Recompute L with first-pass-corrected Uz_Ta.
    Uz_Ta_cor = Uz_Ta / Ts_corr if Ts_corr > 0 else Uz_Ta
    L = _calc_L(Ustr, Tsa, Uz_Ta_cor, config.g, config.von_karman)

    covar["Uz-pV"] = covar["Uz-pV"] / KH_corr if KH_corr > 0 else covar["Uz-pV"]
    covar["Uz-Sd"] = covar["Uz-Sd"] / KH_corr if KH_corr > 0 else covar["Uz-Sd"]
    exchange = ((p_rho * Cp) / (S + Cp / lamb)) * covar["Uz-Sd"]

    # 10. KH-20 oxygen correction (safe for IRGASON: Kw=1, Ko=-0.0045 ~
    #     negligible). Applied only when the caller flagged KH-20 hardware.
    if getattr(config, "apply_kh20_oxygen", False):
        covar["Uz-pV"] += _correct_kh20_oxygen(Uz_Ta_cor, Pr_mean, Tsa, config)

    # 11. Fluxes.
    H = p_rho * Cp * Uz_Ta_cor
    lambdaE = webb_pearman_leuning(
        lamb=lamb,
        Tsa=Tsa,
        pVavg=pV_mean,
        Uz_Ta=Uz_Ta_cor,
        Uz_pV=covar["Uz-pV"],
        p=p_rho,
        Cp=Cp,
        pD=pD,
    )

    Tsa_C = thermo.convert_KtoC(Tsa)
    Td_C = thermo.convert_KtoC(Td)
    zeta = config.UHeight / L if np.isfinite(L) and L != 0 else 0.0
    duration = _duration_days(df)
    ET = lambdaE * thermo.get_watts_to_h2o_conversion_factor(Tsa_C, duration)

    values = [
        Tsa_C,
        Td_C,
        D,
        Ustr,
        zeta,
        H,
        StDevUz,
        StDevTa,
        direction,
        exchange,
        lambdaE,
        ET,
        Uxy,
    ]
    return pd.Series(values, index=_OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# IRGASON / integrated-sensor pipeline
# ---------------------------------------------------------------------------
def run_irga(df, config: FluxConfig, *, rename_map=None, ts_col=None) -> pd.Series:
    """
    Run the full EC processing chain for an integrated open-path IRGA + CSAT3
    system (e.g. Campbell *IRGASON*, LI-COR LI-7500 + CSAT3).

    Expected canonical columns after rename: ``Ux``, ``Uy``, ``Uz`` (m s⁻¹),
    ``Ts`` (°C sonic virtual temp), ``Pr`` (kPa), ``pV`` (water-vapour
    density; g m⁻³ from an IRGASON is converted internally to kg m⁻³).

    Returns a 13-element :class:`pandas.Series` (``Ta``, ``Td``, ``D``,
    ``Ustr``, ``zeta``, ``H``, ``StDevUz``, ``StDevTa``, ``direction``,
    ``exchange``, ``lambdaE``, ``ET``, ``Uxy``).
    """
    df = normalize_input_frame(df, rename_map=rename_map, ts_col=ts_col)
    df = _apply_rename(df, rename_map).copy()

    df = _despike_columns(df, config.despikefields, suffix="_ro")

    for raw, canon in (
        ("Ux_ro", "Ux"),
        ("Uy_ro", "Uy"),
        ("Uz_ro", "Uz"),
        ("Ts_ro", "Ts"),
        ("pV_ro", "pV"),
        ("Pr_ro", "Pr"),
    ):
        if raw in df.columns:
            df[canon] = df[raw]

    # Unit conversions: Ts °C→K, Pr kPa→Pa, pV g m⁻³→kg m⁻³.
    df["Ts"] = thermo.convert_CtoK(df["Ts"].to_numpy(dtype=float))
    df["Pr"] = df["Pr"].to_numpy(dtype=float) * 1000.0
    df["pV"] = df["pV"].to_numpy(dtype=float) * 1e-3

    # Derived thermodynamic columns.
    df["E"] = thermo.calc_E(df["pV"].to_numpy(), df["Ts"].to_numpy(), Rv=config.Rv)
    df["Q"] = thermo.calc_Q(
        df["Pr"].to_numpy(), df["E"].to_numpy(), epsilon=config.epsilon
    )
    df["Tsa"] = thermo.calc_Tsa(df["Ts"].to_numpy(), df["Q"].to_numpy())
    df["Sd"] = (
        thermo.calc_Q(
            df["Pr"].to_numpy(),
            thermo.calc_Es(df["Tsa"].to_numpy()),
            epsilon=config.epsilon,
        )
        - df["Q"].to_numpy()
    )

    # IRGASON is an integrated sensor — no KH-20 oxygen correction.
    setattr(config, "apply_kh20_oxygen", False)
    return _compute_fluxes(df, config)


# ---------------------------------------------------------------------------
# KH-20 pipeline
# ---------------------------------------------------------------------------
def run_kh20(df, config: FluxConfig, *, rename_map=None, ts_col=None) -> pd.Series:
    """
    Run the full EC processing chain for a CSAT3 + KH-20 krypton-hygrometer
    deployment (non-integrated sensors).

    Required inputs after rename: ``Ux``, ``Uy``, ``Uz``, ``Ts`` (°C),
    ``Ta`` (°C), ``Pr`` (kPa). Vapour pressure ``Ea`` (kPa) is derived from
    ``Ta`` via Tetens if absent; KH-20 output ``volt_KH20`` (mV) — if
    supplied — is log-transformed for the *Kw* calibration.
    """
    df = normalize_input_frame(df, rename_map=rename_map, ts_col=ts_col)
    df = _apply_rename(df, rename_map).copy()

    if "Ea" not in df.columns and "Ta" in df.columns:
        df["Ea"] = thermo.tetens(df["Ta"].to_numpy(dtype=float))

    if "LnKH" not in df.columns and "volt_KH20" in df.columns:
        df["LnKH"] = np.log(df["volt_KH20"].to_numpy(dtype=float))

    # Kw calibration from mean vapour density (Ea kPa → Pa for consistency).
    if "Ea" in df.columns:
        pV_calib = thermo.calc_pV(
            df["Ea"].to_numpy(dtype=float) * 1000.0,
            thermo.convert_CtoK(df["Ts"].to_numpy(dtype=float)),
            Rv=config.Rv,
        )
        XKw = config.XKwC1 + 2.0 * config.XKwC2 * (float(np.nanmean(pV_calib)) * 1000.0)
        config.Kw = XKw / config.XKH20

    df = _despike_columns(df, config.despikefields, suffix="_ro")
    for raw, canon in (
        ("Ux_ro", "Ux"),
        ("Uy_ro", "Uy"),
        ("Uz_ro", "Uz"),
        ("Ts_ro", "Ts"),
        ("Pr_ro", "Pr"),
    ):
        if raw in df.columns:
            df[canon] = df[raw]

    df["Ts"] = thermo.convert_CtoK(df["Ts"].to_numpy(dtype=float))
    if "Ta" in df.columns:
        df["Ta"] = thermo.convert_CtoK(df["Ta"].to_numpy(dtype=float))
    df["Pr"] = df["Pr"].to_numpy(dtype=float) * 1000.0

    # Derived thermodynamics.
    df["pV"] = thermo.calc_pV(
        df["Ea"].to_numpy(dtype=float) * 1000.0, df["Ts"].to_numpy(), Rv=config.Rv
    )
    df["Tsa"] = thermo.calc_Tsa_sonic_temp(
        df["Ts"].to_numpy(), df["Pr"].to_numpy(), df["pV"].to_numpy(), Rv=config.Rv
    )
    df["E"] = thermo.calc_E(df["pV"].to_numpy(), df["Tsa"].to_numpy(), Rv=config.Rv)
    df["Q"] = thermo.calc_Q(
        df["Pr"].to_numpy(), df["E"].to_numpy(), epsilon=config.epsilon
    )
    df["Sd"] = (
        thermo.calc_Q(
            df["Pr"].to_numpy(),
            thermo.calc_Es(df["Tsa"].to_numpy()),
            epsilon=config.epsilon,
        )
        - df["Q"].to_numpy()
    )

    setattr(config, "apply_kh20_oxygen", True)
    return _compute_fluxes(df, config)
