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

from . import cleaning, covariance, thermo
from .config import FluxConfig
from .corrections import webb_pearman_leuning
from .frame_utils import normalize_input_frame
from .wind import determine_wind_dir


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


# ---------------------------------------------------------------------------
# Micrometeorological helpers (ported from legacy CalcFlux class)
# ---------------------------------------------------------------------------
def _shadow_correction(Ux, Uy, Uz, n_iter: int = 4):
    """CSAT3 transducer-shadow correction (Horst, Wilczak & Cook 2015)."""
    h = np.array([
        [ 0.25,  0.4330127018922193,  0.8660254037844386],
        [-0.5,   0.0,                 0.8660254037844386],
        [ 0.25, -0.4330127018922193,  0.8660254037844386],
    ])
    hinv = np.array([
        [ 0.6666666666666666, -1.3333333333333333,  0.6666666666666666],
        [ 1.1547005383792517,  0.0,                -1.1547005383792517],
        [ 0.38490017945975047, 0.38490017945975047, 0.38490017945975047],
    ])
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


def _coord_rotation(Ux, Uy, Uz):
    """Double rotation (yaw + pitch) — Kaimal & Finnigan 1994."""
    xmean = float(np.nanmean(Ux))
    ymean = float(np.nanmean(Uy))
    zmean = float(np.nanmean(Uz))
    Uxy = np.sqrt(xmean**2 + ymean**2)
    Uxyz = np.sqrt(xmean**2 + ymean**2 + zmean**2)
    if Uxy < 1e-9 or Uxyz < 1e-9:
        return 1.0, 0.0, 0.0, 1.0, Uxy, Uxyz
    cosv = xmean / Uxy
    sinv = ymean / Uxy
    sinTheta = zmean / Uxyz
    cosTheta = Uxy / Uxyz
    return cosv, sinv, sinTheta, cosTheta, Uxy, Uxyz


def _rotate_velocities(Ux, Uy, Uz, cosv, sinv, sinTheta, cosTheta):
    Uxr = Ux * cosTheta * cosv + Uy * cosTheta * sinv + Uz * sinTheta
    Uyr = Uy * cosv - Ux * sinv
    Uzr = Uz * cosTheta - Ux * sinTheta * cosv - Uy * sinTheta * sinv
    return Uxr, Uyr, Uzr


def _rotate_covariances(covar, errvals, cosv, sinv, sinTheta, cosTheta,
                         scalar_key: str = "Ts"):
    """Rotate scalar and momentum covariances into the streamline frame."""
    cov = dict(covar)

    Ux_s = cov.get(f"Ux-{scalar_key}", 0.0)
    Uy_s = cov.get(f"Uy-{scalar_key}", 0.0)
    Uz_s = cov.get(f"Uz-{scalar_key}", 0.0)
    cov[f"Uz-{scalar_key}"] = (
        Uz_s * cosTheta - Ux_s * sinTheta * cosv - Uy_s * sinTheta * sinv
    )

    for key in ("pV", "Sd"):
        Ux_k = cov.get(f"Ux-{key}", 0.0)
        Uy_k = cov.get(f"Uy-{key}", 0.0)
        Uz_k = cov.get(f"Uz-{key}", 0.0)
        cov[f"Uz-{key}"] = (
            Uz_k * cosTheta - Ux_k * sinTheta * cosv - Uy_k * sinTheta * sinv
        )

    # Momentum covariances (Kaimal & Finnigan 1994, eq. 6.36)
    Ux_Uz = cov.get("Ux-Uz", 0.0)
    Uy_Uz = cov.get("Uy-Uz", 0.0)
    Ux_Uy = cov.get("Ux-Uy", 0.0)
    err_Ux = errvals.get("Ux", 0.0)
    err_Uy = errvals.get("Uy", 0.0)
    err_Uz = errvals.get("Uz", 0.0)

    Ux_Uz_rot = (
        Ux_Uz * cosv * (cosTheta**2 - sinTheta**2)
        - 2.0 * Ux_Uy * sinTheta * cosTheta * sinv * cosv
        + Uy_Uz * sinv * (cosTheta**2 - sinTheta**2)
        - err_Ux * sinTheta * cosTheta * cosv**2
        - err_Uy * sinTheta * cosTheta * sinv**2
        + err_Uz * sinTheta * cosTheta
    )
    Uy_Uz_rot = (
        Uy_Uz * cosTheta * cosv
        - Ux_Uz * cosTheta * sinv
        - Ux_Uy * sinTheta * (cosv**2 - sinv**2)
        + err_Ux * sinTheta * sinv * cosv
        - err_Uy * sinTheta * sinv * cosv
    )
    cov["Ux-Uz"] = Ux_Uz_rot
    cov["Uy-Uz"] = Uy_Uz_rot
    cov["Uxy-Uz"] = np.sqrt(Ux_Uz_rot**2 + Uy_Uz_rot**2)
    return cov


def _calc_L(Ustr: float, Tsa: float, Uz_Ta: float, g: float, kappa: float) -> float:
    if abs(Uz_Ta) < 1e-12 or Ustr <= 0:
        return np.inf
    return -(Ustr**3) * Tsa / (g * kappa * Uz_Ta)


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


def _correct_kh20_oxygen(Uz_Ta: float, P: float, T: float, config: FluxConfig) -> float:
    """Oren et al. (1998) O2 cross-sensitivity correction for KH-20."""
    return ((config.Co * config.Mo * P) / (config.Ru * T**2)) * (config.Ko / config.Kw) * Uz_Ta


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
            df[col + suffix] = cleaning.despike_med_mod(df[col])
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
    Ux, Uy, Uz = _shadow_correction(Ux_raw, Uy_raw, Uz_raw)
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
    cosv, sinv, sinTheta, cosTheta, Uxy, Uxyz = _coord_rotation(Ux, Uy, Uz)
    Uxr, Uyr, Uzr = _rotate_velocities(Ux, Uy, Uz, cosv, sinv, sinTheta, cosTheta)
    errvals = {"Ux": covariance.calc_MSE(Uxr),
               "Uy": covariance.calc_MSE(Uyr),
               "Uz": covariance.calc_MSE(Uzr),
               "Q":  covariance.calc_MSE(Q)}

    covar = _rotate_covariances(covar, errvals, cosv, sinv, sinTheta, cosTheta)
    Uxy_Uz = covar["Uxy-Uz"]
    Ustr = float(np.sqrt(Uxy_Uz)) if Uxy_Uz > 0 else 0.0

    # 4. Sonic-adjusted air temperature (Wallace & Hobbs / Campbell EasyFlux).
    Tsa = thermo.calc_Tsa_sonic_temp(Ts_mean, Pr_mean, pV_mean, Rv=config.Rv)
    lamb = 2500800.0 - 2366.8 * thermo.convert_KtoC(Tsa)

    # Std of air temperature from sonic-temperature variance.
    StDevTa = float(
        np.sqrt(np.abs(
            covar["Ts-Ts"]
            - 1.02 * Ts_mean * covar["Ts-Q"]
            - 0.2601 * errvals["Q"] * Ts_mean**2
        ))
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

    values = [Tsa_C, Td_C, D, Ustr, zeta, H, StDevUz, StDevTa,
              direction, exchange, lambdaE, ET, Uxy]
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

    for raw, canon in (("Ux_ro", "Ux"), ("Uy_ro", "Uy"), ("Uz_ro", "Uz"),
                        ("Ts_ro", "Ts"), ("pV_ro", "pV"), ("Pr_ro", "Pr")):
        if raw in df.columns:
            df[canon] = df[raw]

    # Unit conversions: Ts °C→K, Pr kPa→Pa, pV g m⁻³→kg m⁻³.
    df["Ts"] = thermo.convert_CtoK(df["Ts"].to_numpy(dtype=float))
    df["Pr"] = df["Pr"].to_numpy(dtype=float) * 1000.0
    df["pV"] = df["pV"].to_numpy(dtype=float) * 1e-3

    # Derived thermodynamic columns.
    df["E"] = thermo.calc_E(df["pV"].to_numpy(), df["Ts"].to_numpy(), Rv=config.Rv)
    df["Q"] = thermo.calc_Q(df["Pr"].to_numpy(), df["E"].to_numpy(), epsilon=config.epsilon)
    df["Tsa"] = thermo.calc_Tsa(df["Ts"].to_numpy(), df["Q"].to_numpy())
    df["Sd"] = (
        thermo.calc_Q(df["Pr"].to_numpy(), thermo.calc_Es(df["Tsa"].to_numpy()),
                      epsilon=config.epsilon)
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
        pV_calib = thermo.calc_pV(df["Ea"].to_numpy(dtype=float) * 1000.0,
                                  thermo.convert_CtoK(df["Ts"].to_numpy(dtype=float)),
                                  Rv=config.Rv)
        XKw = config.XKwC1 + 2.0 * config.XKwC2 * (float(np.nanmean(pV_calib)) * 1000.0)
        config.Kw = XKw / config.XKH20

    df = _despike_columns(df, config.despikefields, suffix="_ro")
    for raw, canon in (("Ux_ro", "Ux"), ("Uy_ro", "Uy"), ("Uz_ro", "Uz"),
                        ("Ts_ro", "Ts"), ("Pr_ro", "Pr")):
        if raw in df.columns:
            df[canon] = df[raw]

    df["Ts"] = thermo.convert_CtoK(df["Ts"].to_numpy(dtype=float))
    if "Ta" in df.columns:
        df["Ta"] = thermo.convert_CtoK(df["Ta"].to_numpy(dtype=float))
    df["Pr"] = df["Pr"].to_numpy(dtype=float) * 1000.0

    # Derived thermodynamics.
    df["pV"] = thermo.calc_pV(df["Ea"].to_numpy(dtype=float) * 1000.0,
                              df["Ts"].to_numpy(), Rv=config.Rv)
    df["Tsa"] = thermo.calc_Tsa_sonic_temp(df["Ts"].to_numpy(),
                                           df["Pr"].to_numpy(),
                                           df["pV"].to_numpy(),
                                           Rv=config.Rv)
    df["E"] = thermo.calc_E(df["pV"].to_numpy(), df["Tsa"].to_numpy(), Rv=config.Rv)
    df["Q"] = thermo.calc_Q(df["Pr"].to_numpy(), df["E"].to_numpy(), epsilon=config.epsilon)
    df["Sd"] = (
        thermo.calc_Q(df["Pr"].to_numpy(), thermo.calc_Es(df["Tsa"].to_numpy()),
                      epsilon=config.epsilon)
        - df["Q"].to_numpy()
    )

    setattr(config, "apply_kh20_oxygen", True)
    return _compute_fluxes(df, config)
