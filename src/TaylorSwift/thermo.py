from __future__ import annotations

import numpy as np

from .constants import R_SPECIFIC, T_ZERO_C


def convert_KtoC(T):
    """Convert temperature from Kelvin to Celsius.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Temperature in Celsius.
    """
    return T - T_ZERO_C


def convert_CtoK(T):
    """Convert temperature from Celsius to Kelvin.

    Parameters
    ----------
    T : float
        Temperature in Celsius.

    Returns
    -------
    float
        Temperature in Kelvin.
    """
    return T + T_ZERO_C


def tetens(t, a: float = 0.611, b: float = 17.502, c: float = 240.97):
    """Tetens formula for saturation vapor pressure.

    Parameters
    ----------
    t : float
        Temperature in Celsius.
    a : float, optional
        The a parameter in the Tetens formula. Default is 0.611.
    b : float, optional
        The b parameter in the Tetens formula. Default is 17.502.
    c : float, optional
        The c parameter in the Tetens formula. Default is 240.97.

    Returns
    -------
    float
        Saturation vapor pressure in hPa.
    """
    return a * np.exp((b * t) / (t + c))


def calc_E(pV, T, Rv: float = R_SPECIFIC["water_vapor"]):
    """Calculate vapor pressure from partial pressure and temperature.

    Parameters
    ----------
    pV : float
        Partial pressure of water vapor.
    T : float
        Temperature in Kelvin.
    Rv : float, optional
        Specific gas constant for water vapor. Default is 461.51.

    Returns
    -------
    float
        Vapor pressure.
    """
    return pV * Rv * T


def calc_Q(P, E, epsilon: float = 18.016 / 28.97):
    """Calculate specific humidity from total pressure and vapor pressure.

    Parameters
    ----------
    P : float
        Total pressure.
    E : float
        Vapor pressure.
    epsilon : float, optional
        The ratio of the molecular weight of water vapor to that of dry air. Default is 18.016 / 28.97.

    Returns
    -------
    float
        Specific humidity.
    """
    return (epsilon * E) / (P - (1.0 - epsilon) * E)


def calc_pV(E, T, Rv: float = R_SPECIFIC["water_vapor"]):
    """Calculate partial pressure of water vapor from vapor pressure and temperature.
    Parameters
    ----------
    E : float
        Vapor pressure.
    T : float
        Temperature in Kelvin.
    Rv : float, optional
        Specific gas constant for water vapor. Default is 461.51.

    Returns
    -------
    float
        Partial pressure of water vapor.
    """
    return E / (Rv * T)


def calc_Tsa(Ts, Q):
    """Calculate sonic temperature from static temperature and specific humidity.

    Parameters
    ----------
    Ts : float
        Static temperature in Kelvin.
    Q : float
        Specific humidity.

    Returns
    -------
    float
        Sonic temperature in Kelvin.
    """
    return Ts / (1.0 + 0.51 * Q)


def calc_Tsa_sonic_temp(Ts, P, pV, Rv: float = R_SPECIFIC["water_vapor"]):
    """Calculate sonic temperature from static temperature, total pressure, and partial pressure of water vapor.

    Parameters
    ----------
    Ts : float
        Static temperature in Kelvin.
    P : float
        Total pressure.
    pV : float
        Partial pressure of water vapor.
    Rv : float, optional
        Specific gas constant for water vapor. Default is 461.51.

    Returns
    -------
    float
        Sonic temperature in Kelvin.
    """
    E = calc_E(pV, Ts, Rv=Rv)
    Q = calc_Q(P, E)
    return calc_Tsa(Ts, Q)


def calc_Es(T):
    """Calculate saturation vapor pressure from temperature.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Saturation vapor pressure.
    """
    Tc = convert_KtoC(T)
    return tetens(Tc) * 1000.0


def calc_Td_dewpoint(E):
    """Calculate dew point temperature from vapor pressure.

    Parameters
    ----------
    E : float
        Vapor pressure.

    Returns
    -------
    float
        Dew point temperature in Kelvin.
    """
    e_kpa = np.asarray(E) / 1000.0
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_ratio = np.log(e_kpa / 0.611)
        td_c = (240.97 * ln_ratio) / (17.502 - ln_ratio)
    return convert_CtoK(td_c)


def latent_heat_vaporization(Tsa_K):
    """Calculate the latent heat of vaporization of water as a function of sonic temperature.
    Parameters
    ----------
    Tsa_K : float
        Sonic temperature in Kelvin.

    Returns
    -------
    float
        Latent heat of vaporization in J/kg.
    """
    return 2500800.0 - 2366.8 * convert_KtoC(Tsa_K)


def get_watts_to_h2o_conversion_factor(Ta_C, duration_days: float) -> float:
    """Calculate the conversion factor from watts to kg H2O for a given air temperature and duration.

    Parameters
    ----------
    Ta_C : float
        Air temperature in Celsius.
    duration_days : float
        Duration in days.

    Returns
    -------
    float
        Conversion factor from watts to kg H2O.
    """
    lamb = latent_heat_vaporization(convert_CtoK(Ta_C))
    seconds = duration_days * 86400.0
    return seconds / lamb
