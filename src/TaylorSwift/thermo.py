from __future__ import annotations
import numpy as np


def convert_KtoC(T):
    return T - 273.16


def convert_CtoK(T):
    return T + 273.16


def tetens(t, a: float = 0.611, b: float = 17.502, c: float = 240.97):
    return a * np.exp((b * t) / (t + c))


def calc_E(pV, T, Rv: float = 461.51):
    return pV * Rv * T


def calc_Q(P, E, epsilon: float = 18.016 / 28.97):
    return (epsilon * E) / (P - (1.0 - epsilon) * E)


def calc_pV(E, T, Rv: float = 461.51):
    return E / (Rv * T)


def calc_Tsa(Ts, Q):
    return Ts / (1.0 + 0.51 * Q)


def calc_Tsa_sonic_temp(Ts, P, pV, Rv: float = 461.51):
    E = calc_E(pV, Ts, Rv=Rv)
    Q = calc_Q(P, E)
    return calc_Tsa(Ts, Q)


def calc_Es(T):
    Tc = convert_KtoC(T)
    return tetens(Tc) * 1000.0


def calc_Td_dewpoint(E):
    e_kpa = np.asarray(E) / 1000.0
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_ratio = np.log(e_kpa / 0.611)
        td_c = (240.97 * ln_ratio) / (17.502 - ln_ratio)
    return convert_CtoK(td_c)


def latent_heat_vaporization(Tsa_K):
    return 2500800.0 - 2366.8 * convert_KtoC(Tsa_K)


def get_watts_to_h2o_conversion_factor(Ta_C, duration_days: float) -> float:
    lamb = latent_heat_vaporization(convert_CtoK(Ta_C))
    seconds = duration_days * 86400.0
    return seconds / lamb
