from __future__ import annotations
from .config import FluxConfig
from .pipelines import run_irga as _run_irga, run_kh20 as _run_kh20
from .wind import determine_wind_dir as _determine_wind_dir
from .corrections import wpl_correction as _wpl
from . import thermo, cleaning, covariance


class CalcFlux:
    def __init__(self, **kwargs):
        self.config = FluxConfig(**kwargs)
        self.avgvals = {}
        self.covar = {}
        self.errvals = {}
        self.wind_compass = None
        self.pathlen = None
        self.p = None
        self.Cp = None
        self.pD = None

    def runall(self, df):
        return _run_kh20(df, self.config)

    def run_irga(self, df, *, rename_map=None, ts_col=None, lazy=False):
        return _run_irga(df, self.config, rename_map=rename_map, ts_col=ts_col)

    def determine_wind_dir(
        self, uxavg=None, uyavg=None, update_existing_vel: bool = False
    ):
        if uxavg is None and "Ux" in self.avgvals:
            uxavg = self.avgvals["Ux"]
        if uyavg is None and "Uy" in self.avgvals:
            uyavg = self.avgvals["Uy"]
        if update_existing_vel:
            if uxavg is not None:
                self.avgvals["Ux"] = uxavg
            if uyavg is not None:
                self.avgvals["Uy"] = uyavg
        pathlen, wind_compass = _determine_wind_dir(
            uxavg, uyavg, self.config.sonic_dir, self.config.PathDist_U
        )
        self.pathlen = pathlen
        self.wind_compass = wind_compass
        return pathlen, wind_compass

    def webb_pearman_leuning(self, lamb, Tsa, pVavg, Uz_Ta, Uz_pV):
        if self.p is None or self.Cp is None or self.pD is None:
            raise ValueError(
                "self.p, self.Cp, and self.pD must be set before calling webb_pearman_leuning."
            )
        return _wpl(lamb, Tsa, pVavg, Uz_Ta, Uz_pV, p=self.p, Cp=self.Cp, pD=self.pD)

    convert_KtoC = staticmethod(thermo.convert_KtoC)
    convert_CtoK = staticmethod(thermo.convert_CtoK)
    tetens = staticmethod(thermo.tetens)
    calc_E = staticmethod(thermo.calc_E)
    calc_Q = staticmethod(thermo.calc_Q)
    calc_pV = staticmethod(thermo.calc_pV)
    calc_Tsa = staticmethod(thermo.calc_Tsa)
    calc_Tsa_sonic_temp = staticmethod(thermo.calc_Tsa_sonic_temp)
    calc_Es = staticmethod(thermo.calc_Es)
    calc_Td_dewpoint = staticmethod(thermo.calc_Td_dewpoint)
    get_Watts_to_H2O_conversion_factor = staticmethod(
        thermo.get_watts_to_h2o_conversion_factor
    )

    despike = staticmethod(cleaning.despike)
    despike_ewma_fb = staticmethod(cleaning.despike_ewma_fb)
    despike_med_mod = staticmethod(cleaning.despike_med_mod)
    despike_quart_filter = staticmethod(cleaning.despike_quart_filter)

    calc_cov = staticmethod(covariance.calc_cov)
    calc_MSE = staticmethod(covariance.calc_MSE)
    calc_max_covariance = staticmethod(covariance.calc_max_covariance)
    calc_covar = staticmethod(covariance.calc_covar)
