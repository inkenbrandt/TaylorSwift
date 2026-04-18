from dataclasses import dataclass, field

@dataclass
class FluxConfig:
    meter_type: str = "IRGASON"
    sonic_dir: float = 225.0
    UHeight: float = 3.52
    PathDist_U: float = 0.0
    lag: int = 10

    Rv: float = 461.51
    Ru: float = 8.3143
    Cpd: float = 1005.0
    Rd: float = 287.05
    md: float = 0.02896
    Co: float = 0.21
    Mo: float = 0.032

    Cpw: float = 1952.0
    Cw: float = 4218.0
    epsilon: float = 18.016 / 28.97
    g: float = 9.81
    von_karman: float = 0.41
    MU_WPL: float = 28.97 / 18.016
    Omega: float = 7.292e-5
    Sigma_SB: float = 5.6718e-8

    XKH20: float = 1.412
    XKwC1: float = -0.152214126
    XKwC2: float = -0.001667836
    Kw: float = 1.0
    Ko: float = -0.0045

    direction_bad_min: float = 0.0
    direction_bad_max: float = 360.0

    despikefields: list[str] = field(default_factory=lambda: [
        "Ux", "Uy", "Uz", "Ts", "volt_KH20", "Pr", "Rh", "pV"
    ])

    parameters: dict[str, list[str]] = field(default_factory=lambda: {
        "Ea": ["Actual Vapor Pressure", "kPa"],
        "LnKH": ["Natural Log of Krypton Hygrometer Output", "ln(mV)"],
        "Pr": ["Air Pressure", "Pa"],
        "Ta": ["Air Temperature", "K"],
        "Ts": ["Sonic Temperature", "K"],
        "Ux": ["X Component of Wind Speed", "m/s"],
        "Uy": ["Y Component of Wind Speed", "m/s"],
        "Uz": ["Z Component of Wind Speed", "m/s"],
        "E": ["Vapor Pressure", "kPa"],
        "Q": ["Specific Humidity", "unitless"],
        "pV": ["Water Vapor Density", "kg/m^3"],
        "Sd": ["Entropy of Dry Air", "J/K"],
        "Tsa": ["Absolute Air Temperature Derived from Sonic Temperature", "K"],
    })
