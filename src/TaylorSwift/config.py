from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class FluxConfig:
    meter_type: str = "IRGASON"
    sonic_dir: float = 225.0
    UHeight: float = 3.52
    PathDist_U: float = 0.0
    lag: int = 10
    Rv: float = 461.51
    Rd: float = 287.05
    Co: float = 0.21
    Mo: float = 0.032
    Cpw: float = 1952.0
    Cw: float = 4218.0
    epsilon: float = 18.016 / 28.97

    MU_WPL: float = 28.97 / 18.016

    XKH20: float = 1.412
    XKwC1: float = -0.152214126
    XKwC2: float = -0.001667836
    Kw: float = 1.0
    Ko: float = -0.0045

    direction_bad_min: float = 0.0
    direction_bad_max: float = 360.0

    despikefields: list[str] = field(
        default_factory=lambda: ["Ux", "Uy", "Uz", "Ts", "volt_KH20", "Pr", "Rh", "pV"]
    )

    parameters: dict[str, list[str]] = field(
        default_factory=lambda: {
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
        }
    )


# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------
@dataclass
class SiteConfig:
    """
    Physical parameters for the sonic anemometer and gas analyser.

    Default values are for the Campbell Scientific IRGASON (integrated
    open-path sonic anemometer + CO₂/H₂O gas analyser).

    Attributes:
        sonic_path_length: Path length of the sonic anemometer [m].
        sonic_path_separation: Separation between horizontal paths [m].
        irga_path_length: Optical path length of the gas analyser [m].
        irga_path_diameter: Optical path diameter of the gas analyser [m].
        sensor_separation_lateral: Lateral separation perpendicular to wind [m].
        sensor_separation_longitudinal: Longitudinal separation parallel to wind [m].
        sensor_separation_vertical: Vertical separation [m].
        tau_sonic_T: Time constant for sonic temperature [s].
        tau_co2: Time constant for CO₂ sensor response [s].
        tau_h2o: Time constant for H₂O sensor response [s].
        tau_T: Time constant for sonic T response [s].
        irga_type: Type of gas analyser ('open_path' or 'enclosed_path').
        model: Instrument model name.
    """

    z_measurement: float = 3.0  # measurement height [m]
    z_canopy: float = 0.3  # canopy height [m]
    d: Optional[float] = None  # displacement height [m]  (default: 2/3 * z_canopy)
    z0: Optional[float] = None  # roughness length [m]     (default: 0.1 * z_canopy)
    latitude: float = 0.0  # site latitude [deg]
    longitude: float = 0.0  # site longitude [deg]
    sampling_freq: float = 20.0  # Hz
    averaging_period: float = 30.0  # minutes

    # --- Sonic anemometer ---
    sonic_path_length: float = 0.10  # sonic path length [m]
    # IRGASON: ~10 cm vertical path
    sonic_path_separation: float = 0.0  # separation between horizontal paths [m]
    # IRGASON: integrated, effectively 0

    # --- Gas analyser (IRGA) ---
    irga_path_length: float = 0.154  # optical path length [m]
    # IRGASON: 15.4 cm
    irga_path_diameter: float = 0.005  # optical path diameter [m]

    # --- Sensor separation ---
    # Distance between sonic measurement volume and IRGA measurement volume.
    # For IRGASON these are co-located, so separation ≈ 0.
    sensor_separation_lateral: float = 0.0  # perpendicular to wind [m]
    sensor_separation_longitudinal: float = 0.0  # parallel to wind [m]
    sensor_separation_vertical: float = 0.0  # vertical [m]

    # --- Time constants ---
    # First-order response time constants for the sensors.
    tau_sonic_T: float = 0.0  # sonic temperature [s] (essentially 0)
    tau_co2: float = 0.1  # CO₂ sensor response [s]
    # IRGASON: ~0.1 s at 20 Hz
    tau_h2o: float = 0.1  # H₂O sensor response [s]
    tau_T: float = 0.0  # sonic T response [s] (virtual instant)

    # --- Instrument type ---
    irga_type: str = "open_path"  # 'open_path' or 'enclosed_path'
    model: str = "IRGASON"  # instrument model name

    @property
    def sensor_separation_total(self) -> float:
        """Total sensor separation distance [m]."""
        return np.sqrt(
            self.sensor_separation_lateral**2
            + self.sensor_separation_longitudinal**2
            + self.sensor_separation_vertical**2
        )

    def __post_init__(self):
        if self.d is None:
            self.d = (2.0 / 3.0) * self.z_canopy
        if self.z0 is None:
            self.z0 = 0.1 * self.z_canopy

    @property
    def z_eff(self) -> float:
        """Effective measurement height above displacement height."""
        return self.z_measurement - self.d  # type: ignore


# Processing parameters
class ProcessingConfig:
    """Default configuration for flux processing"""

    # Time parameters
    AVERAGING_INTERVAL = 1800  # Default averaging interval (seconds)
    SUBINTERVAL = 300  # Sub-interval for stationarity test (seconds)

    # Spectral correction parameters
    FREQ_RESPONSE = {
        "low_freq_cutoff": 0.0001,  # Hz
        "high_freq_cutoff": 5.0,  # Hz
        "num_freq_points": 1000,  # Number of frequency points
    }

    # Despiking parameters
    DESPIKE = {
        "z_threshold": 3.5,  # Z-score threshold for spike detection
        "window_size": 100,  # Window size for moving statistics
    }

    # Rotation parameters
    ROTATION = {
        "max_rotation_angle": 15.0,  # Maximum rotation angle (degrees)
        "num_sectors": 36,  # Number of wind sectors for planar fit
    }

    # Storage flux parameters
    STORAGE = {
        "num_heights": 1,  # Number of measurement heights
        "integration_method": "linear",  # Profile integration method
    }

    # Webb corrections
    DENSITY_CORRECTION = {
        "apply_wpl": True,  # Apply WPL corrections
        "use_measured_h2o": True,  # Use measured H2O for density corrections
    }


# Plotting parameters
class PlottingConfig:
    """Default configuration for plotting"""

    COLORS = {
        "co2_flux": "#1f77b4",
        "h2o_flux": "#2ca02c",
        "heat_flux": "#d62728",
        "momentum_flux": "#9467bd",
    }

    FIGURE_SIZE = (10, 6)
    DPI = 100
    FONT_SIZE = 12
    LINE_WIDTH = 1.5
