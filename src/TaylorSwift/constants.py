"""
Physical constants and configuration parameters for eddy covariance calculations.

This module provides:
1. Physical constants
2. Meteorological parameters
3. Unit conversion factors
4. Quality control thresholds
"""

import numpy as np
from enum import IntEnum

__all__ = [
    # Physical constants
    "K_VON_KARMAN",
    "G0",
    "R_GAS",
    "OMEGA",
    "SIGMA_SB",
    "CP_DRY_AIR",
    "CP_H2O_GAS",
    "L_SUBLIMATION",
    "L_VAPORIZATION",
    # Gas properties and constants
    "MOLAR_MASS",
    "R_SPECIFIC",
    # Temperature
    "T_ZERO_C",
    "T_TRIPLE_POINT",
    # Pressure and density
    "P_REFERENCE",
    "RHO_AIR_STP",
    # Surface roughness
    "SurfaceType",
    "ROUGHNESS_LENGTH",
    "DISPLACEMENT_RATIO",
    "get_displacement_height",
    "get_roughness_length",
    # Hemisphere
    "Hemisphere",
    # Quality control
    "QualityThreshold",
    "ErrorCode",
    # Unit conversion
    "UNIT_CONVERSION",
]

# Physical constants
K_VON_KARMAN = 0.40  # von Karman constant (dimensionless); von_karman
G0 = 9.80665  # Gravitational acceleration (m/s^2); g
R_GAS = 8.3144598  # Universal gas constant (J/mol/K); Ru
OMEGA = 7.2921e-5  # Earth's angular velocity (rad/s); Omega
SIGMA_SB = 5.67e-8  # Stefan-Boltzmann constant (W/m^2/K^4): Sigma_SB
CP_DRY_AIR = (
    1004.67  # Specific heat capacity of dry air at constant pressure (J/kg/K); Cpd
)
CP_H2O_GAS = 1850.0  # Specific heat capacity of water vapor (J/kg/K)
L_SUBLIMATION = 2.834e6  # Latent heat of sublimation (J/kg)
L_VAPORIZATION = 2.501e6  # Latent heat of vaporization at 0°C (J/kg)

# Gas properties
MOLAR_MASS = {
    "co2": 0.044010,  # CO2 molar mass (kg/mol)
    "h2o": 0.018015,  # H2O molar mass (kg/mol)
    "air_dry": 0.028964,  # Dry air molar mass (kg/mol); md
}

# Gas constants
R_SPECIFIC = {
    "dry_air": 287.04,  # Specific gas constant for dry air (J/kg/K)
    "water_vapor": 461.51,  # Specific gas constant for water vapor (J/kg/K)
}

# Temperature conversions
T_ZERO_C = 273.15  # 0°C in Kelvin
T_TRIPLE_POINT = 273.16  # Triple point of water (K)

# Pressure and density reference values
P_REFERENCE = 101.325  # Standard atmospheric pressure (kPa)
RHO_AIR_STP = 1.225  # Air density at STP (kg/m^3)


# Surface types
class SurfaceType(IntEnum):
    """Classification of surface types for roughness calculations"""

    CROP = 1
    GRASS = 2
    SHRUB = 3
    FOREST = 4
    BARELAND = 5
    WATER = 6
    ICE = 7
    URBAN = 8


# Hemisphere indicators
class Hemisphere(IntEnum):
    """Hemisphere indicators for latitude/longitude calculations"""

    NORTH = 1
    SOUTH = -1
    EAST = 1
    WEST = -1


# Default roughness lengths (m) for different surface types
ROUGHNESS_LENGTH = {
    SurfaceType.CROP: 0.02,
    SurfaceType.GRASS: 0.01,
    SurfaceType.SHRUB: 0.05,
    SurfaceType.FOREST: 1.0,
    SurfaceType.BARELAND: 0.001,
    SurfaceType.WATER: 0.0001,
    SurfaceType.ICE: 0.00001,
    SurfaceType.URBAN: 1.0,
}

# Default displacement heights as fraction of canopy height
DISPLACEMENT_RATIO = {
    SurfaceType.CROP: 0.66,
    SurfaceType.GRASS: 0.66,
    SurfaceType.SHRUB: 0.67,
    SurfaceType.FOREST: 0.67,
    SurfaceType.BARELAND: 0.0,
    SurfaceType.WATER: 0.0,
    SurfaceType.ICE: 0.0,
    SurfaceType.URBAN: 0.7,
}


# Quality control thresholds
class QualityThreshold:
    """Default thresholds for quality control"""

    # Standard quality rating thresholds (used for multiple metrics)
    _QUALITY_LEVELS = {
        "high_quality": 0.3,
        "moderate_quality": 0.7,
        "low_quality": 2.0,
    }

    # Steady state test thresholds (uses standard quality levels)
    RN_THRESHOLD = _QUALITY_LEVELS

    # ITC (Integral Turbulence Characteristics) thresholds (uses standard quality levels)
    ITC_THRESHOLD = _QUALITY_LEVELS

    # Wind direction thresholds (degrees from sonic orientation)
    WIND_DIRECTION = {
        "optimal": (-150, 150),  # Optimal wind direction range
        "acceptable": (-170, 170),  # Acceptable wind direction range
    }

    # Signal strength thresholds (minimum acceptable values)
    SIGNAL_STRENGTH = {
        "default": 0.7,  # Default minimum signal strength
        "co2": 0.7,  # Minimum CO2 signal strength
        "h2o": 0.7,  # Minimum H2O signal strength
    }

    # Range limits for various measurements
    VALID_RANGE = {
        "wind_speed": (-30.0, 30.0),  # m/s
        "wind_direction": (0.0, 360.0),  # degrees
        "temperature": (-40.0, 50.0),  # °C
        "co2": (200.0, 900.0),  # ppm
        "h2o": (0.0, 40.0),  # mmol/mol
        "pressure": (80.0, 110.0),  # kPa
    }


# Error codes
class ErrorCode(IntEnum):
    """Error codes for various processing steps"""

    SUCCESS = 0
    MISSING_DATA = 1
    RANGE_ERROR = 2
    QUALITY_ERROR = 3
    PROCESSING_ERROR = 4
    CONFIG_ERROR = 5


# Unit conversion factors
UNIT_CONVERSION = {
    "ppm_to_mgm3": {
        "co2": 1.96,  # Convert CO2 ppm to mg/m^3
    },
    "wm2_to_umolm2s": {
        "par": 4.57,  # Convert PAR W/m^2 to μmol/m^2/s
    },
    "ms_to_kmh": 3.6,  # Convert m/s to km/h
    "pa_to_kpa": 0.001,  # Convert Pa to kPa
}


def get_displacement_height(
    surface_type: SurfaceType,
    canopy_height: float,
) -> float:
    r"""
    Estimate the zero-plane displacement height *d* for a vegetated or
    rough surface.

    The zero-plane displacement height represents the effective level
    above ground at which the mean wind speed becomes zero due to drag
    exerted by the surface elements (e.g., tree crowns, crop stems,
    buildings).  It is frequently parameterised as a fixed fraction of
    the canopy or obstacle height :math:`h`:

    .. math::
        d = k_{d}\,h,

    where the proportionality factor :math:`k_{d}` varies by surface
    type (forest, crop, grass, urban, …).

    Parameters
    ----------
    surface_type : SurfaceType
        Enumerated label describing the roughness class.
        Must be a key in the module-level dictionary
        ``DISPLACEMENT_RATIO`` that maps each surface to a dimensionless
        coefficient :math:`k_{d}`.
    canopy_height : float
        Mean height of the vegetation canopy or obstacle layer
        (*h*, m).  Must be non-negative.

    Returns
    -------
    float
        Displacement height *d* (m).

    Raises
    ------
    KeyError
        If *surface_type* is not present in ``DISPLACEMENT_RATIO``.
    ValueError
        If *canopy_height* is negative.

    Notes
    -----
    * Typical ratios :math:`k_{d}` are
      ``0.60–0.70`` for tall forests,
      ``0.50`` for maize or wheat,
      ``0.20–0.30`` for short grass,
      ``0.70–0.80`` for dense urban canopies (Oke, 1987).
    * The returned *d* is used in logarithmic wind-profile and Monin–
      Obukhov similarity relationships together with the roughness
      length *z₀*.

    References
    ----------
    Oke, T. R. (1987). *Boundary Layer Climates* (2nd ed.). Routledge.
    Stull, R. B. (1988). *An Introduction to Boundary Layer Meteorology*.
    Springer.

    Examples
    --------
    >>> from TaylorSwift.constants import SurfaceType, get_displacement_height
    >>> get_displacement_height(SurfaceType.FOREST, canopy_height=20.0)
    13.0   # (k_d = 0.65)
    >>> get_displacement_height(SurfaceType.GRASS, canopy_height=0.15)
    0.03   # (k_d = 0.25)
    """
    if canopy_height < 0:
        raise ValueError("canopy_height must be non-negative")

    return DISPLACEMENT_RATIO[surface_type] * canopy_height


def get_roughness_length(
    surface_type: SurfaceType,
    canopy_height: float,
    custom_value: float | None = None,
) -> float:
    r"""
    Retrieve or estimate the aerodynamic roughness length :math:`z_{0}`.

    The roughness length is the theoretical height at which the mean
    wind speed goes to zero in the logarithmic wind-profile equation.
    When *custom_value* is supplied the function simply returns it.
    Otherwise, *z₀* is determined from empirical relationships that
    depend on the surface class:

    * **Cropland, grassland, shrubland**
      :math:`z_{0} = 0.15\,h`
    * **Other predefined classes**
      Value taken from the constant dictionary
      ``ROUGHNESS_LENGTH``.

    Parameters
    ----------
    surface_type : SurfaceType
        Enumeration identifying the roughness class (e.g.,
        ``SurfaceType.CROP``).
        Must be a key in either the *crop/grass/shrub* list or the global
        mapping ``ROUGHNESS_LENGTH``.
    canopy_height : float
        Mean canopy or obstacle height *h* (m).  Used only when the
        empirical factor *0.15* applies.  Must be non-negative.
    custom_value : float, optional
        User-specified roughness length (m).  If provided, it overrides
        the empirical estimates and table look-ups.

    Returns
    -------
    float
        Aerodynamic roughness length :math:`z_{0}` (m).

    Raises
    ------
    KeyError
        If *surface_type* is not present in ``ROUGHNESS_LENGTH`` and is
        not one of the crop/grass/shrub classes.
    ValueError
        If *canopy_height* is negative.

    Notes
    -----
    * The factor *0.15* originates from classical wind-tunnel studies
      showing that ``z0 ≈ (0.1–0.2)·h`` for many homogeneous vegetation
      canopies (Stull, 1988).
    * For tall or very sparse canopies site-specific calibration is
      recommended; supply *custom_value* to bypass the defaults.
    * Roughness length is typically paired with the displacement height
      *d* in Monin–Obukhov similarity theory.  Ensure that both are
      derived consistently (e.g., *d ≈ 0.65·h* for tall vegetation).

    References
    ----------
    Stull, R. B. (1988). *An Introduction to Boundary Layer Meteorology*.
    Springer.

    Examples
    --------
    >>> from TaylorSwift.constants import SurfaceType, get_roughness_length
    >>> get_roughness_length(SurfaceType.CROP, canopy_height=2.0)
    0.3
    >>> get_roughness_length(SurfaceType.URBAN, canopy_height=10.0)
    1.0                     # value from ROUGHNESS_LENGTH table
    >>> get_roughness_length(
    ...     SurfaceType.FOREST, canopy_height=25.0, custom_value=2.2)
    2.2
    """
    if canopy_height < 0:
        raise ValueError("canopy_height must be non-negative")

    if custom_value is not None:
        return custom_value

    if surface_type in [SurfaceType.CROP, SurfaceType.GRASS, SurfaceType.SHRUB]:
        return 0.15 * canopy_height

    return ROUGHNESS_LENGTH[surface_type]


def _calc_L(Ustr: float, Tsa: float, Uz_Ta: float, g: float, kappa: float) -> float:
    if abs(Uz_Ta) < 1e-12 or Ustr <= 0:
        return np.inf
    return -(Ustr**3) * Tsa / (g * kappa * Uz_Ta)
