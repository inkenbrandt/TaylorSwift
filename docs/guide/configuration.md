# Configuration

Nearly every entry point takes a configuration dataclass. There are four, and
they do not overlap much.

| Class | Purpose |
| --- | --- |
| [`SiteConfig`](../api/config.md) | Tower geometry, instrument geometry, sampling. Used by the spectral stack. |
| `InstrumentConfig` | Alias of `SiteConfig`, kept for readability at correction call sites. |
| [`FluxConfig`](../api/config.md) | Constants and sensor coefficients for the legacy `CalcFlux` pipelines. |
| [`ScreeningConfig`](../api/screening.md) | Thresholds for Vickers & Mahrt raw-data screening. |

## SiteConfig

```python
import TaylorSwift as tswift

config = tswift.SiteConfig(
    z_measurement=3.0,
    z_canopy=0.3,
    sampling_freq=20.0,
    averaging_period=30.0,
)
```

### Site geometry

| Field | Default | Meaning |
| --- | --- | --- |
| `z_measurement` | `3.0` | Sensor height above ground [m] |
| `z_canopy` | `0.3` | Canopy height [m] |
| `d` | `None` | Displacement height [m]; defaults to $\tfrac{2}{3} h_c$ |
| `z0` | `None` | Roughness length [m]; defaults to $0.1 h_c$ |
| `latitude` | `0.0` | Site latitude [°], used for the Coriolis term in ITC tests |
| `longitude` | `0.0` | Site longitude [°] |

Two derived properties are computed for you:

```python
config.z_eff                    # z_measurement - d
config.sensor_separation_total  # Euclidean norm of the three separations
```

`z_eff` is the height used for the dimensionless frequency
$f = n z_{\text{eff}} / \overline{U}$, so getting `z_canopy` right matters
more than it looks.

!!! tip "Non-default surfaces"
    The $\tfrac{2}{3} h_c$ / $0.1 h_c$ rules are rough. For tabulated values by
    surface type, use the helpers in
    [`constants`](../api/constants.md):

    ```python
    from TaylorSwift.constants import SurfaceType, get_displacement_height, get_roughness_length

    d = get_displacement_height(SurfaceType.GRASS, 0.3)
    z0 = get_roughness_length(SurfaceType.GRASS, 0.3)
    config = tswift.SiteConfig(z_measurement=3.0, z_canopy=0.3, d=d, z0=z0)
    ```

### Sampling

| Field | Default | Meaning |
| --- | --- | --- |
| `sampling_freq` | `20.0` | Acquisition rate [Hz] |
| `averaging_period` | `30.0` | Flux averaging window [minutes] |

The Nyquist frequency is `sampling_freq / 2`; anything the transfer functions
predict above it cannot be recovered, only corrected for.

### Sonic anemometer

| Field | Default (IRGASON) | Meaning |
| --- | --- | --- |
| `sonic_path_length` | `0.10` | Sonic path length [m] |
| `sonic_path_separation` | `0.0` | Separation between horizontal paths [m] |

### Gas analyser

| Field | Default (IRGASON) | Meaning |
| --- | --- | --- |
| `irga_path_length` | `0.154` | Optical path length [m] |
| `irga_path_diameter` | `0.005` | Optical path diameter [m] |
| `irga_type` | `"open_path"` | `"open_path"` or `"enclosed_path"` |
| `model` | `"IRGASON"` | Instrument model name |

### Sensor separation

| Field | Default | Meaning |
| --- | --- | --- |
| `sensor_separation_lateral` | `0.0` | Perpendicular to wind [m] |
| `sensor_separation_longitudinal` | `0.0` | Parallel to wind [m] |
| `sensor_separation_vertical` | `0.0` | Vertical [m] |

The IRGASON is co-located, so all three default to zero. A separate CSAT3 +
LI-7500 pairing is not — measure and set these, because sensor separation is
usually the largest high-frequency loss term.

### Time constants

| Field | Default | Meaning |
| --- | --- | --- |
| `tau_sonic_T` | `0.0` | Sonic temperature [s] |
| `tau_co2` | `0.1` | CO₂ sensor response [s] |
| `tau_h2o` | `0.1` | H₂O sensor response [s] |
| `tau_T` | `0.0` | Sonic T response [s] |

Sonic temperature is effectively instantaneous, hence the zeros.

## A non-IRGASON example

A CSAT3 sonic with an LI-7500 mounted 15 cm laterally and 5 cm below:

```python
config = tswift.SiteConfig(
    z_measurement=10.0,
    z_canopy=1.5,
    sampling_freq=10.0,
    averaging_period=30.0,
    model="CSAT3 + LI-7500",
    sonic_path_length=0.115,
    irga_path_length=0.125,
    irga_path_diameter=0.0095,
    sensor_separation_lateral=0.15,
    sensor_separation_vertical=0.05,
    tau_co2=0.1,
    tau_h2o=0.1,
)

print(f"{config.sensor_separation_total:.3f} m total separation")
```

## ScreeningConfig

See [Raw-data screening](screening.md) for the thresholds and what each
controls.

## FluxConfig

`FluxConfig` belongs to the legacy `CalcFlux` pipelines and carries physical
constants, KH-20 calibration coefficients, and the sonic boom orientation.
See [Legacy flux pipelines](pipelines.md).

## Input validity and gap filling

`process_interval` owns private writable copies of all six channels, including
arrays borrowed from pandas or Polars. Inputs must be one-dimensional, have equal
lengths, and contain at least **four samples**, giving two positive FFT frequencies
after linear detrending. Structural errors raise `ValueError` before screening.
This is a computational minimum, not a recommendation for scientific averaging.

`SiteConfig` validates finite positive sampling frequency, averaging period and
effective measurement height (`z_measurement - d`). Heights and sensor time
constants must be finite and nonnegative (measurement height must be positive).
Settings are checked again when processing to catch later mutations.

The missing-data defaults are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `min_finite_fraction` | 0.95 for each of `u`, `v`, `w`, `T`, `co2`, `h2o` | Required fraction of original finite samples per channel |
| `max_gap_seconds` | 1.0 | Maximum consecutive missing samples divided by sampling frequency; zero disables filling |
| `endpoint_policy` | `"reject"` | Reject leading or trailing gaps; `"nearest"` allows constant extension from the nearest finite sample |

NaN and either infinity count as missing. Interior gaps use linear interpolation.
Endpoint extension obeys the same duration and completeness limits. At least two
finite samples are always required. Each finite fraction must be in `(0, 1]`,
and the dictionary must contain all six channel keys. For example:

```python
config = tswift.SiteConfig(max_gap_seconds=0.5, endpoint_policy="reject")
config.min_finite_fraction["co2"] = 0.98
```

`result.qc_flags` contains `{channel}_finite_fraction` and `{channel}_status`.
Statuses are `ok`, `interpolated`, `insufficient_finite_data`, `gap_too_long`,
or `endpoint_gap`, checked in that order of failure priority. These fields are
included in scalar table exports. `interval_status` is `ok`, `partial` (one or
more invalid scalars), or `invalid_wind` (any invalid wind component).

Invalid wind returns empty spectra and NaN statistics. Invalid scalars retain
NaN arrays matching the wind frequency grid, with NaN covariances; invalid
temperature also invalidates temperature mean, heat flux and stability results.
Other usable channels remain available. The legacy `too_many_nans` flag denotes
wind completeness failure and now includes infinities. Completeness is evaluated
per channel rather than using the union of missing wind samples.

`process_file` skips windows shorter than four samples or below its existing 90%
record-count requirement. Fractional-second averaging periods are supported down
to the timestamp resolution of one microsecond. `bins_per_decade` must be a
positive integer.
