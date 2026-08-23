# Legacy flux pipelines

Alongside the spectral stack, `TaylorSwift` ships two end-to-end `CalcFlux`
pipelines that go straight from a raw frame to mean fluxes for one averaging
period. They exist for continuity with the original `CalcFlux` codebase.

!!! question "Which should I use?"
    Use [`process_file`](spectra.md) if you want spectra, cospectra, ogives, or
    per-frequency diagnostics. Use these pipelines if you only want the flux
    numbers for an averaging period and want the whole chain applied for you.

| Pipeline | Deployment |
| --- | --- |
| `run_irga` | Integrated open-path IRGA + CSAT3 — IRGASON, LI-7500 + CSAT3 |
| `run_kh20` | KH-20 krypton hygrometer + CSAT3 |

Both apply, in order: despiking, CSAT3 shadow correction, double coordinate
rotation, lag-optimised covariances, Massman (2000, 2001) spectral corrections,
and the Webb-Pearman-Leuning (1980) density correction.

## Running one

```python
import TaylorSwift as tswift

config = tswift.FluxConfig(
    meter_type="IRGASON",
    UHeight=3.52,     # measurement height [m]
    sonic_dir=225.0,  # sonic boom azimuth [deg]
    lag=10,           # max lag [samples] for the covariance search
)

fluxes = tswift.run_irga(df, config)
print(fluxes)
```

`df` must cover **exactly one averaging period** — the pipeline returns one set
of numbers, not a time series. Group and apply for a whole day:

```python
daily = (
    df.set_index("TIMESTAMP")
      .groupby(pd.Grouper(freq="30min"))
      .apply(lambda block: tswift.run_irga(block, config))
)
```

## What comes back

A 13-element `pandas.Series`:

| Field | Meaning | Units |
| --- | --- | --- |
| `Ta` | Air temperature | K |
| `Td` | Dewpoint temperature | K |
| `D` | Vapour pressure deficit | kPa |
| `Ustr` | Friction velocity $u_*$ | m s⁻¹ |
| `zeta` | Stability parameter $z/L$ | — |
| `H` | Sensible heat flux | W m⁻² |
| `StDevUz` | Standard deviation of vertical wind | m s⁻¹ |
| `StDevTa` | Standard deviation of air temperature | K |
| `direction` | Wind direction | ° |
| `exchange` | Exchange coefficient | — |
| `lambdaE` | Latent heat flux | W m⁻² |
| `ET` | Evapotranspiration | mm |
| `Uxy` | Horizontal wind speed | m s⁻¹ |

## Input columns

After renaming, the pipelines expect:

| Column | Meaning | Units |
| --- | --- | --- |
| `Ux`, `Uy`, `Uz` | Wind components | m s⁻¹ |
| `Ts` | Sonic virtual temperature | °C |
| `Pr` | Air pressure | kPa |
| `pV` | Water-vapour density (`run_irga`) | g m⁻³ |
| `volt_KH20` | Hygrometer output (`run_kh20`) | mV |

Units differ from the spectral stack — °C and kPa here, converted internally to
K and Pa. Feeding K into `Ts` will not error; it will just produce wrong
fluxes.

A default rename map already handles common logger names:

| Logger name | Canonical |
| --- | --- |
| `T_SONIC` | `Ts` |
| `PA`, `amb_press` | `Pr` |
| `H2O_density` | `pV` |
| `TA_1_1_1`, `t_hmp` | `Ta` |
| `RH_1_1_1` | `Rh` |
| `kh` | `volt_KH20` |
| `e_hmp` | `Ea` |
| `q` | `Q` |

Override or extend it per call:

```python
fluxes = tswift.run_irga(
    df,
    config,
    rename_map={"sonic_temp": "Ts", "press_kpa": "Pr"},
    ts_col="datetime",
)
```

`ts_col` names your timestamp column when it is not `TIMESTAMP`.

## FluxConfig

`FluxConfig` carries the physical constants and sensor coefficients. The fields
you are most likely to change:

| Field | Default | Meaning |
| --- | --- | --- |
| `meter_type` | `"IRGASON"` | Sensor suite |
| `UHeight` | `3.52` | Measurement height [m] |
| `sonic_dir` | `225.0` | Sonic boom azimuth [°] |
| `lag` | `10` | Max lag [samples] for the covariance maximisation |
| `PathDist_U` | `0.0` | Sonic-to-hygrometer path distance [m] |
| `direction_bad_min` / `direction_bad_max` | `0.0` / `360.0` | Wind-direction rejection sector |
| `despikefields` | `Ux, Uy, Uz, Ts, volt_KH20, Pr, Rh, pV` | Columns to despike |

KH-20 users will also want `XKH20`, `XKwC1`, `XKwC2`, `Kw`, and `Ko`, which are
the hygrometer calibration coefficients from your instrument's certificate —
the defaults are placeholders, not your sensor.

## The `CalcFlux` class

For code written against the original API, `CalcFlux` is preserved as a thin
wrapper. Keyword arguments forward straight to `FluxConfig`:

```python
from TaylorSwift import CalcFlux

cf = CalcFlux(UHeight=5.0, meter_type="KH20")
print(cf.config.UHeight)   # 5.0
print(cf.convert_CtoK(0.0))  # 273.15
```

Nothing is re-implemented in the wrapper — the physics delegates to
[`thermo`](../api/thermo.md) and [`covariance`](../api/covariance.md). New code
should call `run_irga` / `run_kh20` directly.

## Lag-optimised covariances

Both pipelines maximise the covariance over a lag window rather than assuming
zero lag, which matters when the gas analyser sits downstream of the sonic:

```python
from TaylorSwift.covariance import calc_max_covariance

result = calc_max_covariance(w, co2, lag=10)
best_lag, best_cov = result[0]
print(f"peak covariance {best_cov:.4g} at lag {best_lag} samples")
```

It searches lags in `[-lag, +lag]` for the one maximising `|cov(x, y)|`, and
returns a list of `(lag, covariance)` tuples — note the **lag comes first**.
For every velocity-variable pair at once, use `build_covariance_dict`, which
prepares and transforms each array only once:

```python
from TaylorSwift.covariance import build_covariance_dict

covs = build_covariance_dict(
    velocities={"Ux": Ux, "Uy": Uy, "Uz": Uz},
    variables={"Ts": Ts, "pV": pV},
    lag=10,
)
print(covs["Uz-Ts"])
```

Set `config.lag` from your actual tube delay where you know it. Too wide a
window lets the search lock onto noise.
