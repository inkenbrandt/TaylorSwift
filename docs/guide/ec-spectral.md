# Spectral diagnostics and fitted corrections

`TaylorSwift.ec_spectral` adapts the supplied standalone module for lag detection,
cospectral fitting, instrument response diagnostics, and covariance correction.
It includes Kaimal and generalized Massman models, open- and closed-path transfer
functions, analytical and integral corrections, relative uncertainty, and six
diagnostic plots. All original helper names are available in this submodule.

## Reproducible workflow

```python
from TaylorSwift import ec_spectral as ecs

system = ecs.ECSystem(fs=20, avg_period=1800, z_ref=3, lateral_sep=0.1)
w, c = ecs.synthetic_turbulence(fs=system.fs, T=system.avg_period, seed=7)
lag, lags_s, correlation = ecs.find_lag(w, c, system.fs)
co = ecs.cospectrum(w, ecs.shift(c, lag), system.fs)
fit = ecs.fit_cospectrum(co["f_bin"], co["fCo_bin"], system.fs, system.avg_period)
factor = ecs.correction_factor_integral(system, u=3, fx=fit["fx"], mu=fit["mu"])
corrected_covariance = co["cov"] * factor
ax = ecs.plot_cospectrum(co, fit, system, u=3)
```

Run `examples/04_ec_spectral_diagnostics.ipynb` from a checkout after installing
`pip install -e ".[notebooks]"`. It uses seeded synthetic data and needs no
site files. For field data, `load_toa5` returns a pandas table indexed by
timestamp, `double_rotation` rotates wind, and `despike` supplies a simple
rolling median/MAD filter. Existing package screening and despiking routines
remain available for more comprehensive quality control.

## Conventions and limits

- Frequencies and sampling rate are Hz; lengths are metres; time constants and
  `ECSystem.avg_period` are **seconds** (unlike `SiteConfig.averaging_period`,
  which is minutes). Flow is litres per minute. Wind speed must be positive.
- Positive lag means the scalar arrives after vertical wind. `shift` aligns it
  and fills the exposed edge with the endpoint value. Lag correlations use
  finite overlapping pairs, with a restricted search available via `expected_s`.
- `cospectrum` interpolates missing pairs, detrends, applies an optional Hamming
  taper, and rescales the density to the unwindowed detrended population
  covariance. Thus `sum(Co)*fs/N == cov`, and the high-to-low normalized ogive
  starts at one, including for negative flux. Zero/near-zero covariance or an
  incompatible tapered covariance raises `ValueError`. This rescaling is a
  diagnostic convention and differs from `compute_cospectrum`'s window-energy
  normalization. Do not mix their outputs without accounting for this difference.
- Binned values are for fitting/plotting; integrate the raw density. Fits use
  bounded nonlinear least squares on the central band (default `2/T` to `fs/10`).
  A high fit score alone does not validate site-specific similarity assumptions.
- `ECSystem(detrend="running_mean")` selects a recursive running-mean response;
  the default selects block averaging. This models instrument/processing losses;
  it does not change the separate `cospectrum(detrend=...)` preprocessing.
- The historical `include_lowpass=False` argument omits **low-frequency loss**
  from averaging, while retaining sensor and path attenuation.
- Integral factors use a finite logarithmic grid (default 0.0001 to `5*fs` Hz).
  Supply `n=` to check grid convergence. Analytical factors use equivalent time
  constants and need not match the integral model exactly. Tube models do not
  capture humidity-dependent wall adsorption.

## Correct a covariance table

```python
import pandas as pd
from TaylorSwift import ec_spectral as ecs

table = pd.DataFrame({"u": [1., 3., 5.], "cov_wCO2": [-0.2, -0.3, -0.1]})
corrected = ecs.correct_flux_table(
    table, ecs.ECSystem(), eta_x=0.11, cov_cols=("cov_wCO2",), method="integral"
)
```

The input is preserved. Outputs add `F`, `dF_rel`, `F_flag`, and `<column>_corr`.
Nonfinite/nonpositive wind speeds produce NaN factors and are flagged; factors
above `max_F` (default 1.5) are flagged but not clipped. Integral factors are
evaluated at each distinct valid speed. `dF_rel` is the analytical model's
relative uncertainty estimate, also reported when the integral method is used;
it is not a confidence interval for the fitted model. Reference `eta_x` values
are starting assumptions; estimate a site value from suitable intervals.

Apply these factors to raw **covariances before WPL correction**. Existing
`process_interval`, `process_file`, and `apply_spectral_corrections` retain their
own behavior; this workflow does not automatically run within those pipelines.
Avoid applying both correction workflows to the same covariance.

## Plots and API

`plot_lag`, `plot_cospectrum`, `plot_ogive`, `plot_transfer_functions`,
`plot_flux_loss`, and `plot_correction_vs_wind` return Matplotlib axes and accept
an existing `ax`. `demo(outdir)` saves a six-panel synthetic demonstration.
See the [API reference](../api/ec_spectral.md) for signatures and the source
module's model references (Moncrieff et al., 1997; Massman & Clement, 2004).
