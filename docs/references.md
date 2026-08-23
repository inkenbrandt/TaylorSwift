# References

The methods implemented in `TaylorSwift`, with the modules that implement them.

## Spectra and cospectra

Kaimal, J.C., Wyngaard, J.C., Izumi, Y. & Coté, O.R. (1972). Spectral
characteristics of surface-layer turbulence. *Quarterly Journal of the Royal
Meteorological Society*, 98, 563–589.
: The normalisation conventions and model curves used throughout
  [`core`](api/core.md), [`cospectra`](api/cospectra.md), and
  [`transfer_functions`](api/transfer_functions.md).

Kaimal, J.C., Wyngaard, J.C. & Haugen, D.A. (1968). Deriving power spectra from
a three-component sonic anemometer. *Journal of Applied Meteorology*, 7,
827–837.
: Sonic line-averaging transfer function.

Moraes, O.L.L., Acevedo, O.C., Degrazia, G.A., Anfossi, D., da Silva, R. &
Turnes, V. (2008). Comparing spectra and cospectra of turbulence over different
surface boundary conditions. *Physica A*, 387, 4927–4939.
: Logarithmic binning practice.

## Frequency-response corrections

Moore, C.J. (1986). Frequency response corrections for eddy correlation
systems. *Boundary-Layer Meteorology*, 37, 17–35.
: First-order sensor response, scalar path averaging, and sensor separation
  transfer functions in [`transfer_functions`](api/transfer_functions.md).

Massman, W.J. (2000). A simple method for estimating frequency response
corrections for eddy covariance systems. *Agricultural and Forest Meteorology*,
104, 185–198.
: The numerical correction-factor method, and the default in
  [`corrections`](api/corrections.md).

Massman, W.J. (2001). Reply to comment by Rannik on "A simple method for
estimating frequency response corrections for eddy covariance systems".
*Agricultural and Forest Meteorology*, 107, 247–251.

Horst, T.W. (1997). A simple formula for attenuation of eddy fluxes measured
with first-order-response scalar sensors. *Boundary-Layer Meteorology*, 82,
219–233.
: The closed-form alternative, `horst_analytical_correction`.

## Density corrections

Webb, E.K., Pearman, G.I. & Leuning, R. (1980). Correction of flux measurements
for density effects due to heat and water vapour transfer. *Quarterly Journal
of the Royal Meteorological Society*, 106, 85–100.
: The WPL correction, `wpl_correction`.

Leuning, R. (2007). The correct formula for the WPL correction.
*Boundary-Layer Meteorology*, 126, 263–272.

## Wind corrections

Horst, T.W., Wilczak, J.M. & Cook, D. (2015). Correction of a non-orthogonal,
three-component sonic anemometer for flow distortion by transducer shadowing.
*Boundary-Layer Meteorology*, 155, 371–395.
: CSAT3 transducer-shadow correction, `shadow_correction`.

## Quality control

Vickers, D. & Mahrt, L. (1997). Quality control and flux sampling problems for
tower and aircraft data. *Journal of Atmospheric and Oceanic Technology*, 14,
512–526.
: The raw-data screening tests in [`screening`](api/screening.md). See
  [Raw-data screening](guide/screening.md).

Foken, T. & Wichura, B. (1996). Tools for quality assessment of surface-based
flux measurements. *Agricultural and Forest Meteorology*, 78, 83–105.
: The stationarity test, `stationarity_test`.

Foken, T., Göockede, M., Mauder, M., Mahrt, L., Amiro, B. & Munger, W. (2004).
Post-field data quality control. In X. Lee, W. Massman & B. Law (Eds.),
*Handbook of Micrometeorology* (pp. 181–208). Springer.
: The 9-class quality flag scheme and ITC tests in the `DataQuality` class.

## Despiking

Metzger, S., Junkermann, W., Mauder, M., Beyrich, F., Butterbach-Bahl, K.,
Schmid, H.P. & Foken, T. (2012). Eddy-covariance flux measurements with a
weight-shift microlight aircraft. *Atmospheric Measurement Techniques*, 5,
1699–1717.
: The iterative UKDE despiking method in [`despike`](api/despike.md).

## Background

Stull, R.B. (1988). *An Introduction to Boundary Layer Meteorology*. Springer.

Oke, T.R. (1987). *Boundary Layer Climates* (2nd ed.). Routledge.
: Roughness length and displacement height values behind the
  [`constants`](api/constants.md) helpers.

Aubinet, M., Vesala, T. & Papale, D. (Eds.) (2012). *Eddy Covariance: A
Practical Guide to Measurement and Data Analysis*. Springer.
: A good general reference for the whole workflow.
