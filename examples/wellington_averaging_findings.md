# Wellington H₂O: choosing an averaging interval

**Recommendation: use 60 minutes as a provisional compromise for H₂O spectral
diagnostics and model fitting, subject to stationarity screening.** Thirty minutes
remains the more conservative choice for resolving changing conditions and for a
routine flux product. Two hours is not supported as the default. These data do not
establish a unique interval that minimizes true flux error.

This conclusion comes from the full `Wellington_filtered.parquet` record,
27 June–17 July 2024, rather than comparing unrelated times in the two screenshots.
The analysis is specific to H₂O; it does not establish the best interval for CO₂,
heat, another season, or other deployment conditions.

## Matched comparison

The file contains 16,329,600 observations over 21 days on a nominal 10 Hz grid.
Each two-hour window is processed three ways: four half-hours, two one-hour
intervals, and one two-hour interval. All durations therefore cover the same
observations. Input channels are Ux, Uy, Uz, and H2O_density.

Of 252 possible two-hour windows, 244 pass the shared data-coverage policy.
The policy requires at least 85% finite coverage in each channel of each half-hour
and no missing run longer than 0.2 seconds. Coverage is typically 90%; short gaps
are interpolated on the original grid, with nearest-value filling at short edges.
No extra despiking is applied because earlier filtering is unknown.

The tapered analysis produces 71 incompatible spectra across those 244 windows:
the module cannot safely rescale a tapered covariance when its sign conflicts with
the unwindowed covariance. These failures are recorded, not silently replaced.
Requiring all seven spectra to succeed leaves **191 matched two-hour windows**:
764 half-hours, 382 one-hour intervals, and 191 two-hour intervals.

A second, explicitly defined subset requires |correlation(w, H₂O)| ≥ 0.1 in every
one of the seven intervals. This reduces instability in ratios normalized by weak
covariance. It retains **143 matched windows**, giving 572, 286, and 143 intervals.
It is a sensitivity screen, not a complete turbulence-quality test.

## Results in the stronger-signal subset

| Diagnostic | 30 minutes | 60 minutes | 120 minutes |
|---|---:|---:|---:|
| Median spectral fit R², common band | 0.365 | 0.513 | 0.619 |
| Pass fixed-five-minute stationarity screen | 93.0% | 91.3% | 88.8% |
| Median covariance change after linear detrending | 1.68% | 2.06% | 2.61% |
| Median absolute contribution of lowest three FFT bins | 5.57% | 4.85% | 5.06% |
| Fits near broadness parameter bounds | 16.8% | 10.8% | 11.2% |
| Median integral correction factor | 1.087 | 1.070 | 1.074 |

Hourly intervals give a substantial improvement in fitting a smooth cospectral
shape, fewer fits near parameter bounds, and a modestly smaller slow-frequency
endpoint contribution. Moving to two hours improves R² further but does not further
reduce that endpoint contribution and increases trend sensitivity and stationarity
failures. A smaller correction factor is not evidence of greater accuracy.

For all 191 matched windows, median R² is 0.368 / 0.491 / 0.587 and stationarity
pass rates are 82.2% / 79.3% / 75.4%. Weak-signal periods are harder to assess and
do not become reliable merely by extending the averaging period.

## Methods and sensitivity checks

- Each interval receives double wind rotation and an independent lag search over
  ±0.5 seconds; exposed shifted edges are trimmed. Boundary lags are recorded.
  Differences between interval covariances therefore include changes in the
  rotation and lag estimate as well as averaging duration.
- Spectra use block-mean removal, a Hamming taper, and the normalization in
  `ec_spectral.cospectrum`. Raw density integration is checked against covariance.
  The primary model fits use the same physical band, 0.001111–1 Hz. Default
  duration-dependent-band fits are also saved. There are 60 logarithmic bins;
  their boundaries and numbers of raw frequencies differ with record length.
  Longer records provide more averaging within bins, so improved R² must not be
  equated with a measured reduction in flux error.
- Stationarity compares the full covariance with the mean of covariances from
  fixed five-minute pieces: `abs(full - mean_short) / abs(full)`. The primary screen
  is ≤30%. A six-equal-piece alternative gives 93.0% / 92.0% / 91.6% pass rates in
  the stronger subset, showing that the penalty for longer averaging depends on
  the subinterval definition. The fixed-five-minute choice holds the comparison
  timescale constant across durations. This test is necessary but not sufficient
  for stationarity; real slow transport and changing conditions can both alter it.
- The lowest-three-bin diagnostic is the sum of absolute bin contributions divided
  by absolute net covariance. It covers periods T, T/2, and T/3, not the same
  physical frequency band for each duration. It is a relative endpoint diagnostic,
  not a standardized proof of ogive convergence or missing flux. The normalized
  ogive's endpoint of one is imposed by construction. Contributions at periods
  longer than 30 minutes are exported separately where those frequencies exist.
- Day-level paired bootstrapping, with 5,000 resamples and a fixed seed, preserves
  matching within each two-hour window and does not treat the seven overlapping
  intervals as independent. In the stronger subset, the hourly stationarity-pass
  difference from 30 minutes is −1.1 percentage points with an approximate 95%
  interval of −3.2 to +0.9 points. The two-hour difference is −4.7 points
  (−9.5 to −0.4). The hourly R² improvement is clearer: a day-weighted mean
  difference of +0.115 (0.096–0.133). These are differences of daily means, not
  differences of the medians in the table; 21 days limit generalization.
- Repeating the complete analysis **without tapering** retains all 244 eligible
  windows. Its stronger subset has 148 windows. Median R² is 0.400 / 0.557 / 0.671;
  stationarity pass rates are 91.0% / 89.5% / 86.5%; and median trend sensitivity
  is 1.76% / 2.12% / 2.99%. The overall tradeoff persists, although individual fits
  and the strength of some differences depend on tapering.

## The screenshot period: 27 June, 14:00–16:00

The 14:00–15:00 hour has a weak fit (common-band R² 0.040; default-band R² 0.079)
and 22.1% stationarity deviation. The next hour is substantially better (common-band
R² 0.575; stationarity deviation 2.8%). Combining the two into one interval gives
**66.8% stationarity deviation**, **25.5% sensitivity to linear detrending**, and
**51.6% absolute lowest-three-bin contribution**. The default-band two-hour fit
even has negative R². This particular example argues strongly against combining
those hours, despite the smoother appearance a long record can sometimes produce.

## Practical use and limits

Use hourly records for the spectral comparison when the signal is adequate and
the interval passes stationarity review. Retain half-hour diagnostics and shorten
the interval when changing conditions make an hour unsuitable. A passed stationarity
screen does not rescue an obviously poor spectral fit, as the 14:00 example shows.
If the priority is a standard flux time series with better time resolution, retain
30-minute processing and use suitable hourly spectra to help assess response models;
the correction's averaging-period setting must still match the actual flux period.

No energy-balance closure, independent reference flux, footprint screening, full
instrument-diagnostic screening, or comprehensive stationarity validation was
performed. Instrument geometry and the 0.1-second H₂O response remain illustrative
settings from the example notebooks. Interpolation of approximately 10% missing
samples affects the high-frequency tail. No WPL or latent-heat conversion is applied.
Consequently, this is a defensible spectral-analysis preference, not proof that an
hour gives the most accurate evapotranspiration estimate.

The decision to balance low-frequency capture with stationarity is consistent with
[Karimindla et al. (2024)](https://amt.copernicus.org/articles/17/5477/2024/index.html).
The stationarity statistic, 30% screen, and its limitations at weak flux are discussed
by [Vitale et al. (2020)](https://bg.copernicus.org/articles/17/1367/2020/bg-17-1367-2020.html).
The numerical recommendation above comes from Wellington, not from transferring
another site's preferred interval.

## Reproduce and inspect

Run `07_wellington_averaging_comparison.ipynb` from this checkout. Its saved outputs
include the summary and matched-period plots. Set `RUN_ANALYSIS = True` to rerun
both processing variants against the local Parquet file. The analysis and summary
scripts are `compare_wellington_averaging.py` and `summarize_wellington_averaging.py`.
Detailed interval metrics, coverage records, failures, settings, paired covariance
comparisons, threshold sensitivities, and bootstrap results are saved under
`examples/outputs/wellington_averaging_comparison/` and
`examples/outputs/wellington_averaging_untapered/`.
