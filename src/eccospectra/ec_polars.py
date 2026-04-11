# Original scripts in Fortran by Lawrence Hipps USU
# Transcibed from original Visual Basic scripts by Clayton Lewis and Lawrence Hipps

import polars as pl
import pandas as pd  # optional compat
import numpy as np
from scipy import signal
import statsmodels.api as sm

# --- Polars compatibility helpers ---
def _to_pl_df(df):
    if isinstance(df, pl.DataFrame):
        return df
    elif isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    else:
        raise TypeError("df must be pandas or polars DataFrame")

def _to_same_type(df_like, df_pl: pl.DataFrame):
    # Return df_pl as pandas if original was pandas
    if isinstance(df_like, pd.DataFrame):
        return df_pl.to_pandas()
    return df_pl

def _get_series(df, col):
    if isinstance(df, pl.DataFrame):
        return df[col]
    elif isinstance(df, pd.DataFrame):
        return pl.Series(col, df[col].to_numpy())
    else:
        raise TypeError("df must be pandas or polars DataFrame")

def _assign(df, **cols):
    # cols: name=array-like/Series
    if isinstance(df, pl.DataFrame):
        newcols = []
        for k,v in cols.items():
            if isinstance(v, pl.Series):
                s = v
                if v.name != k: s = v.alias(k)
            else:
                s = pl.Series(k, np.asarray(v))
            newcols.append(s)
        return df.with_columns(newcols)
    else:
        # pandas
        for k,v in cols.items():
            df[k] = np.asarray(v)
        return df

def _interpolate_bfill_ffill(s: pl.Series):
    if not isinstance(s, pl.Series):
        s = pl.Series(np.asarray(s))
    s2 = s.interpolate()
    s2 = s2.fill_null(strategy="backward")
    s2 = s2.fill_null(strategy="forward")
    return s2

def _rolling_median_centered(s: pl.Series, win: int):
    if not isinstance(s, pl.Series):
        s = pl.Series(np.asarray(s))
    out = s.rolling_median(window_size=win, center=True)
    out = out.fill_null(strategy="backward").fill_null(strategy="forward")
    return out

def _shift(s, n: int):
    if isinstance(s, pl.Series):
        return s.shift(n)
    else:
        return pl.Series(np.asarray(s)).shift(n)

def _first_last_index_duration(df, unit="D"):
    """Mimic (df.last_valid_index() - df.first_valid_index()) / pd.to_timedelta(1, unit=unit)"""
    if isinstance(df, pd.DataFrame) and df.index.size>0:
        delta = (df.last_valid_index() - df.first_valid_index())  # type: ignore
        return delta / pd.to_timedelta(1, unit=unit) # type: ignore
    # try to infer datetime column
    cand = None
    for name in ("TIMESTAMP","timestamp","time","TIMESTAMP_START","datetime"):
        if name in df.columns:
            cand = name; break
    if cand is None:
        raise ValueError("Cannot infer datetime column for duration; expected one of TIMESTAMP, timestamp, time, TIMESTAMP_START, datetime")
    s = _get_series(df, cand)
    return (s[-1] - s[0]).dt.total_days() if unit=="D" else (s[-1] - s[0]).dt.seconds()


class CalcFlux:
    # … class-level attributes & docstring remain unchanged …

    def __init__(self, **kwargs):
        """
        Initialize a :class:`CalcFlux` instance.

        This constructor sets a collection of physical constants, sensor
        configuration parameters, and run-time options that govern the eddy-
        covariance flux calculations performed by the class.  All attributes
        are first given sensible defaults (see *Attributes* below) and may be
        selectively overridden by supplying keyword arguments.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments whose names match one or more public
            attributes listed in *Attributes*.  Any supplied key–value pair
            replaces the default set during initialization, e.g.::

                flux = CalcFlux(UHeight=2.0, meter_type="KH20")

        Attributes
        ----------
        Cp : float or None
            Specific heat of (moist) air at constant pressure, *J kg⁻¹ K⁻¹*.
            Computed later from :pydata:`Cpd` and the mean specific humidity.
        Rv : float, default 461.51
            Gas constant of water vapour, *J kg⁻¹ K⁻¹*.
        Ru : float, default 8.3143
            Universal gas constant, *J mol⁻¹ K⁻¹*.
        Cpd : float, default 1005.0
            Specific heat of dry air at constant pressure, *J kg⁻¹ K⁻¹*.
        Rd : float, default 287.05
            Gas constant of dry air, *J kg⁻¹ K⁻¹*.
        md : float, default 0.02896
            Molar mass of dry air, *kg mol⁻¹*.
        Co : float, default 0.21
            Molar fraction of O₂ in the atmosphere.
        XKH20 : float, default 1.412
            Optical path length of the KH-20 krypton hygrometer, *cm*.
        sonic_dir : float, default 225.0
            Azimuth (° clockwise from true north) of the CSAT sonic anemometer.
        UHeight : float, default 3.52
            Measurement height of the sonic anemometer above ground, *m*.
        PathDist_U : float, default 0.0
            Horizontal separation between hygrometer and sonic, *m*.
        lag : int, default 10
            Number of lags (±) searched when maximising covariances.
        direction_bad_min, direction_bad_max : float
            Wind-direction sector to discard, degrees clockwise from the
            KH-20–to-sonic baseline.
        Kw, Ko : float
            Extinction coefficients for water and oxygen used in KH-20
            cross-sensitivity corrections.
        covar, avgvals, stdvals, errvals : dict
            Containers populated during processing for covariances, means,
            standard deviations and variances, respectively.
        despikefields : list[str]
            Column names to be despiked by default.
        wind_compass : float or None
            Mean wind direction (°) in meteorological convention; computed in
            :meth:`determine_wind_dir`.
        pathlen : float or None
            Effective horizontal path separation projected onto the wind, *m*.
        df : pandas.DataFrame or None
            Working DataFrame for convenience when methods rely on internal
            state.

        Notes
        -----
        * All attributes can be overridden via ``**kwargs``—be careful when
          passing physical constants.
        * No I/O occurs during instantiation; heavy calculations begin with
          :meth:`runall` or :meth:`run_irga`.

        Examples
        --------
        Instantiate with default settings:

        >>> flux = CalcFlux()

        Change the sensor height and specify a KH-20 hygrometer setup:

        >>> flux = CalcFlux(UHeight=2.0, meter_type="KH20", PathDist_U=0.1)
        """
        # ---- physical constants ------------------------------------------------
        self.Cp = None
        self.Rv = 461.51  # Water-vapour gas constant (J kg⁻¹ K⁻¹)
        self.Ru = 8.3143  # Universal gas constant (J mol⁻¹ K⁻¹)
        self.Cpd = 1005.0  # Specific heat of dry air (J kg⁻¹ K⁻¹)
        self.Rd = 287.05  # Dry-air gas constant (J kg⁻¹ K⁻¹)
        self.md = 0.02896  # Dry-air molar mass (kg mol⁻¹)
        self.Co = 0.21  # Atmospheric O₂ molar fraction
        self.Mo = 0.032  # O₂ molar mass (kg mol⁻¹)

        # ---- thermodynamic & spectral constants --------------------------------
        self.Cpw = 1952.0  # c_p of H₂O vapour (J kg⁻¹ K⁻¹)
        self.Cw = 4218.0  # c_p of liquid water (J kg⁻¹ K⁻¹)
        self.epsilon = 18.016 / 28.97
        self.g = 9.81  # Acceleration due to gravity (m s⁻²)
        self.von_karman = 0.41  # von Kármán constant
        self.MU_WPL = 28.97 / 18.016
        self.Omega = 7.292e-5  # Earth’s angular velocity (rad s⁻¹)
        self.Sigma_SB = 5.6718e-8  # Stefan–Boltzmann constant (J K⁻⁴ m⁻² s⁻¹)

        # ---- instrument configuration ------------------------------------------
        self.meter_type = "IRGASON"  # {'IRGASON', 'KH20'} supported
        self.XKH20 = 1.412
        self.XKwC1 = -0.152214126
        self.XKwC2 = -0.001667836
        self.sonic_dir = 225.0  # deg clockwise from N
        self.UHeight = 3.52  # m
        self.PathDist_U = 0.0  # m separation sonic–hygrometer

        # ---- processing options -------------------------------------------------
        self.lag = 10
        self.direction_bad_min = 0.0
        self.direction_bad_max = 360.0
        self.Kw = 1.0
        self.Ko = -0.0045

        # ---- run-time containers & state ----------------------------------------
        self.covar: dict[str, float] = {}
        self.avgvals: dict[str, float] = {}
        self.stdvals: dict[str, float] = {}
        self.errvals: dict[str, float] = {}

        self.cosv = self.sinv = self.sinTheta = self.cosTheta = None

        self.despikefields = [
            "Ux",
            "Uy",
            "Uz",
            "Ts",
            "volt_KH20",
            "Pr",
            "Rh",
            "pV",
        ]
        self.wind_compass = None
        self.pathlen = None
        self.df = None

        # ---- allow user overrides via kwargs ------------------------------------
        self.__dict__.update(kwargs)

        # List of common variables and their units
        self.parameters = {
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

    def runall(self, df: pd.DataFrame) -> pd.Series:
        """
        Runs the complete eddy-covariance processing chain and returns
        aggregated fluxes plus key diagnostics for the supplied time series.

        Parameters
        ----------
        df : pandas.DataFrame
            Time-indexed dataframe of high-frequency (≥ 10 Hz) measurements.
            After :meth:`renamedf` standardises column names the *minimum*
            required fields are

            =================  ============================================
            Column             Meaning / units
            -----------------  --------------------------------------------
            ``Ux``, ``Uy``, ``Uz``   Wind components (m s⁻¹)
            ``Ts``                   Sonic temperature (°C)
            ``Ta``                   Reference air temperature (°C)
            ``Pr``                   Air pressure (kPa)
            ``Ea`` **or** ``pV``     Actual vapour pressure (kPa) — if absent
                                     it will be derived from ``LnKH`` or
                                     ``volt_KH20`` when *meter_type* is
                                     ``"KH20"``.
            =================  ============================================

            Optional but supported fields include
            ``LnKH`` / ``volt_KH20`` (krypton hygrometer signal),
            ``Rh`` (relative humidity %), and ``Sd`` (saturation deficit).

        Returns
        -------
        pandas.Series
            A 13-element vector containing mean fluxes and diagnostics for
            the input period:

            =============  ==================================================
            Key            Description
            ------------  --------------------------------------------------
            ``Ta``         Mean air temperature (°C)
            ``Td``         Dew-point temperature (°C)
            ``D``          Vapour-pressure deficit (kPa)
            ``Ustr``       Friction velocity (m s⁻¹)
            ``zeta``       Stability parameter *z/L* (dimensionless)
            ``H``          Sensible heat flux (W m⁻²)
            ``StDevUz``    σᵥ — st. dev. vertical wind (m s⁻¹)
            ``StDevTa``    σ_T — st. dev. air temperature (K)
            ``direction``  Mean wind direction (° clockwise from north)
            ``exchange``   Scalar exchange coefficient (kg m⁻² s⁻¹)
            ``lambdaE``    Latent heat flux (W m⁻²)
            ``ET``         Evapotranspiration (mm d⁻¹)
            ``Uxy``        Mean rotated horizontal wind speed (m s⁻¹)
            =============  ==================================================

        Notes
        -----
        1. **No high-pass filtering** or detrending is applied to the raw data.
        2. Covariances are evaluated at the time-lag that maximises their
           absolute value within ``±self.lag`` samples.
        3. Wind vectors undergo double (triple) rotation into a flow-aligned
           frame; spectral attenuation is corrected using Massman (2000, 2001);
           density corrections follow Webb–Pearman–Leuning (1980).
        4. The routine depends on numerous class attributes (sensor geometry,
           path separation, etc.). Set them before calling :meth:`runall`.

        See Also
        --------
        renamedf : Harmonises input column names.
        calculated_parameters : Adds derived thermodynamic variables.
        coord_rotation : Performs triple rotation of wind vectors.
        webb_pearman_leuning : Applies WPL density corrections.

        Examples
        --------
        >>> calc = CalcFlux(UHeight=3.5, meter_type="IRGASON")
        >>> df_raw = pd.read_csv("flux_2025-06-21_1330.csv",
        ...                      index_col=0, parse_dates=True)
        >>> flux = calc.runall(df_raw)
        >>> float(flux["lambdaE"])
        138.5
        """

        df = self.renamedf(df)

        if "Ea" in df.columns:
            pass
        else:
            df["Ea"] = self.tetens(df["Ta"].to_numpy())

        if self.meter_type == "IRGASON":
            pass
        else:
            if "LnKH" in df.columns:
                pass
            elif "volt_KH20" in df.columns:
                df["LnKH"] = np.log(df["volt_KH20"].to_numpy())
            # Calculate the Correct XKw Value for KH20
            XKw = self.XKwC1 + 2 * self.XKwC2 * (df["pV"].mean() * 1000.0)
            self.Kw = XKw / self.XKH20
            # TODO Calc pV from lnKH20 and add to dataframe as variable

        for col in self.despikefields:
            if col in df.columns:
                df[col] = self.despike_quart_filter(df[col])

        # Convert Sonic and Air Temperatures from Degrees C to Kelvin
        df = _assign(df, Ts=self.convert_CtoK(_get_series(df,"Ts").to_numpy())) # type: ignore
        df = _assign(df, Ta=self.convert_CtoK(_get_series(df,"Ta").to_numpy())) # type: ignore

        # Remove shadow effects of the CSAT (this is also done by the CSAT Firmware)
        df["Ux"], df["Uy"], df["Uz"] = self.shadow_correction(
            df["Ux"].to_numpy(), df["Uy"].to_numpy(), df["Uz"].to_numpy()
        )
        self.avgvals = df.mean().to_dict()

        # Calculate Sums and Means of Parameter Arrays
        df = self.calculated_parameters(df)

        # Calculate Covariances (Maximum Furthest From Zero With Sign in Lag Period)
        self.calc_covar(
            df["Ux"].to_numpy(),
            df["Uy"].to_numpy(),
            df["Uz"].to_numpy(),
            df["Ts"].to_numpy(),
            df["Q"].to_numpy(),
            df["pV"].to_numpy(),
        )

        # Calculate max variance to close separation between sensors
        velocities = {
            "Ux": df["Ux"].to_numpy(),
            "Uy": df["Uy"].to_numpy(),
            "Uz": df["Uz"].to_numpy(),
        }

        covariance_variables = {
            "Ux": df["Ux"].to_numpy(),
            "Uy": df["Uy"].to_numpy(),
            "Uz": df["Uz"].to_numpy(),
            "Ts": df["Ts"].to_numpy(),
            "pV": df["pV"].to_numpy(),
            "Q": df["Q"].to_numpy(),
            "Sd": df["Sd"].to_numpy(),
        }

        # This iterates through the velocities and calculates the maximum covariance between
        # the velocity and the other variables
        for ik, iv in velocities.items():
            for jk, jv in covariance_variables.items():
                self.covar[f"{ik}-{jk}"] = self.calc_max_covariance(iv, jv)[0][1] # type: ignore

        self.covar["Ts-Q"] = self.calc_max_covariance(df["Ts"], df["Q"], self.lag)[0][1] # type: ignore

        # Traditional Coordinate Rotation
        cosv, sinv, sinTheta, cosTheta, Uxy, Uxyz = self.coord_rotation(df)

        df = self.rotate_velocity_values(df, "Ux", "Uy", "Uz")

        # Find the Mean Squared Error of Velocity Components and Humidity
        self.UxMSE = self.calc_MSE(df["Ux"])
        self.UyMSE = self.calc_MSE(df["Uy"])
        self.UzMSE = self.calc_MSE(df["Uz"])
        self.QMSE = self.calc_MSE(df["Q"])

        # Correct Covariances for Coordinate Rotation
        self.covar_coord_rot_correction(cosv, sinv, sinTheta, cosTheta)

        Ustr = np.sqrt(self.covar["Uxy-Uz"])

        # Find Average Air Temperature From Average Sonic Temperature
        Tsa = self.calc_Tsa(df["Ts"].mean(), df["Pr"].mean(), df["pV"].mean())

        # Calculate the Latent Heat of Vaporization (eq. 2.57 in Foken)
        lamb = 2500800 - 2366.8 * (self.convert_KtoC(Tsa))

        # Determine Vertical Wind and Water Vapor Density Covariance
        # Uz_pV = (self.covar["Uz-pV"] / XKw) / 1000

        # Calculate the Correct Average Values of Some Key Parameters
        self.Cp = self.Cpd * (1 + 0.84 * df["Q"].mean())
        self.pD = (df["Pr"].mean() - df["E"].mean()) / (self.Rd * Tsa)
        self.p = self.pD + df["pV"].mean()

        # Calculate Variance of Air Temperature From Variance of Sonic Temperature
        StDevTa = np.sqrt(
            self.covar["Ts-Ts"]
            - 1.02 * df["Ts"].mean() * self.covar["Ts-Q"]
            - 0.2601 * self.QMSE * df["Ts"].mean() ** 2
        )
        Uz_Ta = self.covar["Uz-Ts"] - 0.07 * lamb * self.covar["Uz-pV"] / (
            self.p * self.Cp
        )

        # Determine Saturation Vapor Pressure of the Air Using Highly Accurate Wexler's Equations Modified by Hardy
        Td = self.calc_Td_dewpoint(df["E"].mean())
        D = self.calc_Es(Tsa) - df["E"].mean()
        S = (
            self.calc_Q(df["Pr"].mean(), self.calc_Es(Tsa + 1))
            - self.calc_Q(df["Pr"].mean(), self.calc_Es(Tsa - 1))
        ) / 2

        # Determine Wind Direction
        Ux_avg = np.mean(df["Ux"].to_numpy())
        Uy_avg = np.mean(df["Uy"].to_numpy())
        Uz_avg = np.mean(df["Uz"].to_numpy())

        pathlen, direction = self.determine_wind_dir(Ux_avg, Uy_avg)

        # Calculate the Average and Standard Deviations of the Rotated Velocity Components
        StDevUz = df["Uz"].std()
        UMean = Ux_avg * cosTheta * cosv + Uy_avg * cosTheta * sinv + Uz_avg * sinTheta

        # Frequency Response Corrections (Massman, 2000 & 2001)
        tauB = 3600 / 2.8
        tauEKH20 = np.sqrt((0.01 / (4 * UMean)) ** 2 + (pathlen / (1.1 * UMean)) ** 2)
        tauETs = np.sqrt((0.1 / (8.4 * UMean)) ** 2)
        tauEMomentum = np.sqrt((0.1 / (5.7 * UMean)) ** 2 + (0.1 / (2.8 * UMean)) ** 2)

        # Calculate ζ and Correct Values of Uᕽ and Uz_Ta
        L = self.calc_L(Ustr, Tsa, Uz_Ta)
        alpha, X = self.calc_AlphX(L)
        fX = X * UMean / self.UHeight
        B = 2 * np.pi * fX * tauB
        momentum = 2 * np.pi * fX * tauEMomentum
        _Ts = 2 * np.pi * fX * tauETs
        _KH20 = 2 * np.pi * fX * tauEKH20
        Ts = self.correct_spectral(B, alpha, _Ts)
        self.covar["Uxy_Uz"] /= self.correct_spectral(B, alpha, momentum)
        Ustr = np.sqrt(self.covar["Uxy_Uz"])

        # Recalculate L With New Uᕽ and Uz_Ta, and Calculate High Frequency Corrections
        L = self.calc_L(Ustr, Tsa, Uz_Ta / Ts)
        alpha, X = self.calc_AlphX(L)
        Ts = self.correct_spectral(B, alpha, _Ts)
        KH20 = self.correct_spectral(B, alpha, _KH20)

        # Correct the Covariance Values
        Uz_Ta /= Ts
        self.covar["Uz-pV"] /= KH20
        self.covar["Uxy_Uz"] /= self.correct_spectral(B, alpha, momentum)
        Ustr = np.sqrt(self.covar["Uxy_Uz"])
        self.covar["Uz_Sd"] /= KH20
        exchange = ((self.p * self.Cp) / (S + self.Cp / lamb)) * self.covar["Uz_Sd"]

        # KH20 Oxygen Correction
        self.covar["Uz-pV"] += self.correct_KH20(Uz_Ta, df["Pr"].mean(), Tsa)

        # Calculate New H and LE Values
        H = self.p * self.Cp * Uz_Ta
        lambdaE = lamb * self.covar["Uz-pV"]

        # Webb, Pearman and Leuning Correction
        pVavg = np.mean(df["pV"].to_numpy())
        lambdaE = self.webb_pearman_leuning(
            lamb, Tsa, pVavg, Uz_Ta, self.covar["Uz-pV"]
        )

        # Finish Output
        Tsa = self.convert_KtoC(Tsa)
        Td = self.convert_KtoC(Td)
        zeta = self.UHeight / L
        ET = lambdaE * self.get_Watts_to_H2O_conversion_factor(
            Tsa,
            (df.last_valid_index() - df.first_valid_index())
            / pd.to_timedelta(1, unit="D"),
        )
        # Out.Parameters = CWP
        self.columns = [
            "Ta",
            "Td",
            "D",
            "Ustr",
            "zeta",
            "H",
            "StDevUz",
            "StDevTa",
            "direction",
            "exchange",
            "lambdaE",
            "ET",
            "Uxy",
        ]
        self.out = [
            Tsa,
            Td,
            D,
            Ustr,
            zeta,
            H,
            StDevUz,
            StDevTa,
            direction,
            exchange,
            lambdaE,
            ET,
            Uxy,
        ]
        return pd.Series(data=self.out, index=self.columns)

    def run_irga(self, df: pd.DataFrame) -> pd.Series:
        """
        Execute the full eddy-covariance processing chain for data collected
        with an **integrated open-path IRGA/sonic anemometer system**
        (e.g. Campbell Sci. *IRGASON* or LI-COR *LI-7500 + CSAT3*).

        The routine performs despiking, coordinate rotation, high-frequency
        spectral corrections, density corrections (WPL), oxygen absorption
        adjustments, and derives all standard micrometeorological fluxes for
        the averaging period contained in *df*.

        Parameters
        ----------
        df : pandas.DataFrame
            High-frequency (≥ 10 Hz) time-indexed measurements.  After
            :meth:`renamedf` harmonises the column names the **minimum
            required fields** are

            ===============  ==============================  ============
            Column           Physical quantity              Units
            ---------------  ------------------------------  ------------
            ``Ux``, ``Uy``, ``Uz``   Orthogonal wind speed          m s⁻¹
            ``Ts``            Sonic (virtual) temperature    °C
            ``Ta``            Probe air temperature (HMP)    °C (optional,
                               — improves dew-point estimate)
            ``Pr``            Ambient pressure               kPa
            ``pV``            Water-vapour density           g m⁻³
                              *(from IRGA – converted to kg m⁻³ internally)*
            ===============  ==============================  ============

            Optional yet recognised fields include

            * ``Q``   – specific humidity (kg kg⁻¹); calculated if absent.
            * ``Sd``  – scalar entropy term used in the H/LE exchange
              coefficient.
            * ``Ea``  – vapour pressure, if *pV* is not supplied.
            * ``LnKH`` / ``volt_KH20`` – retained for mixed KH-20 deployments
              but **ignored** by this IRGA pathway.

        Returns
        -------
        pandas.Series
            A 13-element vector of period-mean fluxes and diagnostics
            (index order is preserved for easy concatenation):

            ===========  ============================================  Units
            -----------  --------------------------------------------  ------
            ``Ta``       Mean air temperature                          °C
            ``Td``       Dew-point temperature                         °C
            ``D``        Vapour-pressure deficit                       kPa
            ``Ustr``     Friction velocity *u*★                        m s⁻¹
            ``zeta``     Stability parameter *z/L*                     —
            ``H``        Sensible-heat flux                            W m⁻²
            ``StDevUz``  σ_w – std. dev. vertical wind                 m s⁻¹
            ``StDevTa``  σ_T – std. dev. air temperature               K
            ``direction``Mean wind direction (met. convention)         °
            ``exchange`` Scalar exchange coefficient                   kg m⁻² s⁻¹
            ``lambdaE``  Latent-heat flux (LE)                         W m⁻²
            ``ET``       Evapotranspiration rate                       mm d⁻¹
            ``Uxy``      Mean rotated horizontal wind speed            m s⁻¹
            ===========  ============================================  ======

        Notes
        -----
        1. **Despiking** uses a median-modified Z-score filter followed by
           EWMA gap-filling; the window size is controlled by
           :pyattr:`despikefields`.
        2. Wind vectors undergo a *double/triple* rotation to align with the
           mean flow (Kaimal & Finnigan, 1994).
        3. **Spectral attenuation** is corrected after Massman (2000, 2001);
           the frequency response time constants (τ) depend on the mean wind
           speed and path length determined in :meth:`determine_wind_dir`.
        4. **Density effects** are removed with the Webb-Pearman-Leuning (WPL)
           formulation (Webb *et al.*, 1980).
        5. Water-vapour absorption by O₂ in the Krypton hygrometer
           (if present) is corrected with coefficients
           *Kw* and *Ko* (Oren *et al.*, 1998).
        6. The averaging interval (and, therefore, the ET conversion from
           W m⁻² to mm d⁻¹) is inferred from the time stamps in *df*.

        See Also
        --------
        runall : KH-20 (krypton) hygrometer processing pathway.
        renamedf : Column-name harmonisation helper.
        coord_rotation : Implements double/triple rotation.
        webb_pearman_leuning : WPL density-correction sub-routine.

        References
        ----------
        Kaimal, J. C., & Finnigan, J. J. (1994). *Atmospheric Boundary Layer
        Flows*. Oxford UP.
        Massman, W. J. (2000, 2001) Spectral correction papers.
        Webb, E. K., Pearman, G. I., & Leuning, R. (1980). *QJRMS*, **106**,
        85-102.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> import polars as pl
import pandas as pd  # optional compat
        >>> raw = pd.read_csv("2025-06-21_1330.csv", index_col=0, parse_dates=True)
        >>> fluxcalc = CalcFlux(meter_type="IRGASON", UHeight=3.5)
        >>> period_fluxes = fluxcalc.run_irga(raw)
        >>> period_fluxes["lambdaE"]
        140.2  # W m⁻² (example value)
        """
        df = self.renamedf(df)

        for col in self.despikefields:
            if col in df.columns:
                df[col + "_ro"] = self.despike_med_mod(df[col])

        # Convert Sonic and Air Temperatures from Degrees C to Kelvin
        df.loc[:, "Ts"] = self.convert_CtoK(df["Ts_ro"].to_numpy())
        df = _assign(df, Ta=self.convert_CtoK(_get_series(df,"Ta").to_numpy()))

        # Remove shadow effects of the CSAT (this is also done by the CSAT Firmware)
        df["Ux"], df["Uy"], df["Uz"] = self.shadow_correction(
            df["Ux_ro"].to_numpy(), df["Uy_ro"].to_numpy(), df["Uz_ro"].to_numpy()
        )
        # print(df[['Ux','Uy','Uz']])
        self.avgvals = df.mean().to_dict()

        # Calculate Sums and Means of Parameter Arrays
        # convert air pressure from kPa to Pa
        df["Pr"] = df["Pr_ro"] * 1000.0
        # convert pV to g/m-3
        df["pV"] = df["pV_ro"] * 0.001

        df["E"] = self.calc_E(df["pV"], df["Ts"])
        # convert air pressure from kPa to Pa
        df["Q"] = self.calc_Q(df["Pr"], df["E"])
        df["Tsa"] = self.calc_Tsa(df["Ts"], df["Q"])
        # df['Tsa2'] = self.calc_tc_air_temp_sonic(df['Ts'], df['pV'], df['Pr'])
        df["Es"] = self.calc_Es(df["Tsa"])
        df["Sd"] = self.calc_Q(df["Pr"], self.calc_Es(df["Tsa"])) - df["Q"]

        # Calculate Covariances (Maximum Furthest From Zero With Sign in Lag Period)
        self.calc_covar(
            df["Ux"].to_numpy(),
            df["Uy"].to_numpy(),
            df["Uz"].to_numpy(),
            df["Ts"].to_numpy(),
            df["Q"].to_numpy(),
            df["pV"].to_numpy(),
        )

        # Calculate max variance to close separation between sensors
        velocities = {
            "Ux": _interpolate_bfill_ffill(_get_series(df, "Ux"))
            .to_numpy(),
            "Uy": _interpolate_bfill_ffill(_get_series(df, "Uy"))
            .to_numpy(),
            "Uz": _interpolate_bfill_ffill(_get_series(df, "Uz"))
            .to_numpy(),
        }

        covariance_variables = {
            "Ux": _interpolate_bfill_ffill(_get_series(df, "Ux"))
            .to_numpy(),
            "Uy": _interpolate_bfill_ffill(_get_series(df, "Uy"))
            .to_numpy(),
            "Uz": _interpolate_bfill_ffill(_get_series(df, "Uz"))
            .to_numpy(),
            "Ts": _interpolate_bfill_ffill(_get_series(df, "Ts"))
            .to_numpy(),
            "Tsa": _interpolate_bfill_ffill(_get_series(df, "Tsa"))
            .to_numpy(),
            "pV": _interpolate_bfill_ffill(_get_series(df, "pV"))
            .to_numpy(),
            "Q": _interpolate_bfill_ffill(_get_series(df, "Q"))
            .to_numpy(),
            "Sd": _interpolate_bfill_ffill(_get_series(df, "Sd"))
            .to_numpy(),
        }

        # This iterates through the velocities and calculates the maximum covariance between
        # the velocity and the other variables
        for ik, iv in velocities.items():
            for jk, jv in covariance_variables.items():
                try:
                    self.covar[f"{ik}-{jk}"] = self.calc_max_covariance(iv, jv)[0][1]
                except IndexError:
                    print(f"index error {ik}-{jk}")
                    self.covar[f"{ik}-{jk}"] = self.calc_cov(iv, jv)

        try:
            self.covar["Ts-Q"] = self.calc_max_covariance(
                _interpolate_bfill_ffill(_get_series(df, "Ts")),
                _interpolate_bfill_ffill(_get_series(df, "Q")),
                self.lag,
            )[0][1] # type: ignore
        except IndexError:
            self.covar["Ts-Q"] = self.calc_cov(iv, jv)

        # Traditional Coordinate Rotation
        cosv, sinv, sinTheta, cosTheta, Uxy, Uxy_Uz = self.coord_rotation(df)

        df = self.rotate_velocity_values(df, "Ux", "Uy", "Uz")

        # Find the Mean Squared Error of Velocity Components and Humidity
        for varib in ["Ux", "Uy", "Uz", "Q", "Ts", "Tsa"]:
            self.errvals[varib] = self.calc_MSE(df[varib].to_numpy())

        # Correct Covariances for Coordinate Rotation
        self.covar_coord_rot_correction(cosv, sinv, sinTheta, cosTheta)

        Ustr = np.sqrt(self.covar["Uxy-Uz"])

        # Find Average Air Temperature From Average Sonic Temperature
        Tsa = self.calc_Tsa_sonic_temp(
            df["Ts"].mean(), df["Pr"].mean(), df["pV"].mean()
        )

        # Calculate the Latent Heat of Vaporization (eq. 2.57 in Foken)
        lamb = 2500800 - 2366.8 * (self.convert_KtoC(Tsa))

        StDevTa = np.sqrt(
            np.abs(
                self.covar["Ts-Ts"]
                - 1.02 * df["Ts"].mean() * self.covar["Ts-Q"]
                - 0.2601 * self.errvals["Q"] * df["Ts"].mean() ** 2
            )
        )

        # Calculate the Correct Average Values of Some Key Parameters
        self.Cp = self.Cpd * (1 + 0.84 * df["Q"].mean())
        self.pD = (df["Pr"].mean() - df["E"].mean()) / (self.Rd * Tsa)
        self.p = self.pD + df["pV"].mean()

        # Calculate Variance of Air Temperature From Variance of Sonic Temperature
        StDevTa = np.sqrt(
            np.abs(
                self.covar["Ts-Ts"]
                - 1.02 * df["Ts"].mean() * self.covar["Ts-Q"]
                - 0.2601 * self.errvals["Q"] * df["Ts"].mean() ** 2
            )
        )

        Uz_Ta = self.covar["Uz-Ts"] - 0.07 * lamb * self.covar["Uz-pV"] / (
            self.p * self.Cp
        )

        # Determine Saturation Vapor Pressure of the Air Using Highly Accurate Wexler's Equations Modified by Hardy
        Td = self.calc_Td_dewpoint(df["E"].mean())
        D = self.calc_Es(Tsa) - df["E"].mean()
        S = (
            self.calc_Q(df["Pr"].mean(), self.calc_Es(Tsa + 1))
            - self.calc_Q(df["Pr"].mean(), self.calc_Es(Tsa - 1))
        ) / 2

        # Determine Wind Direction
        Ux_avg = np.mean(df["Uxr"].to_numpy())
        Uy_avg = np.mean(df["Uyr"].to_numpy())
        Uz_avg = np.mean(df["Uzr"].to_numpy())

        pathlen, direction = self.determine_wind_dir(Ux_avg, Uy_avg)

        # Calculate the Average and Standard Deviations of the Rotated Velocity Components
        StDevUz = df["Uz"].std()
        UMean = Ux_avg * cosTheta * cosv + Uy_avg * cosTheta * sinv + Uz_avg * sinTheta

        # Frequency Response Corrections (Massman, 2000 & 2001)
        tauB = 3600 / 2.8
        tauEKH20 = np.sqrt((0.01 / (4 * UMean)) ** 2 + (pathlen / (1.1 * UMean)) ** 2)
        tauETs = np.sqrt((0.1 / (8.4 * UMean)) ** 2)
        tauEMomentum = np.sqrt((0.1 / (5.7 * UMean)) ** 2 + (0.1 / (2.8 * UMean)) ** 2)

        # Calculate ζ and Correct Values of Uᕽ and Uz_Ta
        L = self.calc_L(Ustr, Tsa, Uz_Ta)
        alpha, X = self.calc_AlphX(L)
        fX = X * UMean / self.UHeight
        B = 2 * np.pi * fX * tauB
        momentum = 2 * np.pi * fX * tauEMomentum
        _Ts = 2 * np.pi * fX * tauETs
        _KH20 = 2 * np.pi * fX * tauEKH20
        Ts = self.correct_spectral(B, alpha, _Ts)
        self.covar["Uxy-Uz"] /= self.correct_spectral(B, alpha, momentum)
        Ustr = np.sqrt(self.covar["Uxy-Uz"])

        # Recalculate L With New Uᕽ and Uz_Ta, and Calculate High Frequency Corrections
        L = self.calc_L(Ustr, Tsa, Uz_Ta / Ts)
        alpha, X = self.calc_AlphX(L)
        Ts = self.correct_spectral(B, alpha, _Ts)
        KH20 = self.correct_spectral(B, alpha, _KH20)

        # Correct the Covariance Values
        Uz_Ta /= Ts
        self.covar["Uz-pV"] /= KH20
        self.covar["Uxy-Uz"] /= self.correct_spectral(B, alpha, momentum)
        Ustr = np.sqrt(self.covar["Uxy-Uz"])
        self.covar["Uz-Sd"] /= KH20
        exchange = ((self.p * self.Cp) / (S + self.Cp / lamb)) * self.covar["Uz-Sd"]

        # KH20 Oxygen Correction
        self.covar["Uz-pV"] += self.correct_KH20(Uz_Ta, df["Pr"].mean(), Tsa)

        # Calculate New H and LE Values
        H = self.p * self.Cp * Uz_Ta
        lambdaE = lamb * self.covar["Uz-pV"]

        # Webb, Pearman and Leuning Correction
        pVavg = np.mean(df["pV"].to_numpy())
        lambdaE = self.webb_pearman_leuning(
            lamb, Tsa, pVavg, Uz_Ta, self.covar["Uz-pV"]
        )

        # Finish Output
        Tsa = self.convert_KtoC(Tsa)
        Td = self.convert_KtoC(Td)
        zeta = self.UHeight / L
        ET = lambdaE * self.get_Watts_to_H2O_conversion_factor(
            Tsa,
            (df.last_valid_index() - df.first_valid_index())
            / pd.to_timedelta(1, unit="D"),
        )
        # Out.Parameters = CWP
        self.columns = [
            "Ta",
            "Td",
            "D",
            "Ustr",
            "zeta",
            "H",
            "StDevUz",
            "StDevTa",
            "direction",
            "exchange",
            "lambdaE",
            "ET",
            "Uxy",
        ]
        self.out = [
            Tsa,
            Td,
            D,
            Ustr,
            zeta,
            H,
            StDevUz,
            StDevTa,
            direction,
            exchange,
            lambdaE,
            ET,
            Uxy,
        ]
        return pd.Series(data=self.out, index=self.columns)

    def determine_wind_dir(
        self,
        uxavg: float | None = None,
        uyavg: float | None = None,
        update_existing_vel: bool = False,
    ):
        """
        Compute the **mean wind direction** (meteorological convention) and the
        effective horizontal **path-length separation** between the sonic and
        hygrometer (or IRGA) for the current averaging period.

        The routine uses the supplied *ū* (longitudinal) and *v̄* (lateral)
        wind-speed means—or those already stored in :pyattr:`avgvals`—to derive

        1. *wind_compass* : mean wind direction, ° clockwise from **true north**
           (corrected for the instrument’s azimuth ``self.sonic_dir``).
        2. *pathlen* : the horizontal separation between sensors **projected
           onto the mean wind vector**, used later for high-frequency spectral
           corrections.

        Parameters
        ----------
        uxavg : float or None, default ``None``
            Period-mean *u* component (m s⁻¹).  If *None*, the method looks
            for ``"Ux"`` in :pyattr:`avgvals`.
        uyavg : float or None, default ``None``
            Period-mean *v* component (m s⁻¹).  If *None*, the method looks
            for ``"Uy"`` in :pyattr:`avgvals`.
        update_existing_vel : bool, default ``False``
            If *True*, any user-supplied ``uxavg``/``uyavg`` overwrite the
            values currently held in :pyattr:`avgvals`.

        Returns
        -------
        tuple (pathlen, wind_compass)
            **pathlen** : float
                Lateral separation between sensors projected onto the wind
                vector, *m*.

            **wind_compass** : float
                Mean wind direction, degrees clockwise from north.

        Notes
        -----
        * The method **does not** calculate the mean velocities itself; you
          must supply them or run a routine (e.g. :meth:`rotated_components_statistics`)
          that populates :pyattr:`avgvals`.
        * The sonic’s physical orientation is given by ``self.sonic_dir`` and
          is subtracted from the instrument-frame wind angle to yield
          meteorological bearing.
        * The projected path length is
          ``|PathDist_U| × |sin(wind_compass)|`` :contentReference[oaicite:0]{index=0}.

        Examples
        --------
        >>> calc = CalcFlux(PathDist_U=0.1, sonic_dir=225.0)
        >>> # Assume you already stored mean velocities in calc.avgvals
        >>> calc.avgvals.update({"Ux": 1.2, "Uy": 0.8})
        >>> path, wd = calc.determine_wind_dir()
        >>> round(path, 3), round(wd, 1)
        (0.058, 292.6)
        """

        if uxavg:
            if update_existing_vel:
                self.avgvals["Ux"] = uxavg
        else:
            if "Ux" in self.avgvals.keys():
                uxavg = self.avgvals["Ux"]
            else:
                print("Please calculate wind velocity averages")
        if uyavg:
            if update_existing_vel:
                self.avgvals["Uy"] = uyavg
        else:
            if "Uy" in self.avgvals.keys():
                uyavg = self.avgvals["Uy"]
            else:
                print("Please calculate wind velocity averages")

        if uyavg and uxavg:
            self.v = np.sqrt(uxavg**2 + uyavg**2)
            wind_dir = np.arctan(uyavg / uxavg) * 180.0 / np.pi
            if uxavg < 0:
                if uyavg >= 0:
                    wind_dir += wind_dir + 180.0
                else:
                    wind_dir -= wind_dir - 180.0
            wind_compass = -1.0 * wind_dir + self.sonic_dir
            if wind_compass < 0:
                wind_compass += 360
            elif wind_compass > 360:
                wind_compass -= 360

            self.wind_compass = wind_compass
            # Calculate the Lateral Separation Distance Projected Into the Mean Wind Direction
            self.pathlen = self.PathDist_U * np.abs(
                np.sin((np.pi / 180) * wind_compass)
            )
            return self.pathlen, self.wind_compass

    def covar_coord_rot_correction(
        self,
        cosν: float | None = None,
        sinv: float | None = None,
        sinTheta: float | None = None,
        cosTheta: float | None = None,
    ):
        """
        Apply the **final covariance corrections** that arise after the
        double/triple coordinate rotation of wind vectors into the streamlined
        (earth-aligned) reference frame.

        The method updates entries in the instance dictionaries
        :pyattr:`covar` and :pyattr:`errvals` so that all covariances refer to
        the *rotated* axes.  It should be called **immediately after**
        :meth:`coord_rotation`, which populates the rotation angles stored in
        ``self.cosv``, ``self.sinv``, ``self.cosTheta`` and ``self.sinTheta``.

        Parameters
        ----------
        cosν : float, optional
            Cosine of the *v*-rotation angle (second rotation, about the
            instrument **y**-axis).  If ``None`` the value cached in
            ``self.cosv`` is used.
        sinv : float, optional
            Sine of the *v*-rotation angle.  Defaults to ``self.sinv``.
        sinTheta : float, optional
            Sine of the *θ*-rotation angle (third rotation, about the rotated
            **x′** axis).  Defaults to ``self.sinTheta``.
        cosTheta : float, optional
            Cosine of the *θ*-rotation angle.  Defaults to ``self.cosTheta``.

        Returns
        -------
        None
            The function operates *in-place*: corrected covariances replace the
            pre-rotation entries in :pyattr:`covar` and additional rotation-
            error terms are added to :pyattr:`errvals` where required.

        Modifies
        --------
        covar : dict
            Keys such as ``"Uz-Tsa"``, ``"Uz-Q"``, ``"Ux-Uz"``, ``"Uy-Uz"``,
            ``"Uz-Sd"`` and the derived magnitude ``"Uxy-Uz"`` are overwritten
            with their rotation-corrected values.
        errvals : dict
            Contributes small error terms arising from the rotation of the
            variance tensor (see Kaimal & Finnigan, 1994).

        Notes
        -----
        The corrections follow standard eddy-covariance practice (Kaimal &
        Finnigan, 1994; Wilczak *et al.*, 2001), accounting for the mixing of
        variances and covariances introduced by non-orthogonal rotation
        matrices.  Because the magnitude of the *θ*-rotation is usually only a
        few degrees, its influence on scalar covariances is often small, but
        must be included for accurate friction-velocity (*u*★) and scalar-flux
        estimates.

        Examples
        --------
        >>> calc = CalcFlux()
        >>> # 1) run coordinate rotation to obtain rotation angles
        >>> calc.coord_rotation(Ux, Uy, Uz)
        >>> # 2) form all raw covariances
        >>> calc.calc_covar(Ux, Uy, Uz, Ts, Q, pV)
        >>> # 3) correct the covariances for the rotation
        >>> calc.covar_coord_rot_correction()
        >>> corrected_Uz_Ts = calc.covar['Uz-Tsa']
        """

        if cosTheta is None:
            cosν = self.cosv
            cosTheta = self.cosTheta
            sinv = self.sinv
            sinTheta = self.sinTheta

        #
        Uz_Ts = (
            self.covar["Uz-Tsa"] * cosTheta
            - self.covar["Ux-Tsa"] * sinTheta * cosν
            - self.covar["Uy-Tsa"] * sinTheta * sinv
        )
        if np.abs(Uz_Ts) >= np.abs(self.covar["Uz-Tsa"]):
            self.covar["Uz-Tsa"] = Uz_Ts

        Uz_pV = (
            self.covar["Uz-pV"] * cosTheta
            - self.covar["Ux-pV"] * sinTheta * cosν
            - self.covar["Uy-pV"] * sinv * sinTheta
        )
        if np.abs(Uz_pV) >= np.abs(self.covar["Uz-pV"]):
            self.covar["Uz-pV"] = Uz_pV
        self.covar["Ux-Q"] = (
            self.covar["Ux-Q"] * cosTheta * cosν
            + self.covar["Uy-Q"] * cosTheta * sinv
            + self.covar["Uz-Q"] * sinTheta
        )
        self.covar["Uy-Q"] = self.covar["Uy-Q"] * cosν - self.covar["Uy-Q"] * sinv
        self.covar["Uz-Q"] = (
            self.covar["Uz-Q"] * cosTheta
            - self.covar["Ux-Q"] * sinTheta * cosν
            - self.covar["Uy-Q"] * sinv * sinTheta
        )
        self.covar["Ux-Uz"] = (
            self.covar["Ux-Uz"] * cosν * (cosTheta**2 - sinTheta**2)
            - 2 * self.covar["Ux-Uy"] * sinTheta * cosTheta * sinv * cosν
            + self.covar["Uy-Uz"] * sinv * (cosTheta**2 - sinTheta**2)
            - self.errvals["Ux"] * sinTheta * cosTheta * cosν**2
            - self.errvals["Uy"] * sinTheta * cosTheta * sinv**2
            + self.errvals["Uz"] * sinTheta * cosTheta
        )
        self.covar["Uy-Uz"] = (
            self.covar["Uy-Uz"] * cosTheta * cosν
            - self.covar["Ux-Uz"] * cosTheta * sinv
            - self.covar["Ux-Uy"] * sinTheta * (cosν**2 - sinv**2)
            + self.errvals["Ux"] * sinTheta * sinv * cosν
            - self.errvals["Uy"] * sinTheta * sinv * cosν
        )
        self.covar["Uz-Sd"] = (
            self.covar["Uz-Sd"] * cosTheta
            - self.covar["Ux-Sd"] * sinTheta * cosν
            - self.covar["Uy-Sd"] * sinv * sinTheta
        )
        self.covar["Uxy-Uz"] = np.sqrt(
            self.covar["Ux-Uz"] ** 2 + self.covar["Uy-Uz"] ** 2
        )

    def webb_pearman_leuning(
        self,
        lamb: float,
        Tsa: float,
        pVavg: float,
        Uz_Ta: float,
        Uz_pV: float,
    ) -> float:
        """
        Apply the **Webb–Pearman–Leuning (WPL) density correction** to a
        preliminary latent-heat (LE) estimate.

        The WPL adjustment compensates for the fact that open-path eddy-
        covariance sensors measure **mass mixing ratios** rather than true
        scalar fluxes.  Temperature-induced fluctuations of air density (ρ)
        and variations in dry-air/water-vapour composition both bias the raw
        covariance *w′ρ_v′*.  This routine implements the formulation in
        Webb *et al.* [1980] to convert the biased LE to a transport-corrected
        latent-heat flux.

        Parameters
        ----------
        lamb : float
            Latent heat of vaporisation, :math:`\\lambda`
            (≈ *2.45 × 10⁶ J kg⁻¹* near 20 °C).
        Tsa : float
            Period-mean air (sonic) temperature, **K**.  (Convert from °C
            beforehand.)
        pVavg : float
            Mean water-vapour *density* (ρ_v), **kg m⁻³**.
        Uz_Ta : float
            Covariance :math:`w′T′` between vertical wind speed *w* (m s⁻¹)
            and air temperature *T* (K).
        Uz_pV : float
            Covariance :math:`w′ρ_v′` between *w* and water-vapour density
            ρ_v (kg m⁻³).

        Returns
        -------
        float
            **LE** — latent-heat flux after WPL correction, **W m⁻²**.

        Notes
        -----
        * ``self.p`` – mean air pressure (Pa) – and ``self.Cp`` – specific heat
          of (moist) air (J kg⁻¹ K⁻¹) – **must** be set earlier in the
          processing chain.
        * ``self.pD`` is the *dry-air* density ρ_d (kg m⁻³); the hard-wired
          factor 1.6129 is :math:`1 + R_v / R_d`.
        * The denominator includes the small factor **0.07** that corrects for
          sensible-heat transport by water vapour.  Neglecting it can lead to
          ≈ 2 % error in humid conditions.
        * The correction is usually **≤ 5 %** for H₂O but can exceed **50 %**
          for CO₂ because scalar fluctuations are small relative to the mean
          concentration.

        References
        ----------
        Webb, E. K., Pearman, G. I., & Leuning, R. (1980).  *Correction of flux
        measurements for density effects due to heat and water vapour transfer*.
        **Quarterly Journal of the Royal Meteorological Society**, *106*, 85-100.

        Examples
        --------
        >>> calc = CalcFlux()
        >>> calc.p, calc.Cp, calc.pD = 91_500, 1005.0, 1.06  # Pa, J/kg/K, kg/m³
        >>> LE_raw = 125.   # W m⁻² from uncorrected w'ρ_v'
        >>> lam = 2.45e6    # J kg⁻¹
        >>> T_mean = 293.15 # K (20 °C)
        >>> rho_v = 0.012   # kg m⁻³
        >>> wT = -0.025     # m s⁻¹ K
        >>> wRv = -8.5e-4   # m s⁻¹ kg m⁻³
        >>> LE_corr = calc.webb_pearman_leuning(lam, T_mean, rho_v, wT, wRv)
        >>> round(LE_corr, 1)
        129.3
        """
        # --------------- original implementation (unchanged) ------------------
        # Webb, Pearman and Leuning Correction
        pCpTsa = self.p * self.Cp * Tsa
        pRatio = 1.0 + 1.6129 * (pVavg / self.pD)
        LE = (
            lamb
            * pCpTsa
            * pRatio
            * (Uz_pV + (pVavg / Tsa) * Uz_Ta)
            / (pCpTsa + lamb * pRatio * pVavg * 0.07)
        )
        return LE

    def calc_LnKh(self, mvolts: float | np.ndarray) -> float | np.ndarray:
        """
        Convert the raw KH-20 krypton-hygrometer signal (millivolts) to its
        natural logarithm, *ln(KH20)*.

        The KH-20 output follows Beer–Lambert absorption; taking the natural
        logarithm linearises the relationship between signal attenuation and
        water-vapour density, enabling subsequent calibration routines (e.g.
        :meth:`calc_pV_from_LnKh`).

        Parameters
        ----------
        mvolts : float or array_like
            KH-20 signal in **millivolts**. Accepts a scalar or any NumPy-
            compatible array; the operation is vectorised.

        Returns
        -------
        float or ndarray
            ``np.log(mvolts)`` — the natural logarithm of the input. The
            returned object retains the input type (scalar in → scalar out,
            array in → ndarray out).

        Notes
        -----
        * Input values **must be positive**; ``np.log`` will issue a warning
          and return *nan* for zero or negative entries.
        * Ensure any sensor offset has been removed and engineering units
          converted to mV before calling.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> calc.calc_LnKh(240.0)
        5.480638923341991
        >>> import numpy as np
        >>> v = np.array([220., 230., 240.])
        >>> calc.calc_LnKh(v)
        array([5.39362755, 5.43807931, 5.48063892])
        """
        return np.log(mvolts)

    def renamedf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonise column names in a raw Campbell-Scientific data frame so that
        downstream processing routines recognise the expected variable labels.

        The method performs a simple one-to-one rename (no unit conversion or
        reordering) according to the table below.  Columns not listed are left
        untouched.

        Mapping
        -------
        ==============  ======================  ==============================
        Raw column      Target column           Physical meaning / units
        --------------  ----------------------  ------------------------------
        ``T_SONIC``     ``Ts``                  Sonic (virtual) temperature, °C
        ``TA_1_1_1``    ``Ta``                  Probe air temperature, °C
        ``amb_press``   ``Pr``                  Ambient pressure, kPa
        ``PA``          ``Pr``                  ″ (alternate logger label)
        ``H2O_density`` ``pV``                  Water-vapour density, kg m⁻³
        ``RH_1_1_1``    ``Rh``                  Relative humidity, %
        ``t_hmp``       ``Ta``                  Air temperature from HMP, °C
        ``e_hmp``       ``Ea``                  Vapour pressure from HMP, kPa
        ``kh``          ``volt_KH20``           KH-20 hygrometer signal, mV
        ``q``           ``Q``                   Specific humidity, kg kg⁻¹
        ==============  ======================  ==============================

        Parameters
        ----------
        df : pandas.DataFrame
            Raw high-frequency data set exported from a Campbell Scientific
            datalogger (or compatible).  The index should already be time-
            stamped; otherwise, convert prior to further processing.

        Returns
        -------
        pandas.DataFrame
            *df* with columns renamed.  The operation is **non-destructive** –
            a new DataFrame is returned, leaving the original untouched.

        Notes
        -----
        * The mapping intentionally allows multiple aliases for the same
          target column (e.g. ``amb_press`` and ``PA`` both → ``Pr``).  If
          duplicates exist, the **last occurrence wins** according to Python
          dictionary order.
        * No attempt is made to fill missing columns; the subsequent methods
          will raise informative errors if required variables are absent.

        Examples
        --------
        >>> import polars as pl
import pandas as pd  # optional compat
        >>> raw = pd.DataFrame(
        ...     {"TIMESTAMP": ["2025-06-21 13:30:00.0"],
        ...      "T_SONIC": [23.4], "amb_press": [92.1], "H2O_density": [0.012]},
        ...     index=pd.to_datetime(["2025-06-21 13:30:00"])
        ... )
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> tidy = calc.renamedf(raw)
        >>> list(tidy.columns)
        ['TIMESTAMP', 'Ts', 'Pr', 'pV']
        """
        return df.rename(
            columns={
                "T_SONIC": "Ts",
                "TA_1_1_1": "Ta",
                "amb_press": "Pr",
                "PA": "Pr",
                "H2O_density": "pV",
                "RH_1_1_1": "Rh",
                "t_hmp": "Ta",
                "e_hmp": "Ea",
                "kh": "volt_KH20",
                "q": "Q",
            }
        )

        """
        Remove impulsive outliers (“spikes”) from a 1-D numeric series by
        thresholding departures from the global mean and linearly
        interpolating across flagged samples.

        The test is **not windowed**: the mean (μ) and standard deviation (σ)
        are computed over the *entire* array (ignoring ``NaN``), and any point
        whose absolute deviation exceeds *nstd × σ* is treated as a spike.

        Parameters
        ----------
        arr : array_like
            One-dimensional sequence of numeric values (list, NumPy array, or
            pandas Series).  The input is converted to a NumPy array via
            ``np.asarray``; original dtype is preserved where possible.
        nstd : float, default ``4.5``
            The spike threshold expressed in **standard deviations**.  Smaller
            values remove more points; larger values are more permissive.

        Returns
        -------
        ndarray
            A copy of *arr* with spikes replaced by values obtained from
            piece-wise **linear interpolation** between the nearest valid
            neighbours:

            * Interior NaNs are interpolated linearly.
            * NaNs at the ends are filled with the first/last valid value
              (behaviour of ``numpy.interp``).

        Notes
        -----
        * **Speed** – the routine is fully vectorised and fast for large
          arrays, but can be **over-aggressive** if the underlying signal has
          genuine high-frequency variability comparable to the spike size.
        * The method assumes *arr* is essentially one-dimensional.  For
          multi-dimensional despiking, iterate over the last axis or adapt a
          rolling filter.
        * Because the global μ and σ are used, the presence of broad trends or
          seasonality may reduce sensitivity to short spikes.  Pre-detrend or
          use a running‐window approach if necessary.

        Examples
        --------
        >>> import numpy as np
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> signal = np.array([1, 1.1, 0.9, 8.0, 1.0, 1.05])  # 8.0 is a spike
        >>> cleaned = calc.despike(signal, nstd=3)
        >>> cleaned
        array([1.  , 1.1 , 0.9 , 0.95, 1.  , 1.05])
        """

        stdd = np.nanstd(arr) * nstd
        avg = np.nanmean(arr)
        avgdiff = stdd - np.abs(arr - avg)
        y = np.where(avgdiff >= 0, arr, np.nan)
        nans, x = np.isnan(y), lambda z: z.nonzero()[0]
        if len(x(~nans)) > 0:
            y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y

    def despike_ewma_fb(
        self,
        df_column: pd.Series,
        span: int | float,
        delta: float,
    ) -> np.ndarray:
        """
        Remove impulsive outliers from a signal by comparing it with the mean of
        **forward-** and **backward-looking** exponential-weighted moving
        averages (EWMA).

        The method follows the approach outlined by *Müller (2016)* on
        Stack Overflow [link in *Notes*], where each point is flagged as an
        outlier if its absolute deviation from the *fb-EWMA* exceeds the
        user-supplied threshold *delta*.

        Parameters
        ----------
        df_column : pandas.Series
            One-dimensional time-series data containing potential spikes.
        span : int or float
            *Span* parameter for the EWMA (roughly the window length expressed
            in “effective” observations).  Passed directly to
            :meth:`pandas.Series.ewm`.
        delta : float
            Outlier threshold.  Points whose absolute difference from the
            fb-EWMA is **greater than** ``delta`` are replaced with ``NaN`` and
            later interpolated (if desired) by the caller.

        Returns
        -------
        ndarray
            NumPy array with the same length as *df_column* in which spike
            points have been replaced by ``np.nan``.  **No interpolation** is
            performed in this routine; leaving ``NaN`` values allows the caller
            to choose an appropriate gap-filling strategy.

        Notes
        -----
        * Forward EWMA:  ``fwd = df_column.ewm(span=span).mean()``
        * Backward EWMA: ``bwd = df_column[::-1].ewm(span=span).mean()[::-1]``
          The two series are averaged to form *fb-EWMA*.
        * A point *xᵢ* is considered a spike if
          ``|xᵢ – fb_ewmaᵢ| > delta``.
        * Source discussion:
          https://stackoverflow.com/questions/37556487/remove-spikes-from-signal-in-python

        Examples
        --------
        >>> import polars as pl
import pandas as pd  # optional compat, numpy as np
        >>> from ec import CalcFlux
        >>> ts = pd.Series([1, 1.2, 0.9, 5.5, 1.1, 1.0])  # 5.5 is a spike
        >>> calc = CalcFlux()
        >>> cleaned = calc.despike_ewma_fb(ts, span=3, delta=1.0)
        >>> cleaned
        array([1. , 1.2, 0.9, nan, 1.1, 1. ])
        """
        # Forward EWMA.
        fwd = pd.Series.ewm(df_column, span=span).mean()
        # Backward EWMA.
        bwd = pd.Series.ewm(df_column[::-1], span=span).mean()

        # Mean of forward and reversed backward EWMA.
        stacked_ewma = np.vstack((fwd, bwd[::-1]))
        np_fbewma = np.mean(stacked_ewma, axis=0)

        np_spikey = np.array(df_column)
        cond_delta = np.abs(np_spikey - np_fbewma) > delta
        np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)

        return np_remove_outliers

    def despike_med_mod(
        self,
        df_column: pd.Series,
        win: int = 800,
        fill_na: bool = True,
        addNoise: bool = False,
    ) -> pd.Series:
        """
        Detect and remove spikes using the **median‐filter RLM method**
        (Vickers & Mahrt, 1997-style).

        A *rolling-window median* (length *win*) provides a robust local
        baseline, after which a **robust linear model** (RLM, Huber loss with
        MAD scale) fits *y = β₀ + β₁·median*.  Residuals exceeding
        ``3 × scale`` are tagged as spikes.  Flagged points are replaced by:

        * ``NaN`` if ``fill_na=False``
        * Linearly interpolated values (optionally perturbed by Gaussian noise
          scaled to the RLM *scale* parameter) if ``fill_na=True``.

        Parameters
        ----------
        df_column : pandas.Series
            One-dimensional, **datetime‐indexed** series sampled at constant
            frequency (e.g. 20 Hz).  Must be numeric.
        win : int, default ``800``
            Length of the moving window used for the median filter.  At 20 Hz,
            800 points ≈ 40 s.
        fill_na : bool, default ``True``
            If *True*, gaps created by despiking are filled by linear
            interpolation.  If *False*, the spikes remain as ``NaN`` in the
            output.
        addNoise : bool, default ``False``
            When *True* and ``fill_na`` is enabled, Gaussian noise
            ``N(0, scale)`` (scale from the RLM fit) is added to the
            interpolated values to mimic natural variability.

        Returns
        -------
        pandas.Series
            Despiked series aligned to the input index.  Depending on
            *fill_na*, spikes are either gap-filled or kept as ``NaN``.

        Notes
        -----
        * **Model**:  ``y ~ median(win)`` fitted with `statsmodels.RLM`
          (Huber-T, scale estimate = MAD).
        * A point *i* is a spike if
          ``|resid_i| > 3 × scale``.  The factor “3” is hard-coded.
        * Window edges are back-/forward-filled to avoid edge effects.
        * With ``addNoise=True`` the injected noise keeps the spectral
          characteristics closer to the original series, useful for flux data
          where variance preservation matters.

        Examples
        --------
        >>> import polars as pl
import pandas as pd  # optional compat, numpy as np
        >>> from ec import CalcFlux
        >>> idx = pd.date_range("2025-06-21 13:30", periods=4000, freq="50L")
        >>> s = pd.Series(np.sin(np.linspace(0, 20, 4000)), index=idx)
        >>> s.iloc[1234] = 10      # inject spike
        >>> calc = CalcFlux()
        >>> clean = calc.despike_med_mod(s, win=400, fill_na=True, addNoise=False)
        >>> float(clean.iloc[1234])
        0.86  # interpolated, no more spike
        """

        np_spikey = np.array(df_column)

        y = df_column.interpolate().bfill().ffill()
        x = _rolling_median_centered(df_column, win)

        X = sm.add_constant(x)
        mod_rlm = sm.RLM(y, X)
        mod_fit = mod_rlm.fit(maxiter=300, scale_est="mad")

        cond_delta = np.abs(mod_fit.resid) > 3 * mod_fit.scale
        np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
        nanind = np.array(np.where(np.isnan(np_remove_outliers)))[0]

        data_out = pd.Series(np_remove_outliers, index=df_column.index)

        if fill_na:
            data_out = data_out.interpolate()
            data_outnaind = data_out.index[nanind]
            if addNoise:
                rando = np.random.normal(scale=mod_fit.scale, size=len(data_outnaind))
            else:
                rando = 0.0
            data_out.loc[data_outnaind] = data_out.loc[data_outnaind] + rando

        return data_out

    def despike_quart_filter(
        self,
        df_column: pd.Series,
        win: int = 600,
        fill_na: bool = True,
        top_quant: float = 0.97,
        bot_quant: float = 0.03,
        thresh: float | pd.Series | None = None,
    ) -> pd.Series:
        """
        Despike a time-series using a **rolling inter-quantile range (IQR)
        filter**.

        Within each sliding window of length *win* the algorithm computes

        * ``upper``  – the *top_quant* quantile
        * ``lower``  – the *bot_quant* quantile
        * ``med``    – the median

        A point *xᵢ* is flagged as a spike if

        ``|xᵢ – medᵢ| > thresh``

        where *thresh* defaults to the window-wise IQR
        ``(upper – lower)`` but may be overridden by a scalar or custom
        per-sample Series supplied via *thresh*.

        Parameters
        ----------
        df_column : pandas.Series
            **Datetime-indexed** series containing potential spikes.
        win : int, default ``600``
            Window length in samples.  For 20 Hz data, 600 ≈ 30 s.
        fill_na : bool, default ``True``
            When *True*, spike positions (set to ``NaN``) are filled by linear
            interpolation.  When *False*, they remain ``NaN`` in the output.
        top_quant : float, default ``0.97``
            Upper quantile used to form the IQR.  Common values lie between
            0.9 and 0.99 depending on desired aggressiveness.
        bot_quant : float, default ``0.03``
            Lower quantile used to form the IQR.  Ideally
            ``bot_quant = 1 – top_quant`` for symmetry.
        thresh : float or pandas.Series or None, optional
            Spike threshold.  If *None*, the rolling IQR is used.  Supply a
            scalar to use a fixed threshold or a Series (aligned to
            *df_column*) for a custom, time-varying criterion.

        Returns
        -------
        pandas.Series
            Despiked series on the original index.

        Notes
        -----
        * **Edge handling** – quantile, median, and IQR windows are centred;
          edge gaps are filled with linear interpolation followed by
          back-/forward-filling to provide values at the series ends.
        * **Performance** – rolling quantiles are relatively expensive for
          very long, high-frequency records.  Consider down-sampling or using
          larger *win* if speed becomes limiting.
        * If you need variance preservation, set ``fill_na=False`` and apply
          your own gap-filling that injects stochastic noise.

        Examples
        --------
        >>> import polars as pl
import pandas as pd  # optional compat, numpy as np
        >>> from ec import CalcFlux
        >>> idx = pd.date_range("2025-06-21 13:30", periods=1200, freq="50L")
        >>> s = pd.Series(np.sin(np.linspace(0, 6*np.pi, 1200)), index=idx)
        >>> s.iloc[400] = 5.0   # artificial spike
        >>> calc = CalcFlux()
        >>> clean = calc.despike_quart_filter(s, win=300, top_quant=0.95,
        ...                                   bot_quant=0.05, fill_na=True)
        >>> float(clean.iloc[400])
        0.0  # interpolated value (approx.)
        """
        np_spikey = np.array(df_column)

        # Rolling window statistics
        upper = (
            pd.Series(df_column)
            .rolling(window=win, center=True)
            .quantile(top_quant)
            .interpolate()
            .bfill()
            .ffill()
        )
        lower = (
            pd.Series(df_column)
            .rolling(window=win, center=True)
            .quantile(bot_quant)
            .interpolate()
            .bfill()
            .ffill()
        )
        med = (
            pd.Series(df_column)
            .rolling(window=win, center=True)
            .median()
            .interpolate()
            .bfill()
            .ffill()
        )
        iqr = upper - lower

        # Threshold: supplied or default to IQR
        if thresh is None:
            thresh = iqr

        # Flag spikes
        cond_delta = np.abs(np_spikey - med) > thresh
        np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
        nanind = np.array(np.where(np.isnan(np_remove_outliers)))[0]

        # Gap-fill or leave NaNs
        if fill_na:
            data_out = pd.Series(
                np_remove_outliers, index=df_column.index
            ).interpolate()
        else:
            data_out = pd.Series(np_remove_outliers, index=df_column.index)

        return data_out

    def get_lag(self, x: np.ndarray, y: np.ndarray) -> int:
        """
        Compute the sample **lag** (in index units) that maximises the discrete
        cross-correlation between two equally spaced signals.

        The routine wraps :func:`scipy.signal.correlate` and
        :func:`scipy.signal.correlation_lags` with ``mode="full"`` so that both
        positive and negative lags are considered.

        Parameters
        ----------
        x : ndarray
            First input signal (*reference*).  Must be one-dimensional and of
            the same sampling interval as *y*.
        y : ndarray
            Second input signal (*target*) with the same length as *x* (or
            longer/shorter; unequal lengths are supported by ``mode="full"``).

        Returns
        -------
        int
            Lag *ℓ* (in samples) at which the cross-correlation
            :math:`(x ⋆ y)[ℓ]` attains its maximum.

            * Positive lag ⇒ *y* **lags** *x* (y occurs later).
            * Negative lag ⇒ *y* **leads** *x* (y occurs earlier).

        Notes
        -----
        *Correlation definition* – for zero-mean sequences of equal length *N*:

        .. math::

            (x ⋆ y)[ℓ] = \\sum_{n=0}^{N-1} x[n]\,y[n-ℓ]

        where ℓ ∈ [−(N−1), N−1].  The implementation relies on SciPy’s FFT-
        based convolution, which is efficient for long sequences.
        Reference: SciPy docs for
        :func:`scipy.signal.correlation_lags` (accessed 2025-06-21).

        Examples
        --------
        >>> import numpy as np
        >>> from ec import CalcFlux
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=1000)
        >>> y = np.roll(x, 12)  # y lags x by 12 samples
        >>> calc = CalcFlux()
        >>> calc.get_lag(x, y)
        12
        """
        correlation = signal.correlate(x, y, mode="full")
        lags = signal.correlation_lags(x.size, y.size, mode="full")
        lag = lags[np.argmax(correlation)]
        return lag

    def calc_Td_dewpoint(self, E: float | np.ndarray) -> float | np.ndarray:
        """
        Convert water-vapour *pressure* to **dew-point temperature** using the
        ITS-90 polynomial formulation of Hardy (1998).

        The relationship

        .. math::

            T_d \;[\\text{K}] = \\frac{c_0 + c_1\ln E + c_2(\ln E)^2 + c_3(\ln E)^3}
                                     {d_0 + d_1\ln E + d_2(\ln E)^2 + d_3(\ln E)^3}

        is valid for :math:`E \\in \\lbrack 6.112\\times10^{-5},\\;1.227\\times10^{5}\\rbrack`
        Pa, i.e. dew-points from −100 °C to +100 °C, with an accuracy better than
        0.3 mK relative to the modified-Wexler equation.

        Parameters
        ----------
        E : float or array_like
            Water-vapour *pressure* (not mixing ratio) in **pascals**.

        Returns
        -------
        float or ndarray
            Dew-point temperature in **kelvin**.  The output preserves the
            input type (scalar ↦ scalar, array ↦ ndarray).

        References
        ----------
        Hardy, B. (1998). *ITS-90 formulations for vapor pressure, frost-point
        temperature, dew-point temperature, and enhancement factors in the
        range –100 °C to +100 °C*.  Proc. Intl. Symp. Humidity & Moisture,
        National Physical Laboratory, Teddington, UK.
        (see Eq. 4 and Table 2).

        Notes
        -----
        * Coefficients *c₀–c₃* and *d₀–d₃* are hard-coded directly from Hardy’s
          publication (ITS-90, Table 2, Dew-point scale).
        * **Units** – Input *E* must be in Pa; output is K.  Convert to/from
          °C as required:  *T[°C] = T[K] – 273.15*.
        * Agreement with the modified-Wexler formulation is within 0.3 mK over
          the full range −100 °C…+100 °C.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> # vapour pressure corresponding to Td = 10 °C
        >>> E_pa = 1228.46  # Pa
        >>> Td_K = calc.calc_Td_dewpoint(E_pa)
        >>> round(Td_K - 273.15, 2)   # convert to °C
        10.00
        """
        c0 = 207.98233
        c1 = -20.156028
        c2 = 0.46778925
        c3 = -0.0000092288067

        d0 = 1.0
        d1 = -0.13319669
        d2 = 0.0056577518
        d3 = -0.000075172865

        lne = np.log(E)
        nom = c0 + c1 * lne + c2 * lne**2 + c3 * lne**3
        denom = d0 + d1 * lne + d2 * lne**2 + d3 * lne**3
        return nom / denom

    def calc_Tf_frostpoint(self, E: float | np.ndarray) -> float | np.ndarray:
        """
        Convert water-vapour *pressure* to **frost-point temperature** using the
        ITS-90 polynomial formulation of Hardy (1998).

        The relationship

        .. math::

            T_f \;[\\text{K}] = \\frac{c_0 + c_1\ln E + c_2(\ln E)^2}
                                     {d_0 + d_1\ln E + d_2(\ln E)^2 + d_3(\ln E)^3}

        is valid for :math:`E \\in \\lbrack 2.273\\times10^{-6},\\;6.112\\rbrack`
        kPa (≈ frost-points from −150 °C to +0.1 °C) with agreement better than
        **0.1 mK** relative to the modified-Wexler equation.

        Parameters
        ----------
        E : float or array_like
            Water-vapour *pressure* (Pa).

        Returns
        -------
        float or ndarray
            Frost-point temperature (K), matching the input type
            (scalar → scalar, array → ndarray).

        References
        ----------
        Hardy, B. (1998). *ITS-90 formulations for vapor pressure, frost-point
        temperature, dew-point temperature and enhancement factors in the range
        −100 °C to +100 °C*. Proc. Int. Symp. Humidity & Moisture, NPL.
        (Eq. 5, Table 3).

        Notes
        -----
        * Coefficients *c₀–c₃* and *d₀–d₃* are taken directly from Hardy’s ITS-90
          table for frost-point temperature.
        * Output is **kelvin**; convert to °C with *T[°C] = T[K] – 273.15*.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> # vapour pressure giving Tf = −40 °C
        >>> E_pa = 12.84  # Pa
        >>> Tf_K = calc.calc_Tf_frostpoint(E_pa)
        >>> round(Tf_K - 273.15, 1)
        -40.0
        """
        c0 = 207.98233
        c1 = -20.156028
        c2 = 0.46778925
        c3 = -0.0000092288067

        d0 = 1.0
        d1 = -0.13319669
        d2 = 0.0056577518
        d3 = -0.000075172865

        lne = np.log(E)
        nom = c0 + c1 * lne + c2 * lne**2  # note: no c3 ln³ term
        denom = d0 + d1 * lne + d2 * lne**2 + d3 * lne**3
        return nom / denom

    def calc_E(
        self, pV: float | np.ndarray, T: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute **actual vapour pressure** from water-vapour *density* and
        absolute temperature via the ideal-gas law.

        The relationship is

        .. math::

            E = \\rho_v \\, R_v \\, T ,

        where
        :math:`\\rho_v` = *pV* (kg m⁻³),
        :math:`R_v` = 461.51 J kg⁻¹ K⁻¹ (gas constant for H₂O vapour),
        :math:`T` = *T* (K).

        Parameters
        ----------
        pV : float or array_like
            Water-vapour density (kg m⁻³).  Accepts a scalar or any NumPy-
            compatible array; the operation is vectorised.
        T : float or array_like
            Sonic (virtual) temperature or absolute air temperature (K),
            same shape as *pV*.

        Returns
        -------
        float or ndarray
            Vapour pressure **E** in pascals (Pa), matching the input type
            (scalar → scalar, array → ndarray).

        Notes
        -----
        * ``self.Rv`` stores the constant *R_v* (default ≈ 461.51 J kg⁻¹ K⁻¹).
        * No unit checks are performed; ensure *pV* is in kg m⁻³ and *T* in K.
        * The function is readily JIT-compilable with `@njit` for large arrays.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> calc.calc_E(0.0034, 290.2)     # rho_v in kg m⁻³, T in K
        455362.6868
        """
        e = pV * T * self.Rv
        return e

    def calc_Q(
        self,
        P: float | np.ndarray,
        e: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Convert total air *pressure* and water‐vapour *partial pressure* to
        **specific humidity** using the Bolton (1980) formulation.

        Specific humidity *q* is defined as the ratio of water-vapour mass to
        the total moist-air mass:

        .. math::

            q = \\frac{\\gamma\,e}{P - (1 - \\gamma)\,e},

        where :math:`\\gamma = M_v / M_d \\approx 0.622`.

        Parameters
        ----------
        P : float or array_like
            Ambient **air pressure** (Pa).  Scalar or NumPy-compatible array.
        e : float or array_like
            **Actual vapour pressure** (Pa), same shape as *P*.

        Returns
        -------
        float or ndarray
            Specific humidity *q* (kg H₂O kg⁻¹ moist air), preserving the input
            type (scalar → scalar, array → ndarray).

        References
        ----------
        Bolton, D. (1980). *The computation of equivalent potential temperature*.
        **Monthly Weather Review**, 108, 1046-1053.

        Notes
        -----
        * :math:`\\gamma = 0.622` is the ratio of the molar masses of water
          vapour (18.016 g mol⁻¹) to dry air (28.96 g mol⁻¹).
        * The expression is exact for ideal gases under meteorological
          conditions.

        Examples
        --------
        Scalar input
        ^^^^^^^^^^^
        >>> from ec import CalcFlux
        >>> fluxcalc = CalcFlux()
        >>> fluxcalc.calc_Q(4003.6, 717.0)
        0.11948162313727738

        Vectorised input
        ^^^^^^^^^^^^^^^^
        >>> import numpy as np
        >>> fluxcalc.calc_Q(np.array([4003.6, 4002.1]),
        ...                 np.array([717.0, 710.0]))
        array([0.11948162, 0.11827882])
        """

        # molar mass of water vapor/ molar mass of dry air
        gamma = 0.622
        q = (gamma * e) / (P - 0.378 * e)
        return q

    def calc_tc_air_temp_sonic(self, Ts, pV, P):
        """
        Convert **sonic (virtual) temperature** to true **air temperature**
        following the Campbell Scientific *EasyFlux®* implementation
        (adapted from Wallace & Hobbs, 2006).

        Sonic anemometers measure the **virtual** temperature :math:`T_s`
        because the speed of sound depends on moisture content.
        This routine removes the humidity effect using water-vapour density
        (:math:`\\rho_v`) and ambient pressure (*P*):

        .. math::

            T_a \\;[\\text{K}] =
              \\frac{T_{c1} - \\sqrt{T_{c2}}}{T_{c3}}

        where

        .. math::

            \\begin{aligned}
            T_{c1} &= P_{atm}
                    + \\bigl(2R_v - 3.040446\,R_d\\bigr)\\,\\rho_v\,T_s,\\\\[4pt]
            T_{c2} &= P_{atm}^2
                    + \\bigl(1.040446\,R_d\\,\\rho_v\,T_s\\bigr)^2
                    + 1.696\,R_d\\,\\rho_v\,P_{atm}\,T_s,\\\\[4pt]
            T_{c3} &= 2\\,\\rho_v\\Bigl[(R_v-1.944223\,R_d)
                    + (R_v-R_d)(R_v-2.040446\,R_d)\\,
                      \\dfrac{\\rho_v T_s}{P_{atm}}\\Bigr],
            \\end{aligned}

        with

        * *Ts* – sonic temperature (K)
        * :math:`\\rho_v` = *pV* (kg m⁻³)
        * *P*  – air pressure (Pa)
        * :math:`R_d` = 287.05 J kg⁻¹ K⁻¹
        * :math:`R_v` = 461.51 J kg⁻¹ K⁻¹
        * :math:`P_{atm}` = *P* × 9.86923 × 10⁻⁶ (Pa → atm)

        Parameters
        ----------
        Ts : float or ndarray
            Sonic (virtual) temperature, **kelvin**.
        pV : float or ndarray
            Water-vapour **density** in **g m⁻³**.  The function multiplies
            by 1 000 to obtain kg m⁻³.
        P : float or ndarray
            Ambient air pressure, **pascal**.

        Returns
        -------
        float or ndarray
            True air temperature *Ta* in **kelvin**, matching the shape of the
            inputs.

        References
        ----------
        Wallace, J. M., & Hobbs, P. V. (2006). *Atmospheric Science:
        An Introductory Survey* (2nd ed.), Eq. 3.36.
        Campbell Scientific (2018). *EasyFlux® DL Eddy-Covariance Processing*,
        algorithm notes.

        Notes
        -----
        * The input arrays must be broadcastable to a common shape.
        * Internally, *pV* is converted from g m⁻³ → kg m⁻³ by the factor 1 000.
        * The quadratic form avoids catastrophic cancellation for moist,
          near-saturation conditions.

        Examples
        --------
        >>> import numpy as np
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> Ts  = np.array([300.0, 295.0])      # K
        >>> pV  = np.array([12.0, 10.0])        # g m-3
        >>> P   = np.array([90000.0, 91000.0])  # Pa
        >>> Ta = calc.calc_tc_air_temp_sonic(Ts, pV, P)
        >>> np.round(Ta - 273.15, 2)            # convert to °C for display
        array([26.95, 21.94])
        """
        pV = pV * 1000.0
        P_atm = 9.86923e-6 * P
        T_c1 = P_atm + (2 * self.Rv - 3.040446 * self.Rd) * pV * Ts
        T_c2 = (
            P_atm * P_atm
            + (1.040446 * self.Rd * pV * Ts) * (1.040446 * self.Rd * pV * Ts)
            + 1.696000 * self.Rd * pV * P_atm * Ts
        )
        T_c3 = (
            2
            * pV
            * (
                (self.Rv - 1.944223 * self.Rd)
                + (self.Rv - self.Rd) * (self.Rv - 2.040446 * self.Rd) * pV * Ts / P_atm
            )
        )

        return (T_c1 - np.sqrt(T_c2)) / T_c3

    def calc_Tsa(
        self, Ts: float | np.ndarray, q: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Convert **sonic (virtual) temperature** to **air temperature** using
        specific humidity, after Schotanus *et al.* (1983).

        Sonic anemometers measure a virtual temperature :math:`T_s` that is
        higher than the true air temperature because moist air reduces the
        speed of sound.  The dry-air (or “adjusted sonic”) temperature
        :math:`T_{sa}` is obtained via

        .. math::

            T_{sa} = \\frac{T_s}{1 + 0.51\,q},

        where *q* is **specific humidity** (kg kg⁻¹).

        Parameters
        ----------
        Ts : float or array_like
            Sonic (virtual) temperature, **kelvin**.
        q : float or array_like
            Specific humidity *q* (kg H₂O kg⁻¹ moist air), same shape as *Ts*.

        Returns
        -------
        float or ndarray
            Adjusted air temperature *Tsa* (K), preserving the input type
            (scalar → scalar, array → ndarray).

        References
        ----------
        Schotanus, P., Nieuwstadt, F. T. M., & de Bruin, H. A. R. (1983).
        *Temperature measurement with a sonic anemometer and its application
        to heat and moisture fluxes.* **Boundary-Layer Meteorology**, 26, 81–93.
        Kaimal, J. C., & Gaynor, J. E. (1991). **BLM**, 55, 109–114.
        van Dijk, A. (2002). **Agricultural and Forest Meteorology**, 113, 31–43.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> fluxcalc = CalcFlux()
        >>> fluxcalc.calc_Tsa(291.5, 0.008)      # q = 0.8 % ≈ 0.008 kg/kg
        290.30921592677346
        """
        Tsa = Ts / (1 + 0.51 * q)
        return Tsa

    def calc_L(
        self,
        Ust: float | np.ndarray,
        Tsa: float | np.ndarray,
        Uz_Ta: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Compute the **Monin–Obukhov length** *L*, the fundamental length-scale
        that characterises atmospheric surface-layer stability.

        The formulation follows Monin–Obukhov similarity theory:

        .. math::

            L = -\,\\frac{u_*^{\,3}\,T_{sa}}
                      {\\kappa\,g\,w'\\!T_{sa}'},

        where

        * :math:`u_*`  = *Ust*  – friction velocity (m s⁻¹)
        * :math:`T_{sa}` = *Tsa* – adjusted (virtual) air temperature (K)
        * :math:`w'\\!T_{sa}'` = *Uz_Ta* – kinematic virtual-temperature flux
          (K m s⁻¹)
        * :math:`\\kappa` = 0.41 – von Kármán constant
        * :math:`g` = 9.81 m s⁻² – gravitational acceleration

        Positive *L* denotes **stable** (stratified) conditions, negative *L*
        denotes **unstable** (convective) conditions, and |L| → ∞ represents
        neutral stratification.

        Parameters
        ----------
        Ust : float or array_like
            Friction velocity *u★* (m s⁻¹).
        Tsa : float or array_like
            Virtual (or sonic-adjusted) air temperature *Tₛₐ* (K).
        Uz_Ta : float or array_like
            Covariance :math:`w'\\!T_{sa}'` (K m s⁻¹) – i.e. the kinematic
            virtual-temperature flux.
            Must have the same shape as *Ust* and *Tsa* (broadcastable).

        Returns
        -------
        float or ndarray
            Monin–Obukhov length *L* (m), with sign convention as above.  The
            output preserves the input type (scalar → scalar, array → ndarray).

        Notes
        -----
        * ``self.g`` and ``self.von_karman`` provide *g* and *κ*; override them
          at instantiation if alternative constants are required.
        * Beware of division by zero when *Uz_Ta* ≈ 0 (near-neutral
          conditions); resulting *L* tends to ±∞.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> fluxcalc = CalcFlux()
        >>> fluxcalc.calc_L(1.2, 292.2, 1.5)
        -83.69120608637277
        """
        return (-1 * (Ust**3) * Tsa) / (self.g * self.von_karman * Uz_Ta)

    def calc_Tsa_sonic_temp(self, Ts, P, pV):
        """
        Compute the **average sonic (virtual) temperature** from raw sonic
        temperature, ambient pressure and water-vapour density following the
        closed-form algebra used in Campbell Scientific *EasyFlux DL*.

        The algorithm first derives vapour pressure *E* with
        :meth:`calc_E(pV, Ts)`

        .. math::  E = \\rho_v R_v T_s ,

        and then solves the quadratic relationship between virtual and true
        temperature (Wallace & Hobbs, 2006, Eq. 3.36) to yield

        .. math::

            T_{sa} = -0.01645278052 \\,\\frac{-500P - 189E +
                     \\sqrt{250000P^2 + 128220PE + 35721E^2}}
                    {\\rho_v R_v},

        where

        * :math:`T_s`  = *Ts* : raw sonic temperature (K)
        * :math:`P`    = *P*  : air pressure (Pa)
        * :math:`\\rho_v` = *pV* : water-vapour density (kg m⁻³)
        * :math:`R_v` = 461.51 J kg⁻¹ K⁻¹ (default in ``self.Rv``).

        Parameters
        ----------
        Ts : float or array_like
            Raw sonic (virtual) temperature, **kelvin**.
        P : float or array_like
            Ambient air pressure, **pascal**.  Same shape as *Ts*.
        pV : float or array_like
            Water-vapour density, **kg m⁻³**.  Same shape as *Ts*.

        Returns
        -------
        float or ndarray
            Sonic-adjusted air temperature *Tsa* in **kelvin**,
            preserving the input type (scalar → scalar, array → ndarray).

        References
        ----------
        Wallace, J. M., & Hobbs, P. V. (2006). *Atmospheric Science: An
        Introductory Survey* (2nd ed.). Academic Press.
        Campbell Scientific (2018). *EasyFlux DL Eddy-Covariance Processing*
        algorithm notes.

        Notes
        -----
        * The magic constant *–0.01645278052* incorporates unit conversions
          and coefficients from the derivation.
        * Input arrays must be broadcastable to a common shape.
        * For low humidity (*pV* → 0) the denominator tends to zero; use with
          caution under extremely dry conditions.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> Ts  = 300.15      # K
        >>> P   = 90000.0     # Pa
        >>> pV  = 0.012       # kg m-3
        >>> Ta = calc.calc_Tsa_sonic_temp(Ts, P, pV)
        >>> round(Ta - 273.15, 2)   # convert to °C
        26.93
        """
        E = self.calc_E(pV, Ts)
        return (
            -0.01645278052
            * (
                -500 * P
                - 189 * E
                + np.sqrt(250000 * P**2 + 128220 * E * P + 35721 * E**2)
            )
            / pV
            / self.Rv
        )

    def calc_AlphX(
        self, L: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        Evaluate the empirical **α** and **X** stability-correction factors used
        in Massman’s (2000, 2001) scalar attenuation model.

        The coefficients depend on the sign and magnitude of the stability
        parameter *ζ = z/L*, where *z* is the sensor height (*self.UHeight*)
        and *L* is the Monin–Obukhov length provided via *L*.

        Definition
        ----------
        .. math::

            \\text{If } \\; ζ \\le 0:\\quad
                \\alpha = 0.925,\\;   X = 0.085

            \\text{If } \\; ζ > 0:\\quad
                \\alpha = 1.0,\\;     X = 2 - \\dfrac{1.915}{1 + 0.5\,ζ}

        Parameters
        ----------
        L : float or array_like
            Monin–Obukhov length *L* (m).  Positive for *stable* stratification,
            negative for *unstable*.  Scalars or arrays accepted; arrays must
            broadcast against ``self.UHeight``.

        Returns
        -------
        (alph, X) : tuple
            Two elements with shapes matching *L*:

            * **alph** – empirical coefficient α (dimensionless)
            * **X**    – empirical exponent   X (dimensionless)

        Notes
        -----
        * The function uses the instance attribute ``self.UHeight`` (measurement
          height, m) set during initialisation.
        * When *L* is **negative or zero** (unstable/neutral), ``alph = 0.925``
          and ``X = 0.085`` as per Massman (2000).
        * For **stable** conditions (*L > 0*), ``alph = 1`` and *X* varies with
          stability according to the expression above.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux(UHeight=3.5)
        >>> # Unstable case (L < 0)
        >>> calc.calc_AlphX(-50.0)
        (0.925, 0.085)
        >>> # Stable case (L > 0)
        >>> calc.calc_AlphX(100.0)
        (1.0, 0.27947368421052627)
        """
        if (self.UHeight / L) <= 0:
            alph = 0.925
            X = 0.085
        else:
            alph = 1
            X = 2 - 1.915 / (1 + 0.5 * self.UHeight / L)
        return alph, X

    def tetens(
        self,
        t: float | np.ndarray,
        a: float = 0.611,
        b: float = 17.502,
        c: float = 240.97,
    ) -> float | np.ndarray:
        """
        Compute the **saturation vapour pressure** (*eₛ*) of water over a
        flat surface using the Magnus–Tetens approximation (Wallace & Hobbs,
        2006, Eq. 3-8).

        The empirical formula is

        .. math::

            e_s\\,[\\text{kPa}] = a \\; \\exp\\!\\left(
                \\frac{b\,T}{T + c}
            \\right),

        where *T* is temperature in **°C** and the default coefficients
        *(a, b, c) = (0.611 kPa, 17.502, 240.97 °C)* give errors ≲ 0.1 % for
        −20 °C ≤ *T* ≤ 50 °C.

        Parameters
        ----------
        t : float or array_like
            Air (or surface) temperature in **degrees Celsius**.
        a : float, optional
            Empirical constant *a* in **kPa**.  Defaults to 0.611 kPa.
        b : float, optional
            Empirical constant *b* (dimensionless).  Defaults to 17.502.
        c : float, optional
            Empirical constant *c* in **°C**.  Defaults to 240.97 °C.

        Returns
        -------
        float or ndarray
            Saturation vapour pressure *eₛ* in **kilopascals** (kPa), preserving
            the input type (scalar → scalar, array → ndarray).

        Notes
        -----
        * The Magnus–Tetens form is most accurate for temperatures between
          –40 °C and 50 °C; outside this range, consider alternate formulations
          (e.g. Buck, Goff–Gratch).
        * The function is fully vectorised and Numba-compilable (`@njit`) for
          large arrays.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> calc.tetens(20.0)          # scalar input, °C
        2.339300338662703
        >>> import numpy as np
        >>> T = np.array([0.0, 10.0, 30.0])  # °C
        >>> calc.tetens(T)
        array([0.611 , 1.228 , 4.2465])
        """
        return a * np.exp((b * t) / (t + c))

    # @numba.jit(forceobj=True)
    def calc_Es(self, T: float | np.ndarray) -> float | np.ndarray:
        """
        Saturation vapour pressure over **liquid water** calculated from
        temperature using Hardy’s (1998) ITS-90 polynomial adaptation of the
        modified-Wexler equation.

        The formulation—valid for −100 °C ≤ *T* ≤ +100 °C (173.15 K … 373.15 K)
        with an accuracy of ±0.05 ppm—is

        .. math::

            \\ln E_s = \\sum_{i=0}^{7} g_i \\, f_i(T)

        with  

        .. math::

            \\begin{aligned}
            f_0 &= T^{-2}, & f_1 &= T^{-1}, & f_2 &= 1,\\\\
            f_3 &= T,     & f_4 &= T^{2},  & f_5 &= T^{3},\\\\
            f_6 &= T^{4}, & f_7 &= \\ln T,
            \\end{aligned}

        and coefficients :math:`g_0 … g_7` hard-coded below.

        Parameters
        ----------
        T : float or array_like
            Water (or air) temperature in **kelvin**.  Scalars or arrays
            accepted; arrays are processed element-wise.

        Returns
        -------
        float or ndarray
            Saturation vapour pressure *Eₛ* in **pascals** (Pa),
            preserving the input type (scalar → scalar, array → ndarray).

        References
        ----------
        Hardy, B. (1998). *ITS-90 formulations for vapor pressure, frost-point
        temperature, dew-point temperature, and enhancement factors in the
        range −100 °C to +100 °C*.  Proc. 3rd International Symposium on
        Humidity & Moisture, NPL, Teddington, UK (Eq. 2).

        Notes
        -----
        * Within 0.05 ppm of the modified-Wexler formulation across the entire
          −100 °C … +100 °C range.  
        * Output units are **Pa**; divide by 1 000 for **kPa** if needed.  
        * The function is fully vectorised and Numba-ready (`@njit`) for speed.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> # saturation vapour pressure at 25 °C (298.15 K)
        >>> round(calc.calc_Es(298.15) / 1000, 3)   # convert to kPa
        3.169
        >>> import numpy as np
        >>> temps = np.array([273.15, 293.15, 313.15])  # 0, 20, 40 °C
        >>> calc.calc_Es(temps)                        # Pa
        array([   611.164 ,  23391.503 ,  73850.247 ])
        """
        g0 = -2836.5744
        g1 = -6028.076559
        g2 = 19.54263612
        g3 = -0.02737830188
        g4 = 0.000016261698
        g5 = 0.00000000070229056
        g6 = -0.00000000000018680009
        g7 = 2.7150305

        return np.exp(
            g0 * T ** (-2)
            + g1 * T ** (-1)
            + g2
            + g3 * T
            + g4 * T**2
            + g5 * T**3
            + g6 * T**4
            + g7 * np.log(T)
        )

    def calc_cov(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute the **covariance** between two equally sized 1-D arrays using a
        manually expanded formula suitable for straight-line (and optionally
        JIT-compiled) execution.

        The unbiased sample covariance is

        .. math::

            \\operatorname{cov}(p_1, p_2)
            = \\frac{ \\sum_i p_{1i} p_{2i}
                      - \\frac{1}{N}\\,\\bigl(\\sum_i p_{1i}\\bigr)
                                       \\bigl(\\sum_i p_{2i}\\bigr) }
                     {N - 1},

        where *N* = ``len(p1)``.  This explicit form avoids the intermediate
        demean step and lends itself to Numba optimisation with
        ``@njit(parallel=True)``.

        Parameters
        ----------
        p1 : ndarray
            First input vector (numeric, 1-D).
        p2 : ndarray
            Second input vector (same shape as *p1*).

        Returns
        -------
        float
            Unbiased sample covariance between *p1* and *p2*.

        Notes
        -----
        * When N ≳ 10 000, the JIT-compiled version is typically **an order of
          magnitude faster** than :func:`numpy.cov` or
          :pymeth:`pandas.Series.cov`.
        * No NaN handling is performed; ensure inputs are pre-cleaned.
        * If *p1* and *p2* are large column slices from a contiguous array,
          consider passing the raw NumPy views to avoid extra copies.

        Examples
        --------
        >>> import numpy as np
        >>> from ec import CalcFlux
        >>> rng = np.random.default_rng(42)
        >>> x = rng.standard_normal(10000)
        >>> y = 2.0 * x + rng.standard_normal(10000) * 0.1
        >>> calc = CalcFlux()
        >>> round(calc.calc_cov(x, y), 5)
        3.99923
        """
        sumproduct = np.sum(p1 * p2)
        return (sumproduct - (np.sum(p1) * np.sum(p2)) / len(p1)) / (len(p1) - 1)

    def calc_MSE(self, y: np.ndarray) -> float:
        """
        Compute the **mean square error** (MSE) of a one-dimensional array, i.e.
        the population variance without Bessel’s correction.

        The expression is

        .. math::

            \\operatorname{MSE}(y) = \\frac{1}{N}\\sum_{i=1}^{N}
                                      \\bigl(y_i - \\bar{y}\\bigr)^2,

        where :math:`\\bar{y}` is the arithmetic mean of *y* and *N* is the
        length of the array.

        Parameters
        ----------
        y : ndarray
            One-dimensional NumPy array (numeric).  No NaN handling is
            performed; remove or impute missing values beforehand.

        Returns
        -------
        float
            Mean square error of *y*.

        Notes
        -----
        * This is **not** the unbiased sample variance (which divides by
          *N – 1*); it is the population variance—also equal to the mean of the
          squared deviations.
        * The function is fully vectorised and can be JIT-compiled with Numba
          (`@njit`) for large arrays.

        Examples
        --------
        >>> import numpy as np
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> arr = np.array([1.0, 2.0, 3.0, 4.0])
        >>> calc.calc_MSE(arr)
        1.25
        """
        return np.mean((y - np.mean(y)) ** 2)

    def convert_KtoC(self, T: float | np.ndarray) -> float | np.ndarray:
        """
        Convert **temperature** from kelvin to degrees Celsius.

        Parameters
        ----------
        T : float or array_like
            Temperature(s) in **kelvin**.  Accepts a scalar or any NumPy-
            compatible array; the subtraction is fully vectorised.

        Returns
        -------
        float or ndarray
            Temperature in **°C**, preserving the input type
            (scalar → scalar, array → ndarray).

        Notes
        -----
        * Uses 273.16 rather than the rounded 273.15; this matches the offset
          used elsewhere in the module for ITS-90 consistency.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> calc.convert_KtoC(300.0)
        26.84
        >>> import numpy as np
        >>> calc.convert_KtoC(np.array([273.16, 293.16]))
        array([ 0., 20.])
        """
        return T - 273.16

    def convert_CtoK(self, T: float | np.ndarray) -> float | np.ndarray:
        """
        Convert **temperature** from degrees Celsius to kelvin.

        Parameters
        ----------
        T : float or array_like
            Temperature(s) in **°C**.  Accepts a scalar or any NumPy-compatible
            array; the addition is fully vectorised.

        Returns
        -------
        float or ndarray
            Temperature in **kelvin**, preserving the input type
            (scalar → scalar, array → ndarray).

        Notes
        -----
        * Uses an offset of **273.16 K** for ITS-90 consistency with
          other routines in this module.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> calc.convert_CtoK(20.0)
        293.16
        >>> import numpy as np
        >>> calc.convert_CtoK(np.array([0.0, 100.0]))
        array([273.16, 373.16])
        """
        return T + 273.16

    def correct_KH20(
        self,
        Uz_Ta: float | np.ndarray,
        P: float | np.ndarray,
        T: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Compute the **additive oxygen–water cross-sensitivity correction** for
        a KH-20 krypton hygrometer following Oren *et al.* (1998).

        Krypton hygrometers measure water-vapour density by absorption at
        2.0 µm; however, overlapping absorption by O₂ introduces a bias
        proportional to the sensible-heat covariance *w′T′*.
        The correction term is

        .. math::

            \\Delta\\text{KH}_{\\mathrm{O_2}} =
            \\frac{C_{\\mathrm{O_2}}\\,M_{\\mathrm{O_2}}\,P}
                 {R_u\,T^{2}}
            \\;\\frac{K_o}{K_w}\\;w'\\!T',

        where

        * *Uz_Ta*  = :math:`w′T′`  – covariance of vertical wind and air
          temperature (K m s⁻¹)
        * *P*      – air pressure (Pa)
        * *T*      – air temperature (K)
        * :math:`C_{\\mathrm{O_2}} = 0.21` – volumetric O₂ mixing ratio
        * :math:`M_{\\mathrm{O_2}} = 0.032\\,\\text{kg mol}^{-1}`
        * *Ru*     – universal gas constant (J mol⁻¹ K⁻¹)
        * *Kw*     – water extinction coefficient (instrument specific)
        * *Ko*     – oxygen extinction coefficient (empirical)

        All constants except *Kw* and *Ko* are defined at instantiation;
        *Kw* and *Ko* default to 1.0 and −0.0045 but should be set from
        calibration.

        Parameters
        ----------
        Uz_Ta : float or array_like
            Kinematic sensible-heat flux *w′T′* (K m s⁻¹).
        P : float or array_like
            Ambient air pressure (Pa).
        T : float or array_like
            Air temperature (K).

        Returns
        -------
        float or ndarray
            Additive KH-20 oxygen correction in the units of the raw KH-20
            signal (typically kg m⁻³), matching the shape of the inputs.

        Notes
        -----
        * The correction is usually **small** (≈ 1–3 %) but should be applied
          for high-precision latent-heat fluxes.
        * Sign depends on *Ko*; the default −0.0045 implies a negative bias
          that **reduces** the raw water-vapour covariance.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux(Kw=1.412, Ko=-0.0045)
        >>> Uz_Ta = -0.025     # K m s-1
        >>> P     = 90000.0    # Pa
        >>> T     = 293.15     # K
        >>> round(calc.correct_KH20(Uz_Ta, P, T), 6)
        -4.385e-05
        """
        return (
            ((self.Co * self.Mo * P) / (self.Ru * T**2)) * (self.Ko / self.Kw) * Uz_Ta
        )

    def correct_spectral(
        self,
        B: float | np.ndarray,
        alpha: float | np.ndarray,
        varib: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Compute the **frequency‐response (spectral‐attenuation) correction
        factor** for high-frequency scalar fluxes following Massman (2000,
        2001).

        The multiplicative correction term is

        .. math::

            C = \\frac{B^{\\alpha}}{B^{\\alpha}+1}
                \\;\\frac{B^{\\alpha}}{B^{\\alpha}+V^{\\alpha}}
                \\;\\frac{1}{V^{\\alpha}+1},

        where

        * *B*      – dimensionless sensor **response ratio**
                      :math:`B = f_c / f_s`
                      (cut-off frequency / sensor frequency response)
        * *α*      – empirical **attenuation exponent** (0 < α ≤ 1)
        * *V* (= *varib*) – ratio of **transport velocity** to mean wind
          speed, or more generally a dimensionless quantity representing the
          Reynolds number dependence of the scalar.

        Parameters
        ----------
        B : float or array_like
            Sensor response ratio (:math:`f_c / f_s`), > 0.
        alpha : float or array_like
            Spectral attenuation exponent α (dimensionless), typically between
            0.5 and 1.0.  Must be broadcast-compatible with *B*.
        varib : float or array_like
            Dimensionless parameter *V* associated with the transported scalar
            (often the ratio of molecular diffusivities or a stability‐derived
            velocity scale).  Same shape as *B*.

        Returns
        -------
        float or ndarray
            Spectral correction factor *C* (dimensionless), preserving the
            input type (scalar → scalar, array → ndarray).

        Notes
        -----
        * The correction multiplies the raw covariance to compensate for
          high-frequency loss due to sensor path length, separation, and
          analogue/digital filtering.
        * For **perfect response** (*B ≫ 1* and *V ≈ 1*) the factor approaches
          unity.  For sluggish instruments (*B ≪ 1*) the correction can exceed
          2–3 ×.
        * Ensure that *B*, *α*, and *varib* (V) are derived consistently with
          the spectral model adopted in Massman (2000, 2001).

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> calc.correct_spectral(B=0.3, alpha=0.75, varib=1.2)
        0.23249283581863403
        """
        B_alpha = B**alpha
        V_alpha = varib**alpha
        return (
            (B_alpha / (B_alpha + 1))
            * (B_alpha / (B_alpha + V_alpha))
            * (1 / (V_alpha + 1))
        )

    def get_Watts_to_H2O_conversion_factor(
        self,
        temperature: float | np.ndarray,
        day_fraction: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Return the multiplicative **conversion factor** that turns a latent-heat
        flux expressed in **watts per square metre** (W m⁻²) into **inches of
        liquid-water depth** evaporated (or transpired) over the sampling
        period.

        The factor is derived from

        .. math::

            \\text{Conv} = \\frac{\\rho_w(T)\\;\\Delta t}{\\lambda(T)}\\;
                           \\frac{1}{25.4},

        where

        * :math:`\\rho_w(T)`  – density of liquid water (kg m⁻³) at *T*,
          via :meth:`calc_water_density`.
        * :math:`\\lambda(T)` – latent heat of vaporisation (kJ kg⁻¹) at *T*,
          via :meth:`calc_latent_heat_of_vaporization`.
        * :math:`\\Delta t`   – time step (s) expressed as
          *day_fraction* × 86 400 s; dividing by 1 000 yields kilojoules.
        * 25.4               – millimetres per inch (to convert mm → inch).

        Hence a latent-heat flux *LE* (W m⁻²) multiplied by this factor gives
        inches H₂O (period⁻¹).

        Parameters
        ----------
        temperature : float or array_like
            Mean **air (or surface) temperature** during the period, **°C**.
        day_fraction : float or array_like
            Fraction of a 24-hour day represented by the averaging interval
            (e.g. half-hourly data ⇒ 0.5 h / 24 h = 0.020833).

        Returns
        -------
        float or ndarray
            Conversion factor (inch H₂O / (W m⁻² period)), preserving the input
            type (scalar → scalar, array → ndarray).

        Notes
        -----
        * Internally multiplies by **86.4 = 86 400 / 1 000** to convert
          J m⁻² s⁻¹ → kJ m⁻² day⁻¹.
        * Assumes water density and latent heat routines return
          **kg m⁻³** and **kJ kg⁻¹**, respectively.
        * If you prefer output in **millimetres** rather than inches, remove the
          division by 25.4.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> # 30-min averaging period at 20 °C
        >>> fac = calc.get_Watts_to_H2O_conversion_factor(temperature=20.0,
        ...                                               day_fraction=0.5/24)
        >>> round(fac, 6)
        0.000344
        >>> # Convert a latent-heat flux of 120 W m⁻² to inches H₂O per 30 min
        >>> LE = 120.0        # W m-2
        >>> ET_inches = LE * fac
        >>> round(ET_inches, 3)
        0.041
        """
        to_inches = 25.4
        return (self.calc_water_density(temperature) * 86.4 * day_fraction) / (
            self.calc_latent_heat_of_vaporization(temperature) * to_inches
        )

    def calc_water_density(self, temperature: float | np.ndarray) -> float | np.ndarray:
        """
        Calculate the **density of liquid water** (ρₗ) as a function of
        temperature using the empirical polynomial adopted by Campbell
        Scientific (*EasyFlux®*).

        The relation is an adaptation of the Kell (1975) equation:

        .. math::

            \\rho_w(T) = d_5\\Bigl[
                1 - \\frac{(T + d_1)^2 (T + d_2)}
                             {d_3\\,(T + d_4)}
            \\Bigr],

        where coefficients (d₁–d₅) are hard-coded below.

        Parameters
        ----------
        temperature : float or array_like
            Water (or air) temperature in **degrees Celsius**.

        Returns
        -------
        float or ndarray
            Density of liquid water, **kg m⁻³**, preserving the input type
            (scalar → scalar, array → ndarray).

        Coefficients
        ------------
        ==========  =========  =======================
        Symbol      Value      Units
        ----------  ---------  -----------------------
        d₁          −3.983035  °C
        d₂           301.797   °C
        d₃         522 528.9   °C²
        d₄            69.34881 °C
        d₅           999.97495 kg m⁻³
        ==========  =========  =======================

        Notes
        -----
        * Valid for 0 °C ≤ *T* ≤ 40 °C with errors ≲ 0.01 kg m⁻³.
        * For temperatures outside this interval, consider IAPWS-95 or Kell’s
          original formulation.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> calc.calc_water_density(20.0)
        998.2075749561653
        >>> import numpy as np
        >>> T = np.array([0.0, 4.0, 40.0])  # °C
        >>> calc.calc_water_density(T)
        array([999.84293048, 999.9999972 , 992.21100736])
        """
        d1 = -3.983035  # °C
        d2 = 301.797  # °C
        d3 = 522528.9  # °C²
        d4 = 69.34881  # °C
        d5 = 999.97495  # kg m⁻³
        return d5 * (
            1 - (temperature + d1) ** 2 * (temperature + d2) / (d3 * (temperature + d4))
        )

    def lamb_func(self, x: float | np.ndarray, varb: str) -> float | np.ndarray:
        """
        Polynomial approximation for the latent heat of phase change of
        **water** (*liquid → vapour*) or **ice** (*solid → vapour*)
        as a function of temperature.

        The function evaluates

        .. math::

            \\lambda(T) \\;[\\text{J kg}^{-1}] =
            a_0 + a_1 T + a_2 T^2 + a_3 T^3,

        where *T* = *x* (°C) and the coefficients *(a₀…a₃)* depend on the
        selected phase:

        ====================  ==========  ==========  ==========  ==========
        Phase (*varb*)        a₀          a₁          a₂          a₃
        --------------------  ----------  ----------  ----------  ----------
        ``"water"`` (λ_v)     2 500 800   −2 360       1.6        −0.06
        ``"ice"``   (λ_s)     2 834 100   −290        −4.0         0.0
        ====================  ==========  ==========  ==========  ==========

        Parameters
        ----------
        x : float or array_like
            Temperature in **degrees Celsius** at which λ is evaluated.
        varb : {'water', 'ice'}
            Phase identifier:
            * ``"water"`` – latent heat of **vaporisation** (liquid → vapour)
            * ``"ice"``   – latent heat of **sublimation**  (solid  → vapour)

        Returns
        -------
        float or ndarray
            Latent heat λ (*J kg⁻¹*) for the requested phase, preserving the
            input type (scalar → scalar, array → ndarray).

        Notes
        -----
        * Coefficients originate from standard meteorological tables and are
          the same polynomial used in many eddy-covariance processing
          frameworks (e.g. EasyFlux).
        * Valid for −40 °C ≤ *T* ≤ 40 °C (water) and −40 °C ≤ *T* ≤ 0 °C (ice)
          with typical errors ≲ 0.1 %.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> # Latent heat of vaporisation at 20 °C
        >>> round(calc.lamb_func(20.0, "water") / 1e6, 3)
        2.454
        >>> # Vectorised evaluation for ice phase
        >>> import numpy as np
        >>> T = np.array([-20.0, -10.0, -5.0])  # °C
        >>> calc.lamb_func(T, "ice")
        array([2834100., 2805100., 2790600.])
        """
        varib = dict(water=[2500800, -2360, 1.6, -0.06], ice=[2834100, -290, -4, 0])
        return (
            varib[varb][0]
            + varib[varb][1] * x
            + varib[varb][2] * x**2
            + varib[varb][3] * x**3
        )

    def calc_latent_heat_of_vaporization(
        self,
        temperature: float | np.ndarray,
        units: str = "C",
    ) -> float | np.ndarray:
        """
        Compute the temperature-dependent **latent heat of vaporisation /
        sublimation** (λ) of water.

        The routine delegates to :meth:`lamb_func`, selecting the polynomial
        for **liquid water** when *T* ≥ 0 °C and **ice** when *T* < 0 °C.
        Input may be supplied in °C or K.

        Parameters
        ----------
        temperature : float or array_like
            Temperature at which λ is required.
        units : {'C', 'K'}, default ``'C'``
            Units of *temperature*.
            * ``'C'`` – degrees Celsius (default)
            * ``'K'`` – kelvin (internally converted to °C)
            * Any other string raises a ``ValueError``.  (The tongue-in-cheek
              comment “F is for losers” is left intact but not enforced.)

        Returns
        -------
        float or ndarray
            Latent heat λ in **joules per kilogram (J kg⁻¹)**, matching the
            input type (scalar → scalar, array → ndarray).

        References
        ----------
        Rogers, R. R., & Yau, M. K. (1989). *A Short Course in Cloud Physics*
        (3rd ed.), Eq. 2.26.
        See also the Wikipedia article on *Latent heat*.

        Notes
        -----
        * For *T* ≥ 0 °C the latent heat of **condensation/vaporisation** is
          returned (water phase).
        * For *T* < 0 °C the latent heat of **sublimation** is returned (ice
          phase).
        * The underlying coefficients are identical to those in
          :meth:`lamb_func`.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> # 20 °C (liquid phase)
        >>> round(calc.calc_latent_heat_of_vaporization(20.0) / 1e6, 3)
        2.454
        >>> # −10 °C (ice phase)
        >>> calc.calc_latent_heat_of_vaporization(-10.0)
        2805100.0
        >>> # Kelvin input
        >>> calc.calc_latent_heat_of_vaporization(293.15, units='K')
        2454000.0
        """
        if units == "K":
            temperature = self.convert_KtoC(temperature)
        else:
            pass

        return np.where(
            temperature >= 0,
            self.lamb_func(temperature, "water"),
            self.lamb_func(temperature, "ice"),
        )  # J kg⁻¹

    def shadow_correction(self, Ux, Uy, Uz):
        """
        Correct the three wind-velocity components for **flow distortion**
        (“shadowing”) caused by the **CSAT3/CSAT3A sonic anemometer** support
        struts, after Horst, Wilczak & Cook (2015).

        The routine iteratively (4 passes)

        1. Rotates (*u, v, w*) into the **transducer-path coordinate system**,
        2. Applies angle-dependent *shadow factors* to each path velocity, and
        3. Transforms the adjusted velocities back to the sonic coordinate
           system.

        Parameters
        ----------
        Ux : float or ndarray
            Longitudinal wind component *u* (m s⁻¹).
        Uy : float or ndarray
            Lateral wind component *v* (m s⁻¹).
        Uz : float or ndarray
            Vertical wind component *w* (m s⁻¹).

            All three inputs must be broadcast-compatible and are updated
            **in-place** inside the loop; the final corrected values are also
            returned.

        Returns
        -------
        tuple
            **(Uxc, Uyc, Uzc)** – corrected wind components (same units and
            shape as the inputs).

        Notes
        -----
        * The correction accounts for ~16 % amplitude attenuation when flow
          originates perpendicular to a transducer path and diminishes to
          zero when flow is parallel (sin θ term).
        * Convergence is rapid; four iterations are sufficient for <0.1 %
          residual error.
        * Original coefficients from Horst et al. (2015, *Atmos. Meas. Tech.*)
          Table 2; rotation matrices follow Kaimal & Finnigan (1994).

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> u, v, w = 2.5, 0.4, 0.1  # m s-1
        >>> uc, vc, wc = calc.shadow_correction(u, v, w)
        >>> round(uc, 3), round(vc, 3), round(wc, 3)
        (2.498, 0.393, 0.108)
        """
        # Rotation matrix: instrument → transducer-path coordinates
        h = [
            0.25,
            0.4330127018922193,
            0.8660254037844386,
            -0.5,
            0.0,
            0.8660254037844386,
            0.25,
            -0.4330127018922193,
            0.8660254037844386,
        ]

        # Inverse rotation matrix: path → instrument coordinates
        hinv = [
            0.6666666666666666,
            -1.3333333333333333,
            0.6666666666666666,
            1.1547005383792517,
            0.0,
            -1.1547005383792517,
            0.38490017945975047,
            0.38490017945975047,
            0.38490017945975047,
        ]

        iteration = 0
        while iteration < 4:
            Uxh = h[0] * Ux + h[1] * Uy + h[2] * Uz
            Uyh = h[3] * Ux + h[4] * Uy + h[5] * Uz
            Uzh = h[6] * Ux + h[7] * Uy + h[8] * Uz

            scalar = np.sqrt(Ux**2.0 + Uy**2.0 + Uz**2.0)

            Theta1 = np.arccos(np.abs(h[0] * Ux + h[1] * Uy + h[2] * Uz) / scalar)
            Theta2 = np.arccos(np.abs(h[3] * Ux + h[4] * Uy + h[5] * Uz) / scalar)
            Theta3 = np.arccos(np.abs(h[6] * Ux + h[7] * Uy + h[8] * Uz) / scalar)

            # Angle-dependent shadow factors (Horst et al., 2015)
            Uxa = Uxh / (0.84 + 0.16 * np.sin(Theta1))
            Uya = Uyh / (0.84 + 0.16 * np.sin(Theta2))
            Uza = Uzh / (0.84 + 0.16 * np.sin(Theta3))

            # Transform back to sonic (instrument) frame
            Uxc = hinv[0] * Uxa + hinv[1] * Uya + hinv[2] * Uza
            Uyc = hinv[3] * Uxa + hinv[4] * Uya + hinv[5] * Uza
            Uzc = hinv[6] * Uxa + hinv[7] * Uya + hinv[8] * Uza

            Ux, Uy, Uz = Uxc, Uyc, Uzc
            iteration += 1

        return Uxc, Uyc, Uzc

    def calculated_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive secondary **thermodynamic and micrometeorological quantities**
        needed by subsequent flux-processing steps and append them to the
        working DataFrame.

        The routine expects the raw measurements

        * ``Ea`` – vapour pressure (kPa)
        * ``Ts`` – sonic (virtual) temperature (K)
        * ``Pr`` – ambient pressure (Pa)

        and computes:

        ==========  ===============================================================  Units
        ----------  ---------------------------------------------------------------  -------
        ``pV``      Water-vapour density (ρᵥ) via :meth:`calc_pV`                   kg m⁻³
        ``Tsa``     Sonic-adjusted air temperature via :meth:`calc_Tsa`             K
        ``E``       Actual vapour pressure from ρᵥ and *Tsa* via :meth:`calc_E`      Pa
        ``Q``       Specific humidity via :meth:`calc_Q`                            kg kg⁻¹
        ``Sd``      Saturation deficit = *Q_s* − *Q*                                kg kg⁻¹
        ==========  ===============================================================  =======

        where *Q_s* is the saturation specific humidity computed from
        :meth:`calc_Es(Tsa)` evaluated at *Tsa*.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing at minimum the columns
            ``Ea``, ``Ts`` (K), and ``Pr`` (Pa).  Additional columns are left
            untouched.

        Returns
        -------
        pandas.DataFrame
            Same object with **five new columns** – ``pV``, ``Tsa``, ``E``,
            ``Q``, and ``Sd`` – added in place.

        Notes
        -----
        * The function modifies *df* **in-place** and also returns it for
          convenience / method chaining.
        * All helper methods invoked here are vectorised; performance is close
          to native NumPy speed for large datasets.

        Examples
        --------
        >>> import polars as pl
import pandas as pd  # optional compat
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> df = pd.DataFrame({
        ...     "Ea": [1.2, 1.1],        # kPa
        ...     "Ts": [293.15, 294.15],  # K
        ...     "Pr": [90000, 89950]     # Pa
        ... })
        >>> calc.calculated_parameters(df).columns[-5:]
        Index(['pV', 'Tsa', 'E', 'Q', 'Sd'], dtype='object')
        """
        df["pV"] = self.calc_pV(df["Ea"], df["Ts"])
        df["Tsa"] = self.calc_Tsa(df["Ts"], df["Pr"], df["pV"])
        df["E"] = self.calc_E(df["pV"], df["Tsa"])
        df["Q"] = self.calc_Q(df["Pr"], df["E"])
        df["Sd"] = self.calc_Q(df["Pr"], self.calc_Es(df["Tsa"])) - df["Q"]
        return df

    def calculated_parameters_irga(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive secondary **thermodynamic variables** required for the
        *IRGA/sonic* processing pathway and append them to the working
        DataFrame.

        Expected **input columns**

        ============  ===========================================  Units
        ------------  -------------------------------------------  -------
        ``Ta``        Reference (HMP) air temperature              °C ¹
        ``Ts_ro``     Raw sonic (virtual) temperature              °C
        ``Pr_ro``     Raw ambient pressure                         kPa
        ``pV_ro``     Raw water-vapour density (ρᵥ)                g m⁻³
        ============  ===========================================  =======

        ¹ *Ta* is not used inside this routine but must be present for
        consistency with the KH-20 pathway; you may drop it afterwards.

        **Computed columns**

        ==========  ===========================================================  Units
        ----------  -----------------------------------------------------------  -------
        ``Ts_K``    Sonic temperature converted to kelvin                       K
        ``E``       Actual vapour pressure from ρᵥ and Ts via :meth:`calc_E`     Pa
        ``Q``       Specific humidity via :meth:`calc_Q`                         kg kg⁻¹
        ``Tsa``     Sonic-adjusted air temperature via :meth:`calc_Tsa_air_temp_sonic`  K
        ``Es``      Saturation vapour pressure at *Tsa* via :meth:`calc_Es`      Pa
        ``Sd``      Saturation deficit (*Q_s − Q*)                               kg kg⁻¹
        ==========  ===========================================================  =======

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame containing at least the columns listed above.  Units
            **must** match the table.

        Returns
        -------
        pandas.DataFrame
            Same object with the seven new/updated columns added **in-place**.

        Notes
        -----
        * ``pV_ro`` is converted from **g m⁻³ → kg m⁻³** inside the call to
          :meth:`calc_E` by multiplying by 0.001.
        * ``Pr_ro`` is converted from **kPa → Pa** where required
          (*× 1 000*).
        * The routine is intended for data from integrated open-path IRGAs
          (e.g. IRGASON) where water-vapour density is provided directly.

        Examples
        --------
        >>> import polars as pl
import pandas as pd  # optional compat
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> df = pd.DataFrame({
        ...     "Ta": [20.1],            # °C
        ...     "Ts_ro": [20.3],         # °C
        ...     "Pr_ro": [90.0],         # kPa
        ...     "pV_ro": [10.8]          # g m-3
        ... })
        >>> calc.calculated_parameters_irga(df)[
        ...     ["Ts_K", "E", "Q", "Tsa", "Es", "Sd"]
        ... ].round(3)
             Ts_K         E      Q      Tsa        Es     Sd
        0  293.46  457.092  0.009  292.279  23391.503  0.003
        """
        # convert pV to g m⁻³ → kg m⁻³
        df["Ts_K"] = self.convert_CtoK(df["Ts_ro"])
        df["E"] = self.calc_E(df["pV_ro"] * 0.001, df["Ts_K"])

        # convert air pressure from kPa → Pa
        df["Q"] = self.calc_Q(df["Pr_ro"] * 1000.0, df["E"])

        # sonic-adjusted air temperature
        df["Tsa"] = self.calc_Tsa_air_temp_sonic(df["Ts_K"], df["Q"])

        # saturation vapour pressure and saturation deficit
        df["Es"] = self.calc_Es(df["Tsa"])
        df["Sd"] = self.calc_Q(df["Pr_ro"], self.calc_Es(df["Tsa"])) - df["Q"]
        return df

    def calc_pV(
        self, Ea: float | np.ndarray, Ts: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Convert **actual vapour pressure** (*Eₐ*) to **water-vapour density**
        (ρᵥ) using the ideal-gas law.

        The relationship is

        .. math::

            \\rho_v = \\frac{E_a}{R_v \\, T_s},

        but because *Ea* is supplied here in **kilopascals** (kPa) rather than
        pascals (Pa), the function first multiplies by 1 000:

        .. math::

            \\rho_v\\,[\\text{kg m}^{-3}] =
            \\frac{E_a\\,[\\text{kPa}] \\times 1000}
                 {R_v \\, T_s},

        where

        * *Ea* – actual vapour pressure (kPa)
        * *Ts* – sonic (virtual) temperature (K)
        * :math:`R_v` – 461.51 J kg⁻¹ K⁻¹ (stored in ``self.Rv``)

        Parameters
        ----------
        Ea : float or array_like
            Actual vapour pressure, **kilopascals** (kPa).
        Ts : float or array_like
            Sonic or virtual temperature, **kelvin** (K).  Must broadcast with
            *Ea*.

        Returns
        -------
        float or ndarray
            Water-vapour density ρᵥ in **kg m⁻³**, preserving the input type
            (scalar → scalar, array → ndarray).

        Notes
        -----
        * If *Ea* is already in pascals, omit the × 1 000 factor.
        * The function is fully vectorised and Numba-ready (`@njit`) for large
          datasets.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> calc.calc_pV(Ea=1.2, Ts=293.15)        # Ea in kPa
        0.004179274732149666
        >>> import numpy as np
        >>> Ea_kpa = np.array([0.8, 1.0, 1.2])
        >>> Ts_k   = np.array([283.15, 293.15, 303.15])
        >>> calc.calc_pV(Ea_kpa, Ts_k)
        array([0.00278239, 0.00351606, 0.00422463])
        """
        return (Ea * 1000.0) / (self.Rv * Ts)

    def calc_max_covariance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int = 10,
    ):
        """
        Search a symmetric *lag window* for the **maximum (and minimum)
        covariance** between two time-aligned series and return summary
        statistics together with the full lag–covariance map.

        For every integer lag ℓ in the range −*lag* … +*lag* (inclusive) the
        method computes the sample covariance

        .. math::

            \\operatorname{cov}(x_{t+ℓ}, y_t),

        achieved by slicing the arrays appropriately (no wrapping).  Three key
        extrema are then extracted:

        * **maxcov** – largest positive covariance and its lag
        * **mincov** – largest negative covariance and its lag
        * **abscov** – covariance with greatest *absolute* magnitude
          (which could be positive or negative)

        Parameters
        ----------
        x : ndarray
            First input series (*reference*).  Must be one-dimensional and at
            least *lag*+1 samples long.
        y : ndarray
            Second input series (*target*), same shape as *x*.
        lag : int, default ``10``
            Maximum lag (in samples) explored both forward (+) and backward
            (−).  The total window therefore spans ``2*lag + 1`` covariance
            values.

        Returns
        -------
        abscov : tuple
            ``(lag_of_abs_extreme, covariance_value)``.
        maxcov : tuple
            ``(lag_of_max_positive, covariance_value)``.
        mincov : tuple
            ``(lag_of_max_negative, covariance_value)``.
        xy : dict
            Dictionary mapping every examined lag to its covariance value,
            e.g. ``{0: 0.01234, 1: 0.0101, …, -10: -0.0034}``.

        Notes
        -----
        * Covariances are computed via :func:`numpy.cov` (unbiased, *N–1*
          denominator) on the sliced arrays.
        * The function relies on an auxiliary method :meth:`findextreme`
          (defined elsewhere in the class) to locate extrema in *vals*.
        * No NaN handling is applied; pre-clean or mask missing data.

        Examples
        --------
        >>> import numpy as np
        >>> from ec import CalcFlux
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=1000)
        >>> y = np.roll(x, 4) + rng.normal(scale=0.1, size=1000)  # y lags x by 4
        >>> calc = CalcFlux()
        >>> abscov, maxcov, mincov, covdict = calc.calc_max_covariance(x, y, lag=10)
        >>> maxcov
        (4, covdict[4])  # maximum positive covariance at lag 4
        """
        xy = {}

        for i in range(0, lag + 1):
            if i == 0:
                xy[0] = np.round(np.cov(x, y)[0][1], 8)
            else:
                # covariance for positive lags
                xy[i] = np.round(np.cov(x[i:], y[: -1 * i])[0][1], 8)
                # covariance for negative lags
                xy[-i] = np.round(np.cov(x[: -1 * i], x[i:])[0][1], 8)

        # convert dictionary to arrays
        keys = np.array(list(xy.keys()))
        vals = np.array(list(xy.values()))

        # get index and value for maximum positive covariance
        valmax, maxlagindex = self.findextreme(vals, ext="max")
        maxlag = keys[maxlagindex]
        maxcov = (maxlag, valmax)

        # get index and value for maximum negative covariance
        valmin, minlagindex = self.findextreme(vals, ext="min")
        minlag = keys[minlagindex]
        mincov = (minlag, valmin)

        # get index and value for maximum absolute covariance
        absmax, abslagindex = self.findextreme(vals, ext="min")
        absmaxlag = keys[abslagindex]
        abscov = (absmaxlag, absmax)

    def findextreme(self, vals, ext: str = "abs"):
        """
        Locate an **extreme value** (absolute, minimum, or maximum) in a
        one-dimensional array and return the value together with its index.

        Parameters
        ----------
        vals : array_like
            Numeric 1-D array (NumPy array, list, or similar).  Internally
            converted to ``np.ndarray`` for processing.
        ext : {'abs', 'min', 'max'}, default ``'abs'``
            Which extreme to search for:

            * ``'abs'`` – largest **absolute** magnitude
            * ``'max'`` – largest **positive** value
            * ``'min'`` – largest **negative** (most negative) value

        Returns
        -------
        bigval : float
            The extreme value requested.
        lagindex : int
            Index of *bigval* within *vals*.

        Notes
        -----
        * ``np.nan`` values are ignored in the extrema search
          (``np.nanmax`` / ``np.nanmin``).
        * If multiple indices tie for the extreme, the **first** occurrence is
          returned.
        * The function is primarily a helper for
          :meth:`calc_max_covariance`, but is general-purpose.

        Examples
        --------
        >>> import numpy as np
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> arr = np.array([-3, 5, -7, 2])
        >>> calc.findextreme(arr, ext='abs')
        (7, 2)
        >>> calc.findextreme(arr, ext='max')
        (5, 1)
        >>> calc.findextreme(arr, ext='min')
        (-7, 2)
        """
        if ext == "abs":
            vals = np.abs(vals)
            bigval = np.nanmax(vals)
        elif ext == "max":
            bigval = np.nanmax(vals)
        elif ext == "min":
            bigval = np.nanmin(vals)
        else:
            vals = np.abs(vals)
            bigval = np.nanmax(np.abs(vals))

        lagindex = np.where(vals == bigval)[0][0]

        return bigval, lagindex

    def calc_max_covariance_df(
        self,
        df: pd.DataFrame,
        colx: str,
        coly: str,
        lags: int = 10,
    ) -> tuple[float, int]:
        """
        Search a *data-frame* column pair for the lag that yields the **largest
        absolute covariance** and return both the covariance value and its lag.

        The routine

        1. Creates shifted copies of *coly* for every integer lag ``ℓ`` in
           ``[-lags, …, lags)`` (note the **half-open** upper bound).
        2. Computes the unbiased sample covariance between *colx* and each
           shifted series using :pymeth:`pandas.DataFrame.cov`.
        3. Identifies the shift that maximises ``|cov|`` and returns:

           ``maxcov`` – covariance at that lag
           ``lagno``  – lag (samples) where the maximum occurs

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing both variables.
        colx : str
            Name of the **reference** column (*x*).
        coly : str
            Name of the **target** column (*y*), which is shifted.
        lags : int, default ``10``
            Half-window of lags explored.  The function inspects
            ``2*lags`` covariances (from −lags to +lags−1).

        Returns
        -------
        maxcov : float
            Covariance corresponding to the maximum absolute magnitude.
        lagno : int
            Lag (samples) at which *maxcov* occurs.
            Positive ⇒ *y* **lags** *x*, negative ⇒ *y* **leads** *x*.

        Notes
        -----
        * Intermediate shifted columns are **dropped immediately** after each
          covariance calculation to minimise memory overhead.
        * If multiple lags tie for |cov|max, the **first** occurrence in the
          scan order is chosen.
        * The half-open range means the largest positive lag checked is
          ``lags − 1``; adjust if symmetric bounds are desired.

        Examples
        --------
        >>> import polars as pl
import pandas as pd  # optional compat, numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 5000
        >>> df = pd.DataFrame({
        ...     "u": rng.standard_normal(n),
        ... })
        >>> df["v"] = df["u"].shift(5).fillna(0) + rng.normal(0, 0.05, n)
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> cov, lag = calc.calc_max_covariance_df(df, "u", "v", lags=10)
        >>> lag
        5
        """
        dfcov = []
        for i in np.arange(-1 * lags, lags):
            df[f"{coly}_{i}"] = df[coly].shift(i)
            dfcov.append(df[[colx, f"{coly}_{i}"]].cov().loc[colx, f"{coly}_{i}"])
            df = df.drop([f"{coly}_{i}"], axis=1)

        abscov = np.abs(dfcov)
        maxabscov = np.max(abscov)
        try:
            maxlagindex = np.where(abscov == maxabscov)[0][0]
            lagno = maxlagindex - lags
            maxcov = dfcov[maxlagindex]
        except IndexError:
            lagno = 0
            maxcov = dfcov[10]
        return maxcov, lagno

    def coord_rotation(
        self,
        df: pd.DataFrame | None = None,
        Ux: str = "Ux",
        Uy: str = "Uy",
        Uz: str = "Uz",
    ):
        """
        Perform the **double (planar-fit) coordinate rotation** that aligns the
        sonic-anemometer axes with the mean wind direction and sets the mean
        vertical velocity to zero.

        The routine follows the classical scheme of Tanner & Thurtell (1969)
        and Hyson et al. (1977) as summarised in Kaimal & Finnigan (1994):

        1. **Yaw rotation** (about *w*) aligns the *x*-axis with the mean
           horizontal wind, defining

           .. math:: \\cos\\nu = \\frac{\\bar u}{\\sqrt{\\bar u^2+\\bar v^2}},\\qquad
                     \\sin\\nu = \\frac{\\bar v}{\\sqrt{\\bar u^2+\\bar v^2}}.

        2. **Pitch rotation** (about the new *y′* axis) sets
           :math:`\\bar w′ = 0`, giving

           .. math:: \\sin\\theta = \\frac{\\bar w}{\\sqrt{\\bar u^2+\\bar v^2+\\bar w^2}},\\qquad
                     \\cos\\theta = \\frac{\\sqrt{\\bar u^2+\\bar v^2}}
                                          {\\sqrt{\\bar u^2+\\bar v^2+\\bar w^2}}.

        These four trigonometric factors are cached on the instance for later
        use in covariance rotation and spectral corrections.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Dataframe containing raw wind components.  If *None*, the method
            uses the instance attribute ``self.df`` already set elsewhere.
        Ux, Uy, Uz : str, default ``'Ux'`, ``'Uy'``, ``'Uz'``
            Column names for the longitudinal (*u*), lateral (*v*), and
            vertical (*w*) wind components.

        Returns
        -------
        tuple
            ``(cosν, sinν, sinθ, cosθ, Uxy, Uxyz)`` where

            * **cosν**, **sinν** – yaw-rotation cos/sin
            * **sinθ**, **cosθ** – pitch-rotation sin/cos
            * **Uxy**  – mean horizontal wind speed
              :math:`\\sqrt{\\bar u^2+\\bar v^2}` (m s⁻¹)
            * **Uxyz** – mean wind magnitude
              :math:`\\sqrt{\\bar u^2+\\bar v^2+\\bar w^2}` (m s⁻¹)

        Notes
        -----
        * The rotation angles are stored in ``self.cosv``, ``self.sinv``,
          ``self.sinTheta``, and ``self.cosTheta`` for reuse by
          :meth:`covar_coord_rot_correction` and other routines.
        * No *roll* rotation (third, about the *x″* axis) is applied here;
          add it if instrument tilt correction is required.
        * Prior detrending is **not** performed; mean values are taken
          directly from *df*.

        References
        ----------
        Tanner, B. D., & Thurtell, G. W. (1969). *Anemoclinometer
        measurements of Reynolds stress and heat transport in the atmospheric
        surface layer*.
        Hyson, P., et al. (1977).
        Kaimal, J. C., & Finnigan, J. J. (1994). *Atmospheric Boundary Layer
        Flows*.

        Examples
        --------
        >>> import polars as pl
import pandas as pd  # optional compat, numpy as np
        >>> rng = np.random.default_rng(1)
        >>> df = pd.DataFrame({
        ...     "Ux": rng.normal(2.0, 0.5, 10000),
        ...     "Uy": rng.normal(0.3, 0.2, 10000),
        ...     "Uz": rng.normal(0.05, 0.1, 10000),
        ... })
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> cosv, sinv, sinT, cosT, Uxy, Uxyz = calc.coord_rotation(df)
        >>> round(Uxy, 3), round(Uxyz, 3)
        (2.021, 2.022)
        """
        if df is None:
            df = self.df
        else:
            pass

        xmean = df[Ux].mean()
        ymean = df[Uy].mean()
        zmean = df[Uz].mean()
        Uxy = np.sqrt(xmean**2 + ymean**2)
        Uxyz = np.sqrt(xmean**2 + ymean**2 + zmean**2)

        # save for later use
        self.cosv = xmean / Uxy
        self.sinv = ymean / Uxy
        self.sinTheta = zmean / Uxyz
        self.cosTheta = Uxy / Uxyz

        return self.cosv, self.sinv, self.sinTheta, self.cosTheta, Uxy, Uxyz

    def rotate_velocity_values(
        self,
        df: pd.DataFrame | None = None,
        Ux: str = "Ux",
        Uy: str = "Uy",
        Uz: str = "Uz",
    ) -> pd.DataFrame:
        """
        Apply the **double‐rotation matrix** (yaw + pitch) obtained from
        :meth:`coord_rotation` to every instantaneous wind sample, producing
        velocity components aligned with the mean flow.

        For each record the transformation is

        .. math::

            \\begin{aligned}
            u_r &=  u\\cos\\theta\\cos\\nu
                 + v\\cos\\theta\\sin\\nu
                 + w\\sin\\theta,\\\\[4pt]
            v_r &=  v\\cos\\nu - u\\sin\\nu,\\\\[4pt]
            w_r &=  w\\cos\\theta
                 - u\\sin\\theta\\cos\\nu
                 - v\\sin\\theta\\sin\\nu,
            \\end{aligned}

        where :math:`\\nu` and :math:`\\theta` are the yaw and pitch angles
        cached in ``self.cosv``, ``self.sinv``, ``self.cosTheta``,
        ``self.sinTheta`` by a prior call to :meth:`coord_rotation`.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            DataFrame containing raw wind components.  If *None*, the method
            uses the instance attribute ``self.df``.
        Ux, Uy, Uz : str, default ``'Ux'``, ``'Uy'``, ``'Uz'``
            Column names for the longitudinal (*u*), lateral (*v*), and
            vertical (*w*) wind components.

        Returns
        -------
        pandas.DataFrame
            Same object with **three new columns** added:

            * ``Uxr`` – rotated longitudinal component *u_r* (m s⁻¹)
            * ``Uyr`` – rotated lateral component     *v_r* (m s⁻¹)
            * ``Uzr`` – rotated vertical component    *w_r* (m s⁻¹)

            The instance attribute ``self.df`` is updated in place.

        Notes
        -----
        * **Must** call :meth:`coord_rotation` first; otherwise the cached
          trigonometric factors are ``None`` and the method prints a warning.
        * Only the first two rotations of a full triple rotation are applied
          here; *roll* (tilt) correction is handled elsewhere if required.

        Examples
        --------
        >>> import polars as pl
import pandas as pd  # optional compat, numpy as np
        >>> from ec import CalcFlux
        >>> rng = np.random.default_rng(2)
        >>> df = pd.DataFrame({
        ...     "Ux": rng.normal(2.0, 0.3, 5000),
        ...     "Uy": rng.normal(0.4, 0.2, 5000),
        ...     "Uz": rng.normal(0.05, 0.1, 5000),
        ... })
        >>> calc = CalcFlux()
        >>> calc.coord_rotation(df)          # computes cosv, sinv, etc.
        >>> df_rot = calc.rotate_velocity_values(df)
        >>> df_rot[["Uxr", "Uyr", "Uzr"]].head()
               Uxr       Uyr       Uzr
        0  2.054905 -0.079402 -0.048402
        1  1.980984  0.094779 -0.009772
        2  1.999265 -0.093132  0.096403
        3  2.065206 -0.079623 -0.021688
        4  2.204025 -0.004639 -0.010206
        """
        if df is None:
            df = self.df

        if self.cosTheta is None:
            print("Please run coord_rotation")
        else:
            df["Uxr"] = (
                df[Ux] * self.cosTheta * self.cosv
                + df[Uy] * self.cosTheta * self.sinv
                + df[Uz] * self.sinTheta
            )
            df["Uyr"] = df[Uy] * self.cosv - df[Ux] * self.sinv
            df["Uzr"] = (
                df[Uz] * self.cosTheta
                - df[Ux] * self.sinTheta * self.cosv
                - df[Uy] * self.sinTheta * self.sinv
            )

            self.df = df
            return df

    def rotated_components_statistics(
        self,
        df: pd.DataFrame | None = None,
        Ux: str = "Ux",
        Uy: str = "Uy",
        Uz: str = "Uz",
    ):
        """
        Compute **means** and **standard deviations** of the *rotated* wind
        components and store them in the instance dictionaries
        ``self.avgvals`` and ``self.stdvals``.

        Prerequisite
        ------------
        The dataframe *df* **must already contain** the columns
        ``'Uxr'``, ``'Uyr'``, and ``'Uzr'`` produced by
        :meth:`rotate_velocity_values`.
        The original (unrotated) means ``self.avgvals['Ux']``, etc., should
        likewise be available—typically set earlier by
        :meth:`coord_rotation`.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            DataFrame holding both raw and rotated wind components.  If
            *None*, the method uses ``self.df``.
        Ux, Uy, Uz : str, default ``'Ux'``, ``'Uy'``, ``'Uz'``
            Column names for the **raw** wind components; used only for the
            auxiliary calculation of ``self.avgvals['Uav']``.

        Returns
        -------
        None
            The function operates **in place**, updating:

            * ``self.avgvals['Uxr'|'Uyr'|'Uzr']`` – mean rotated components
            * ``self.stdvals['Uxr'|'Uyr'|'Uzr']`` – standard deviations
            * ``self.avgvals['Uav']`` – mean wind speed along the rotated
              *x′*-axis (scalar).

        Notes
        -----
        * ``self.avgvals`` and ``self.stdvals`` are ordinary Python dicts
          initialised in :meth:`__init__`.
        * ``Uav`` is recomputed from the **raw** means and the cached rotation
          angles (*cosν*, *sinν*, *cosθ*, *sinθ*); it should equal
          ``self.avgvals['Uxr']`` within numerical tolerance if all steps were
          executed correctly.

        Examples
        --------
        >>> from ec import CalcFlux
        >>> calc = CalcFlux()
        >>> # assume df already has Uxr/Uyr/Uzr via prior rotation
        >>> calc.rotated_components_statistics(df)
        >>> calc.avgvals['Uxr'], calc.stdvals['Uxr']
        (2.021, 0.503)
        """
        if df is None:
            df = self.df

        # Means and standard deviations of rotated components
        self.avgvals["Uxr"] = df["Uxr"].mean()
        self.avgvals["Uyr"] = df["Uyr"].mean()
        self.avgvals["Uzr"] = df["Uzr"].mean()
        self.stdvals["Uxr"] = df["Uxr"].std()
        self.stdvals["Uyr"] = df["Uyr"].std()
        self.stdvals["Uzr"] = df["Uzr"].std()

        # Auxiliary: mean wind speed along rotated x′ axis (should ≈ Uxr mean)
        self.avgvals["Uav"] = (
            self.avgvals["Ux"] * self.cosTheta * self.cosv
            + self.avgvals["Uy"] * self.cosTheta * self.sinv
            + self.avgvals["Uz"] * self.sinTheta
        )
        return

    def dayfrac(self, df: pd.DataFrame) -> float:
        """
        Return the **fraction of a day** represented by the time span between
        the first and last **valid** index entries of *df*.

        The calculation is

        .. math::

            \\text{dayfrac} =
              \\frac{t_{\\text{last}} - t_{\\text{first}}}{1\\;\\text{day}},

        where *t_first* and *t_last* are obtained via
        :pymeth:`pandas.DataFrame.first_valid_index` and
        :pymeth:`pandas.DataFrame.last_valid_index`.

        Parameters
        ----------
        df : pandas.DataFrame
            Time-indexed dataframe (typically 10 Hz or 20 Hz raw flux data).
            Any index entries that are ``NaT`` or rows that are entirely ``NaN``
            are ignored by the “valid” index methods.

        Returns
        -------
        float
            Fractional part of a 24-hour day covered by the data, e.g.
            ``0.020833`` for a 30-minute period.

        Examples
        --------
        >>> import polars as pl
import pandas as pd  # optional compat
        >>> from ec import CalcFlux
        >>> idx = pd.date_range("2025-06-21 13:30", periods=36000, freq="100L")  # 1 h @ 10 Hz
        >>> df = pd.DataFrame({"Ux": 0.0}, index=idx)
        >>> CalcFlux().dayfrac(df)
        0.041666666666666664  # 1 h / 24 h
        """
        return (df.last_valid_index() - df.first_valid_index()) / pd.to_timedelta(
            1, unit="D"
        )

    def calc_covar(
        self,
        Ux: np.ndarray,
        Uy: np.ndarray,
        Uz: np.ndarray,
        Ts: np.ndarray,
        Q: np.ndarray,
        pV: np.ndarray,
    ):
        """
        Compute the **second-order moments (variances and covariances)**
        required for eddy-flux calculations and store them in
        :pyattr:`self.covar`.

        For every input pair *(a, b)* the routine calls
        :meth:`calc_cov(a, b)`—an unbiased, vectorised covariance
        implementation—to obtain

        =========================  Stored under key      Units
        -------------------------  ---------------------  -----
        ``cov(Ts, Ts)``            ``'Ts-Ts'``            K²
        ``cov(Ux, Ux)``            ``'Ux-Ux'``            (m s⁻¹)²
        ``cov(Uy, Uy)``            ``'Uy-Uy'``            (m s⁻¹)²
        ``cov(Uz, Uz)``            ``'Uz-Uy'``\*          (m s⁻¹)²
        ``cov(Q, Q)``              ``'Q-Q'``              (kg kg⁻¹)²
        ``cov(pV, pV)``            ``'pV-pV'``            (kg m⁻³)²
        =========================  =====================  =====

        \* *Note:* the key ``"Uz-Uy"`` follows the original code and appears
        to be a typographical slip; it actually stores
        ``cov(Uz, Uz)``.

        Parameters
        ----------
        Ux, Uy, Uz : ndarray
            Instantaneous wind components *u*, *v*, *w* (m s⁻¹).
        Ts : ndarray
            Sonic (virtual) temperature (K).
        Q : ndarray
            Specific humidity (kg kg⁻¹).
        pV : ndarray
            Water-vapour density (kg m⁻³).

        Returns
        -------
        None
            Covariance terms are saved to the instance dictionary
            ``self.covar`` for later use by spectral corrections,
            WPL density adjustments, and flux computations.

        Examples
        --------
        >>> import numpy as np
        >>> from ec import CalcFlux
        >>> rng = np.random.default_rng(3)
        >>> N = 20000
        >>> calc = CalcFlux()
        >>> calc.calc_covar(
        ...     Ux=rng.normal(0, 1, N),
        ...     Uy=rng.normal(0, 1, N),
        ...     Uz=rng.normal(0, 0.2, N),
        ...     Ts=rng.normal(300, 0.5, N),
        ...     Q=rng.normal(0.01, 1e-4, N),
        ...     pV=rng.normal(0.004, 2e-5, N),
        ... )
        >>> calc.covar.keys()
        dict_keys(['Ts-Ts', 'Ux-Ux', 'Uy-Uy', 'Uz-Uy', 'Q-Q', 'pV-pV'])
        """
        self.covar["Ts-Ts"] = self.calc_cov(Ts, Ts)
        self.covar["Ux-Ux"] = self.calc_cov(Ux, Ux)
        self.covar["Uy-Uy"] = self.calc_cov(Uy, Uy)
        self.covar["Uz-Uy"] = self.calc_cov(Uz, Uz)  # key kept as in original
        self.covar["Q-Q"] = self.calc_cov(Q, Q)
        self.covar["pV-pV"] = self.calc_cov(pV, pV)
