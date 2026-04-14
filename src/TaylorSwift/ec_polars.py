# Original scripts in Fortran by Lawrence Hipps USU
# Transcibed from original Visual Basic scripts by Clayton Lewis and Lawrence Hipps

import polars as pl
import pandas as pd
import numpy as np
from scipy import signal
from typing import Union

from .constants import (
    K_VON_KARMAN,
    G0,
    R_GAS,
    OMEGA,
    SIGMA_SB,
    CP_DRY_AIR,
    MOLAR_MASS,
    R_SPECIFIC,
    T_ZERO_C,
)

# ---------------------------------------------------------------------------
# Hardy (1998) ITS-90 polynomial coefficients for dew/frost point
# ---------------------------------------------------------------------------
_HARDY_C = (207.98233, -20.156028, 0.46778925, -9.2288067e-6)
_HARDY_D = (1.0, -0.13319669, 0.0056577518, -7.5172865e-5)


# ---------------------------------------------------------------------------
# Polars/pandas compatibility helpers
# ---------------------------------------------------------------------------
# ── top-of-module helper ───────────────────────────────────────────────────────
_AnyFrame = Union[pd.DataFrame, pl.DataFrame]


def _is_polars(df: _AnyFrame) -> bool:
    return isinstance(df, pl.DataFrame)


def _is_pandas(df: _AnyFrame | None) -> bool:
    return isinstance(df, pd.DataFrame)


def _validate_frame(
    df: _AnyFrame | None, *, allow_none: bool = True
) -> _AnyFrame | None:
    """Validate that *df* is a supported DataFrame backend."""
    if df is None and allow_none:
        return None
    if isinstance(df, (pd.DataFrame, pl.DataFrame)):
        return df
    raise TypeError(f"CalcFlux expects a pandas or polars DataFrame, got {type(df)}")


def _to_pl_df(df):
    """Convert a pandas or Polars DataFrame to a Polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(f"Expected DataFrame, got {type(df)}")


def _to_same_type(original, pl_df):
    """Convert *pl_df* back to the same type as *original*."""
    if isinstance(original, pd.DataFrame):
        return pl_df.to_pandas()
    return pl_df


def _get_series(df, col: str) -> pl.Series:
    """Extract a column as a Polars Series regardless of input type."""
    if isinstance(df, pd.DataFrame):
        return pl.Series(col, df[col].values)
    return df[col]


def _assign(df, **kwargs):
    """Assign new columns (numpy arrays) to a DataFrame of either type."""
    if isinstance(df, pd.DataFrame):
        return df.assign(**{k: v for k, v in kwargs.items()})
    exprs = [pl.Series(k, v) for k, v in kwargs.items()]
    return df.with_columns(exprs)

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

def _first_last_index_duration(df:pd.DataFrame, unit:str="D") -> float | None:
    """Mimic (df.last_valid_index() - df.first_valid_index()) / pd.to_timedelta(1, unit=unit)"""
    if isinstance(df, pd.DataFrame) and df.index.size>0 and isinstance(df.index, pd.DatetimeIndex):
        delta = (df.last_valid_index() - df.first_valid_index()) # type: ignore
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
        # ---- physical constants (from constants.py) ----------------------------
        self.Cp = None
        self.Rv = R_SPECIFIC["water_vapor"]
        self.Ru = R_GAS
        self.Cpd = CP_DRY_AIR
        self.Rd = R_SPECIFIC["dry_air"]
        self.md = MOLAR_MASS["air_dry"]
        self.Co = 0.21  # Atmospheric O₂ molar fraction
        self.Mo = 0.032  # O₂ molar mass (kg mol⁻¹)

        # ---- thermodynamic & spectral constants --------------------------------
        self.Cpw = 1952.0  # c_p of H₂O vapour (J kg⁻¹ K⁻¹)
        self.Cw = 4218.0  # c_p of liquid water (J kg⁻¹ K⁻¹)
        self.epsilon = MOLAR_MASS["h2o"] / MOLAR_MASS["air_dry"]
        self.g = G0
        self.von_karman = K_VON_KARMAN
        self.MU_WPL = MOLAR_MASS["air_dry"] / MOLAR_MASS["h2o"]
        self.Omega = OMEGA
        self.Sigma_SB = SIGMA_SB

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

    def set_dataframe(self, df: _AnyFrame | None) -> _AnyFrame | None:
        """Store a pandas or polars DataFrame on the instance."""
        self.df = _validate_frame(df)
        return self.df

    # -----------------------------------------------------------------
    # Wind direction & coordinate rotation
    # -----------------------------------------------------------------

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
        if uxavg is not None:
            if update_existing_vel:
                self.avgvals["Ux"] = uxavg
        else:
            if "Ux" in self.avgvals:
                uxavg = self.avgvals["Ux"]
            else:
                print("Please calculate wind velocity averages")
                return

        if uyavg is not None:
            if update_existing_vel:
                self.avgvals["Uy"] = uyavg
        else:
            if "Uy" in self.avgvals:
                uyavg = self.avgvals["Uy"]
            else:
                print("Please calculate wind velocity averages")
                return

        # rest of method unchanged …
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
        self.pathlen = self.PathDist_U * np.abs(np.sin((np.pi / 180) * wind_compass))
        return self.pathlen, self.wind_compass

    def coord_rotation(
        self,
        df: _AnyFrame | None = None,
        Ux: str = "Ux",
        Uy: str = "Uy",
        Uz: str = "Uz",
    ):
        """
        Perform double coordinate rotation that aligns the sonic-anemometer
        axes with the mean wind direction and sets the mean vertical velocity
        to zero, following Tanner & Thurtell (1969) and Kaimal & Finnigan (1994).
        """
        if df is None:
            df = self.df
        df = _validate_frame(df, allow_none=False)

        # .mean() returns a Python float in both pandas and polars
        xmean = df[Ux].mean()  # type: ignore
        ymean = df[Uy].mean()  # type: ignore
        zmean = df[Uz].mean()  # type: ignore
        Uxy = np.sqrt(xmean**2 + ymean**2)
        Uxyz = np.sqrt(xmean**2 + ymean**2 + zmean**2)

        self.cosv = xmean / Uxy
        self.sinv = ymean / Uxy
        self.sinTheta = zmean / Uxyz
        self.cosTheta = Uxy / Uxyz

        return self.cosv, self.sinv, self.sinTheta, self.cosTheta, Uxy, Uxyz

    def rotate_velocity_values(
        self,
        df: _AnyFrame | None = None,
        Ux: str = "Ux",
        Uy: str = "Uy",
        Uz: str = "Uz",
    ) -> _AnyFrame:
        """
        Apply the double-rotation matrix (yaw + pitch) from
        :meth:`coord_rotation` to every instantaneous wind sample.
        """
        if df is None:
            df = self.df
        df = _validate_frame(df, allow_none=False)

        if self.cosTheta is None:
            print("Please run coord_rotation")
            return df

        cv, sv = self.cosv, self.sinv
        cT, sT = self.cosTheta, self.sinTheta

        if _is_polars(df):
            df = df.with_columns(  # type: ignore
                (pl.col(Ux) * cT * cv + pl.col(Uy) * cT * sv + pl.col(Uz) * sT).alias(
                    "Uxr"
                ),
                (pl.col(Uy) * cv - pl.col(Ux) * sv).alias("Uyr"),
                (pl.col(Uz) * cT - pl.col(Ux) * sT * cv - pl.col(Uy) * sT * sv).alias(
                    "Uzr"
                ),
            )  # type: ignore
        else:
            df = df.copy()  # type: ignore # avoid mutating the caller's frame
            df["Uxr"] = df[Ux] * cT * cv + df[Uy] * cT * sv + df[Uz] * sT
            df["Uyr"] = df[Uy] * cv - df[Ux] * sv
            df["Uzr"] = df[Uz] * cT - df[Ux] * sT * cv - df[Uy] * sT * sv

        self.df = df
        return df

    def rotated_components_statistics(
        self,
        df: _AnyFrame | None = None,
        Ux: str = "Ux",
        Uy: str = "Uy",
        Uz: str = "Uz",
    ):
        """
        Compute means and standard deviations of the rotated wind components.
        """
        if df is None:
            df = self.df
        df = _validate_frame(df, allow_none=False)

        # .mean() / .std() return Python scalars in both backends
        for col in ("Uxr", "Uyr", "Uzr"):
            self.avgvals[col] = df[col].mean()  # type: ignore
            self.stdvals[col] = df[col].std()  # type: ignore

        self.avgvals["Uav"] = (
            self.avgvals["Ux"] * self.cosTheta * self.cosv  # type: ignore
            + self.avgvals["Uy"] * self.cosTheta * self.sinv  # type: ignore
            + self.avgvals["Uz"] * self.sinTheta  # type: ignore
        )
        return

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

        Uz_Ts = (
            self.covar["Uz-Tsa"] * cosTheta  # type: ignore
            - self.covar["Ux-Tsa"] * sinTheta * cosν  # type: ignore
            - self.covar["Uy-Tsa"] * sinTheta * sinv  # type: ignore
        )
        if np.abs(Uz_Ts) >= np.abs(self.covar["Uz-Tsa"]):
            self.covar["Uz-Tsa"] = Uz_Ts

        Uz_pV = (
            self.covar["Uz-pV"] * cosTheta  # type: ignore
            - self.covar["Ux-pV"] * sinTheta * cosν  # type: ignore
            - self.covar["Uy-pV"] * sinv * sinTheta  # type: ignore
        )
        if np.abs(Uz_pV) >= np.abs(self.covar["Uz-pV"]):
            self.covar["Uz-pV"] = Uz_pV
        self.covar["Ux-Q"] = (
            self.covar["Ux-Q"] * cosTheta * cosν  # type: ignore
            + self.covar["Uy-Q"] * cosTheta * sinv  # type: ignore
            + self.covar["Uz-Q"] * sinTheta  # type: ignore
        )
        self.covar["Uy-Q"] = self.covar["Uy-Q"] * cosν - self.covar["Uy-Q"] * sinv  # type: ignore
        self.covar["Uz-Q"] = (
            self.covar["Uz-Q"] * cosTheta  # type: ignore
            - self.covar["Ux-Q"] * sinTheta * cosν  # type: ignore
            - self.covar["Uy-Q"] * sinv * sinTheta  # type: ignore
        )
        self.covar["Ux-Uz"] = (
            self.covar["Ux-Uz"] * cosν * (cosTheta**2 - sinTheta**2)  # type: ignore
            - 2 * self.covar["Ux-Uy"] * sinTheta * cosTheta * sinv * cosν  # type: ignore
            + self.covar["Uy-Uz"] * sinv * (cosTheta**2 - sinTheta**2)  # type: ignore
            - self.errvals["Ux"] * sinTheta * cosTheta * cosν**2  # type: ignore
            - self.errvals["Uy"] * sinTheta * cosTheta * sinv**2  # type: ignore
            + self.errvals["Uz"] * sinTheta * cosTheta  # type: ignore
        )
        self.covar["Uy-Uz"] = (
            self.covar["Uy-Uz"] * cosTheta * cosν  # type: ignore
            - self.covar["Ux-Uz"] * cosTheta * sinv  # type: ignore
            - self.covar["Ux-Uy"] * sinTheta * (cosν**2 - sinv**2)  # type: ignore
            + self.errvals["Ux"] * sinTheta * sinv * cosν  # type: ignore
            - self.errvals["Uy"] * sinTheta * sinv * cosν  # type: ignore
        )
        self.covar["Uz-Sd"] = (
            self.covar["Uz-Sd"] * cosTheta  # type: ignore
            - self.covar["Ux-Sd"] * sinTheta * cosν  # type: ignore
            - self.covar["Uy-Sd"] * sinv * sinTheta  # type: ignore
        )
        self.covar["Uxy-Uz"] = np.sqrt(
            self.covar["Ux-Uz"] ** 2 + self.covar["Uy-Uz"] ** 2
        )

    # -----------------------------------------------------------------
    # Signal processing
    # -----------------------------------------------------------------

    def get_lag(self, x: np.ndarray, y: np.ndarray) -> int:
        """Compute the sample lag that maximises cross-correlation."""
        correlation = signal.correlate(x, y, mode="full")
        lags = signal.correlation_lags(x.size, y.size, mode="full")
        lag = lags[np.argmax(correlation)]
        return lag

    def calc_max_covariance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int = 10,
    ):
        """
        Search a symmetric lag window for the maximum and minimum
        covariance between two time-aligned series.
        """
        xy = {}

        for i in range(0, lag + 1):
            if i == 0:
                xy[0] = np.round(np.cov(x, y)[0][1], 8)
            else:
                xy[i] = np.round(np.cov(x[i:], y[: -1 * i])[0][1], 8)
                xy[-i] = np.round(np.cov(x[: -1 * i], x[i:])[0][1], 8)

        keys = np.array(list(xy.keys()))
        vals = np.array(list(xy.values()))

        valmax, maxlagindex = self.findextreme(vals, ext="max")
        maxlag = keys[maxlagindex]
        maxcov = (maxlag, valmax)

        valmin, minlagindex = self.findextreme(vals, ext="min")
        minlag = keys[minlagindex]
        mincov = (minlag, valmin)

        absmax, abslagindex = self.findextreme(vals, ext="abs")
        absmaxlag = keys[abslagindex]
        abscov = (absmaxlag, absmax)

        return maxcov, mincov, abscov, xy

    def findextreme(self, vals, ext: str = "abs"):
        """Locate an extreme value (absolute, minimum, or maximum) in a 1-D array."""
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

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    def calc_cov(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute covariance using the manually expanded formula."""
        sumproduct = np.sum(p1 * p2)
        return (sumproduct - (np.sum(p1) * np.sum(p2)) / len(p1)) / (len(p1) - 1)

    def calc_MSE(self, y: np.ndarray) -> float:
        """Compute the population variance (mean square error)."""
        return np.mean((y - np.mean(y)) ** 2)

    # -----------------------------------------------------------------
    # Thermodynamics & conversions
    # -----------------------------------------------------------------

    def convert_KtoC(self, T: float | np.ndarray) -> float | np.ndarray:
        """Convert temperature from kelvin to degrees Celsius."""
        return T - T_ZERO_C

    def convert_CtoK(self, T: float | np.ndarray) -> float | np.ndarray:
        """Convert temperature from degrees Celsius to kelvin."""
        return T + T_ZERO_C

    def calc_Td_dewpoint(self, E: float | np.ndarray) -> float | np.ndarray:
        """
        Convert water-vapour pressure to dew-point temperature using the
        ITS-90 polynomial formulation of Hardy (1998).
        """
        c0, c1, c2, c3 = _HARDY_C
        d0, d1, d2, d3 = _HARDY_D
        lne = np.log(E)
        nom = c0 + c1 * lne + c2 * lne**2 + c3 * lne**3
        denom = d0 + d1 * lne + d2 * lne**2 + d3 * lne**3
        return nom / denom

    def calc_Tf_frostpoint(self, E: float | np.ndarray) -> float | np.ndarray:
        """
        Convert water-vapour pressure to frost-point temperature using the
        ITS-90 polynomial formulation of Hardy (1998).
        """
        c0, c1, c2, _ = _HARDY_C  # frost-point omits the c3·ln³ term
        d0, d1, d2, d3 = _HARDY_D
        lne = np.log(E)
        nom = c0 + c1 * lne + c2 * lne**2
        denom = d0 + d1 * lne + d2 * lne**2 + d3 * lne**3
        return nom / denom

    def calc_E(
        self, pV: float | np.ndarray, T: float | np.ndarray
    ) -> float | np.ndarray:
        """Compute actual vapour pressure from water-vapour density and
        absolute temperature via the ideal-gas law."""
        return pV * T * self.Rv

    def calc_Q(
        self,
        P: float | np.ndarray,
        e: float | np.ndarray,
    ) -> float | np.ndarray:
        """Convert total air pressure and vapour partial pressure to
        specific humidity (Bolton 1980)."""
        gamma = 0.622
        return (gamma * e) / (P - 0.378 * e)

    def calc_tc_air_temp_sonic(self, Ts, pV, P):
        """
        Convert sonic (virtual) temperature to true air temperature
        following the Campbell Scientific EasyFlux implementation.
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
        """Convert sonic (virtual) temperature to air temperature using
        specific humidity (Schotanus et al. 1983)."""
        return Ts / (1 + 0.51 * q)

    def calc_Es(self, T: float | np.ndarray) -> float | np.ndarray:
        """Saturation vapour pressure over liquid water using Hardy's (1998)
        ITS-90 polynomial adaptation of the modified-Wexler equation."""
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

    def tetens(
        self,
        t: float | np.ndarray,
        a: float = 0.611,
        b: float = 17.502,
        c: float = 240.97,
    ) -> float | np.ndarray:
        """Saturation vapour pressure using the Magnus-Tetens approximation."""
        return a * np.exp((b * t) / (t + c))

    def calc_LnKh(self, mvolts: float | np.ndarray) -> float | np.ndarray:
        """Compute ln(Kw) from KH-20 krypton hygrometer output millivolts."""
        return self.XKwC1 + self.XKwC2 * mvolts

    # -----------------------------------------------------------------
    # Stability & flux
    # -----------------------------------------------------------------

    def calc_L(
        self,
        Ust: float | np.ndarray,
        Tsa: float | np.ndarray,
        Uz_Ta: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute the Monin-Obukhov length."""
        return (-1 * (Ust**3) * Tsa) / (self.g * self.von_karman * Uz_Ta)

    def calc_AlphX(
        self, L: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Evaluate Massman (2000, 2001) stability-correction factors α and X."""
        if (self.UHeight / L) <= 0:
            alph = 0.925
            X = 0.085
        else:
            alph = 1
            X = 2 - 1.915 / (1 + 0.5 * self.UHeight / L)
        return alph, X

    # -----------------------------------------------------------------
    # CSAT3A shadow correction
    # -----------------------------------------------------------------

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

        for _ in range(4):
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
            Ux = hinv[0] * Uxa + hinv[1] * Uya + hinv[2] * Uza
            Uy = hinv[3] * Uxa + hinv[4] * Uya + hinv[5] * Uza
            Uz = hinv[6] * Uxa + hinv[7] * Uya + hinv[8] * Uza

        return Ux, Uy, Uz

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------


def dayfrac(self, df: _AnyFrame, time_col: str | None = None) -> float:
    """
    Return the fraction of a day represented by the DataFrame's time span.

    For pandas, the method uses the DatetimeIndex by default to preserve the
    original behaviour. For polars, provide ``time_col`` or include at least
    one Datetime/Date column that can be inferred automatically.
    """
    df = _validate_frame(df, allow_none=False)

    if _is_pandas(df):
        if time_col is not None:
            time_values = pd.to_datetime(df[time_col])
            start = time_values.iloc[0]
            end = time_values.iloc[-1]
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError(
                    "pandas DataFrame must have a DatetimeIndex when time_col is not provided"
                )
            start = df.first_valid_index()
            end = df.last_valid_index()
    else:
        if time_col is None:
            datetime_cols = [
                col
                for col, dtype in zip(df.columns, df.dtypes)
                if str(dtype).startswith("Datetime") or str(dtype) == "Date"
            ]
            if not datetime_cols:
                raise TypeError(
                    "polars DataFrame requires a datetime/date column or an explicit time_col"
                )
            time_col = datetime_cols[0]

        time_values = df[time_col]
        if time_values.is_empty():
            return 0.0
        start = time_values.min()
        end = time_values.max()
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

    return (end - start) / pd.to_timedelta(1, unit="D")  # type: ignore
