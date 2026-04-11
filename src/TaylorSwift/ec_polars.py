# Original scripts in Fortran by Lawrence Hipps USU
# Transcibed from original Visual Basic scripts by Clayton Lewis and Lawrence Hipps

import polars as pl
import pandas as pd  # optional compat
import numpy as np
from scipy import signal
import statsmodels.api as sm


class CalcFlux:
    # … class-level attributes & docstring remain unchanged …

    def __init__(self, **kwargs):
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

    def determine_wind_dir(
        self,
        uxavg: float | None = None,
        uyavg: float | None = None,
        update_existing_vel: bool = False,
    ):

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


    def get_lag(self, x: np.ndarray, y: np.ndarray) -> int:
        """
        Compute the sample **lag** (in index units) that maximises the discrete
        cross-correlation between two equally spaced signals.

        """
        correlation = signal.correlate(x, y, mode="full")
        lags = signal.correlation_lags(x.size, y.size, mode="full")
        lag = lags[np.argmax(correlation)]
        return lag

    def calc_Td_dewpoint(self, E: float | np.ndarray) -> float | np.ndarray:
        """
        Convert water-vapour *pressure* to **dew-point temperature** using the
        ITS-90 polynomial formulation of Hardy (1998).
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
        """
        return (-1 * (Ust**3) * Tsa) / (self.g * self.von_karman * Uz_Ta)

  
    def calc_AlphX(
        self, L: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        Evaluate the empirical **α** and **X** stability-correction factors used
        in Massman’s (2000, 2001) scalar attenuation model.

        The coefficients depend on the sign and magnitude of the stability
        parameter *ζ = z/L*, where *z* is the sensor height (*self.UHeight*)
        and *L* is the Monin–Obukhov length provided via *L*.

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

        """
        return a * np.exp((b * t) / (t + c))

    # @numba.jit(forceobj=True)
    def calc_Es(self, T: float | np.ndarray) -> float | np.ndarray:
        """
        Saturation vapour pressure over **liquid water** calculated from
        temperature using Hardy’s (1998) ITS-90 polynomial adaptation of the
        modified-Wexler equation.

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

        """
        sumproduct = np.sum(p1 * p2)
        return (sumproduct - (np.sum(p1) * np.sum(p2)) / len(p1)) / (len(p1) - 1)

    def calc_MSE(self, y: np.ndarray) -> float:
        """
        Compute the **mean square error** (MSE) of a one-dimensional array, i.e.
        the population variance without Bessel’s correction.

        """
        return np.mean((y - np.mean(y)) ** 2)

    def convert_KtoC(self, T: float | np.ndarray) -> float | np.ndarray:
        """
        Convert **temperature** from kelvin to degrees Celsius.
        """
        return T - 273.16

    def convert_CtoK(self, T: float | np.ndarray) -> float | np.ndarray:
        """
        Convert **temperature** from degrees Celsius to kelvin.
        """
        return T + 273.16

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

        """
        return (df.last_valid_index() - df.first_valid_index()) / pd.to_timedelta(
            1, unit="D"
        )
