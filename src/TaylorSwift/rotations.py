"""
rotations.py — Coordinate rotation for sonic-anemometer wind vectors.

One implementation of the standard double rotation (Kaimal & Finnigan 1994)
serves both stacks:

* :func:`coord_rotation` computes the rotation angles and
  :func:`rotate_velocities` applies them — the primitive form used by the
  CalcFlux pipelines, where the angles are also needed to rotate
  covariances (:func:`rotate_covariances`).
* :func:`rotate_wind` is the convenience wrapper used by the spectral
  stack: it derives the angles and applies them in one call.
"""

from __future__ import annotations

import numpy as np


def determine_wind_dir(
    uxavg: float | None, uyavg: float | None, sonic_dir: float, path_dist_u: float
) -> tuple[float, float]:
    if uxavg is None or uyavg is None:
        raise ValueError("uxavg and uyavg are required")
    wind_dir = np.degrees(np.arctan2(uyavg, uxavg))
    wind_compass = -wind_dir + sonic_dir
    if wind_compass < 0:
        wind_compass += 360.0
    elif wind_compass > 360.0:
        wind_compass -= 360.0
    pathlen = path_dist_u * np.abs(np.sin(np.radians(wind_compass)))
    return pathlen, wind_compass


# ---------------------------------------------------------------------------
# Double rotation (yaw + pitch) — Kaimal & Finnigan 1994
# ---------------------------------------------------------------------------
def coord_rotation(
    Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """
    Rotation angles for the double rotation (yaw + pitch).

    Returns
    -------
    cosv, sinv : float
        Cosine/sine of the yaw angle (first rotation, aligns x with the
        mean horizontal wind).
    sinTheta, cosTheta : float
        Sine/cosine of the pitch angle (second rotation, zeroes mean w).
    Uxy : float
        Mean horizontal wind speed [m/s].
    Uxyz : float
        Mean total wind speed [m/s].
    """
    xmean = float(np.nanmean(Ux))
    ymean = float(np.nanmean(Uy))
    zmean = float(np.nanmean(Uz))
    Uxy = np.sqrt(xmean**2 + ymean**2)
    Uxyz = np.sqrt(xmean**2 + ymean**2 + zmean**2)
    if Uxy < 1e-9 or Uxyz < 1e-9:
        return 1.0, 0.0, 0.0, 1.0, Uxy, Uxyz
    cosv = xmean / Uxy
    sinv = ymean / Uxy
    sinTheta = zmean / Uxyz
    cosTheta = Uxy / Uxyz
    return cosv, sinv, sinTheta, cosTheta, Uxy, Uxyz


def rotate_velocities(
    Ux: np.ndarray,
    Uy: np.ndarray,
    Uz: np.ndarray,
    cosv: float,
    sinv: float,
    sinTheta: float,
    cosTheta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the double rotation defined by :func:`coord_rotation` angles."""
    Uxr = Ux * cosTheta * cosv + Uy * cosTheta * sinv + Uz * sinTheta
    Uyr = Uy * cosv - Ux * sinv
    Uzr = Uz * cosTheta - Ux * sinTheta * cosv - Uy * sinTheta * sinv
    return Uxr, Uyr, Uzr


def rotate_wind(
    u_raw: np.ndarray, v_raw: np.ndarray, w_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Apply double rotation so that mean(v_rot) = 0 and mean(w_rot) = 0.

    This aligns the x-axis with the mean horizontal wind vector and tilts
    the coordinate system so the mean vertical velocity vanishes — the
    standard approach in eddy-covariance processing.  Thin wrapper around
    :func:`coord_rotation` + :func:`rotate_velocities`, so both processing
    stacks share one implementation of the rotation math.

    Parameters
    ----------
    u_raw, v_raw, w_raw : array-like
        Raw wind components from the sonic anemometer.

    Returns
    -------
    u_rot, v_rot, w_rot : np.ndarray
        Rotated wind components.
    wind_dir : float
        Horizontal wind direction [degrees from north] — NB: only meaningful
        if the sonic x-axis orientation is known; here it gives the angle
        of the mean wind relative to the sonic x-axis.
    """
    u = np.asarray(u_raw, dtype=np.float64)
    v = np.asarray(v_raw, dtype=np.float64)
    w = np.asarray(w_raw, dtype=np.float64)

    cosv, sinv, sinTheta, cosTheta, _Uxy, _Uxyz = coord_rotation(u, v, w)
    u_rot, v_rot, w_rot = rotate_velocities(u, v, w, cosv, sinv, sinTheta, cosTheta)

    wind_dir = float(np.degrees(np.arctan2(sinv, cosv))) % 360.0
    return u_rot, v_rot, w_rot, wind_dir


def rotate_covariances(
    covar: dict[str, float],
    errvals: dict[str, float],
    cosv: float,
    sinv: float,
    sinTheta: float,
    cosTheta: float,
    scalar_key: str = "Ts",
) -> dict[str, float]:
    """Rotate scalar and momentum covariances into the streamline frame."""
    cov = dict(covar)

    Ux_s = cov.get(f"Ux-{scalar_key}", 0.0)
    Uy_s = cov.get(f"Uy-{scalar_key}", 0.0)
    Uz_s = cov.get(f"Uz-{scalar_key}", 0.0)
    cov[f"Uz-{scalar_key}"] = (
        Uz_s * cosTheta - Ux_s * sinTheta * cosv - Uy_s * sinTheta * sinv
    )

    for key in ("pV", "Sd"):
        Ux_k = cov.get(f"Ux-{key}", 0.0)
        Uy_k = cov.get(f"Uy-{key}", 0.0)
        Uz_k = cov.get(f"Uz-{key}", 0.0)
        cov[f"Uz-{key}"] = (
            Uz_k * cosTheta - Ux_k * sinTheta * cosv - Uy_k * sinTheta * sinv
        )

    # Momentum covariances (Kaimal & Finnigan 1994, eq. 6.36)
    Ux_Uz = cov.get("Ux-Uz", 0.0)
    Uy_Uz = cov.get("Uy-Uz", 0.0)
    Ux_Uy = cov.get("Ux-Uy", 0.0)
    err_Ux = errvals.get("Ux", 0.0)
    err_Uy = errvals.get("Uy", 0.0)
    err_Uz = errvals.get("Uz", 0.0)

    Ux_Uz_rot = (
        Ux_Uz * cosv * (cosTheta**2 - sinTheta**2)
        - 2.0 * Ux_Uy * sinTheta * cosTheta * sinv * cosv
        + Uy_Uz * sinv * (cosTheta**2 - sinTheta**2)
        - err_Ux * sinTheta * cosTheta * cosv**2
        - err_Uy * sinTheta * cosTheta * sinv**2
        + err_Uz * sinTheta * cosTheta
    )
    Uy_Uz_rot = (
        Uy_Uz * cosTheta * cosv
        - Ux_Uz * cosTheta * sinv
        - Ux_Uy * sinTheta * (cosv**2 - sinv**2)
        + err_Ux * sinTheta * sinv * cosv
        - err_Uy * sinTheta * sinv * cosv
    )
    cov["Ux-Uz"] = Ux_Uz_rot
    cov["Uy-Uz"] = Uy_Uz_rot
    cov["Uxy-Uz"] = np.sqrt(Ux_Uz_rot**2 + Uy_Uz_rot**2)
    return cov
