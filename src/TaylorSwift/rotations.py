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
# Wind coordinate rotation (double rotation)
# ---------------------------------------------------------------------------
def rotate_wind(u_raw: np.ndarray, v_raw: np.ndarray, w_raw: np.ndarray):
    """
    Apply double rotation so that mean(v_rot) = 0 and mean(w_rot) = 0.

    This aligns the x-axis with the mean horizontal wind vector and tilts
    the coordinate system so the mean vertical velocity vanishes — the
    standard approach in eddy-covariance processing.

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

    u_bar = np.nanmean(u)
    v_bar = np.nanmean(v)

    # First rotation: align u with horizontal wind vector
    alpha = np.arctan2(v_bar, u_bar)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)

    u1 = u * cos_a + v * sin_a
    v1 = -u * sin_a + v * cos_a
    w1 = w.copy()

    # Second rotation: tilt to make mean(w) = 0
    u1_bar = np.nanmean(u1)
    w1_bar = np.nanmean(w1)
    beta = np.arctan2(w1_bar, u1_bar)
    cos_b, sin_b = np.cos(beta), np.sin(beta)

    u2 = u1 * cos_b + w1 * sin_b
    v2 = v1.copy()
    w2 = -u1 * sin_b + w1 * cos_b

    wind_dir = np.degrees(alpha) % 360.0

    return u2, v2, w2, wind_dir


def coord_rotation(Ux, Uy, Uz):
    """Double rotation (yaw + pitch) — Kaimal & Finnigan 1994."""
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


def rotate_velocities(Ux, Uy, Uz, cosv, sinv, sinTheta, cosTheta):
    Uxr = Ux * cosTheta * cosv + Uy * cosTheta * sinv + Uz * sinTheta
    Uyr = Uy * cosv - Ux * sinv
    Uzr = Uz * cosTheta - Ux * sinTheta * cosv - Uy * sinTheta * sinv
    return Uxr, Uyr, Uzr


def rotate_covariances(
    covar, errvals, cosv, sinv, sinTheta, cosTheta, scalar_key: str = "Ts"
):
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
