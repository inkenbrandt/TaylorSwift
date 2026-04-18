import numpy as np


def determine_wind_dir(uxavg: float | None, uyavg: float | None, sonic_dir: float, path_dist_u: float) -> tuple[float, float]:
    if uxavg is None or uyavg is None:
        raise ValueError('uxavg and uyavg are required')
    wind_dir = np.degrees(np.arctan2(uyavg, uxavg))
    wind_compass = -wind_dir + sonic_dir
    if wind_compass < 0:
        wind_compass += 360.0
    elif wind_compass > 360.0:
        wind_compass -= 360.0
    pathlen = path_dist_u * np.abs(np.sin(np.radians(wind_compass)))
    return pathlen, wind_compass
