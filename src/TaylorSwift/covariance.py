from __future__ import annotations
import numpy as np


def calc_cov(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return float("nan")
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 2:
        return float("nan")
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (n - 1))


def calc_MSE(y) -> float:
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if not np.any(mask):
        return float("nan")
    y = y[mask]
    return float(np.mean((y - y.mean()) ** 2))


def calc_max_covariance(x, y, lag: int = 10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    candidates = []
    for k in range(-lag, lag + 1):
        if k < 0:
            xv = x[:k]
            yv = y[-k:]
        elif k > 0:
            xv = x[k:]
            yv = y[:-k]
        else:
            xv = x
            yv = y
        if len(xv) == 0 or len(yv) == 0:
            continue
        candidates.append((k, calc_cov(xv, yv)))
    if not candidates:
        return []
    best = max(candidates, key=lambda item: abs(item[1]))
    return [best]


def calc_covar(Ux, Uy, Uz, Ts, Q, pV) -> dict[str, float]:
    data = {"Ux": Ux, "Uy": Uy, "Uz": Uz, "Ts": Ts, "Q": Q, "pV": pV}
    out = {}
    for k1, v1 in data.items():
        for k2, v2 in data.items():
            out[f"{k1}-{k2}"] = calc_cov(v1, v2)
    return out


def build_covariance_dict(
    velocities: dict[str, np.ndarray], variables: dict[str, np.ndarray], lag: int = 10
) -> dict[str, float]:
    out = {}
    for ik, iv in velocities.items():
        for jk, jv in variables.items():
            result = calc_max_covariance(iv, jv, lag=lag)
            out[f"{ik}-{jk}"] = result[0][1] if result else calc_cov(iv, jv)
    return out
