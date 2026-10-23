"""Skill scores, diurnal diagnostics and ensemble calibration metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .sanity import check_skill


def _valid(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    m = np.isfinite(obs) & np.isfinite(pred)
    return obs[m], pred[m]


def skill(obs, pred) -> dict:
    o, p = _valid(obs, pred)
    if len(o) < 10:
        return {k: np.nan for k in ("n", "rmse", "mae", "bias", "r2", "r", "kge", "d")}
    err = p - o
    r = np.corrcoef(o, p)[0, 1]
    alpha = p.std() / o.std()
    beta = p.mean() / o.mean()
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    d = 1 - np.sum(err**2) / np.sum((np.abs(p - o.mean()) + np.abs(o - o.mean())) ** 2)
    return check_skill({
        "n": len(o),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(err.mean()),
        "r2": float(1 - np.sum(err**2) / np.sum((o - o.mean()) ** 2)),  # Nash-Sutcliffe form
        "r": float(r),
        "kge": float(kge),
        "d": float(d),  # Willmott index of agreement
    })


def daily_extremes(index: pd.DatetimeIndex, obs, pred) -> dict:
    """Errors of daily maximum/minimum temperature and of the diurnal range (complete days only)."""
    df = pd.DataFrame({"o": obs, "p": pred}, index=index)
    day = (df.index - pd.Timedelta(minutes=30)).normalize()
    g = df.groupby(day)
    full = g.count().min(axis=1) >= 46
    mx, mn = g.max()[full], g.min()[full]
    return {
        "tmax_rmse": float(np.sqrt(np.mean((mx.p - mx.o) ** 2))),
        "tmax_bias": float(np.mean(mx.p - mx.o)),
        "tmin_rmse": float(np.sqrt(np.mean((mn.p - mn.o) ** 2))),
        "tmin_bias": float(np.mean(mn.p - mn.o)),
        "dtr_bias": float(np.mean((mx.p - mn.p) - (mx.o - mn.o))),
        "n_days": int(full.sum()),
    }


def nocturnal_cooling(index: pd.DatetimeIndex, T, start_h: int = 20, end_h: int = 23) -> pd.Series:
    """T(end_h) - T(start_h) for each night (the de Paulo et al. 2023 statistic)."""
    s = pd.Series(np.asarray(T, float), index=index)
    a = s[(s.index.hour == start_h) & (s.index.minute == 0)]
    b = s[(s.index.hour == end_h) & (s.index.minute == 0)]
    a.index, b.index = a.index.normalize(), b.index.normalize()
    return (b - a).dropna()


def gaussian_interval_coverage(obs, mu, sigma, level: float = 0.9) -> float:
    o, m = np.asarray(obs), np.asarray(mu)
    s = np.asarray(sigma)
    ok = np.isfinite(o) & np.isfinite(m) & np.isfinite(s)
    z = stats.norm.ppf(0.5 + level / 2)
    return float(np.mean(np.abs(o[ok] - m[ok]) <= z * s[ok]))


def crps_gaussian(obs, mu, sigma) -> float:
    o, m, s = (np.asarray(x, float) for x in (obs, mu, sigma))
    ok = np.isfinite(o) & np.isfinite(m) & np.isfinite(s) & (s > 0)
    z = (o[ok] - m[ok]) / s[ok]
    c = s[ok] * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(c))


def ensemble_predictive(members: np.ndarray, sigma_res: float = 0.0):
    """Gaussian predictive distribution from an ensemble (M, N) plus a residual term."""
    mu = np.nanmean(members, axis=0)
    sd = np.sqrt(np.nanvar(members, axis=0) + sigma_res**2)
    return mu, sd
