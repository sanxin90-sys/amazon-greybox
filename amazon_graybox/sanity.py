"""Physical sanity checks shared by all scripts.

Two kinds of check:

* hard invariants (``require``): temperature above absolute zero, relative humidity in
  [0, 100] %, vapour pressure not above saturation, net radiation within the QC limits and
  bounded at night, a positive effective heat capacity, positive physical coefficients and
  consistent skill scores. A failure raises :class:`PhysicalBoundError` (an
  ``AssertionError`` that is not removed by ``python -O``) and stops the script;
* a plausibility bound on simulated temperature (:data:`T_PLAUSIBLE`). A free run that
  leaves it (and so any run that reaches absolute zero) is not stopped, because a diverged
  run is itself a result, but every script records the fraction of such values and the
  tables report the run as diverged (:func:`diverged`). The tendency
  itself is always evaluated within this range (``graybox.GrayBoxODE.rhs``), so it stays
  finite and a run cannot overflow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import QC_LIMITS
from .thermo import es_kpa

T_PLAUSIBLE = (250.0, 340.0)  # K, the range over which graybox.rhs evaluates the tendency
DIVERGED_RMSE = 10.0  # K, free-run RMSE above which a run is reported as diverged
NIGHT_HOURS = (20, 4)  # local time, 20:00-04:00, well away from sunrise and sunset
R_NIGHT_MAX = 50.0  # W m-2, upper bound of nocturnal net radiation (net long-wave loss)
RH_TOL = 1e-6  # %, rounding tolerance of RH diagnosed from e


class PhysicalBoundError(AssertionError):
    """A physical bound or a mathematical invariant was violated."""


def require(cond, msg: str) -> None:
    if not bool(np.all(cond)):
        raise PhysicalBoundError(msg)


def _finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def night_mask(index: pd.DatetimeIndex) -> np.ndarray:
    h = (index - pd.Timedelta(minutes=30)).hour  # stamps mark the end of each 30-min interval
    return np.asarray((h >= NIGHT_HOURS[0]) | (h < NIGHT_HOURS[1]))


def check_temperature(T, name: str = "") -> None:
    """Every finite temperature [K] is above absolute zero."""
    require(_finite(T) > 0.0, f"{name}: temperature at or below absolute zero")


def implausible_fraction(T) -> float:
    """Fraction of finite simulated temperatures outside :data:`T_PLAUSIBLE`."""
    t = _finite(T)
    return float(np.mean((t < T_PLAUSIBLE[0]) | (t > T_PLAUSIBLE[1]))) if t.size else 0.0


def diverged(rmse: float, implausible: float = 0.0) -> bool:
    """Reporting rule: a run is diverged when its RMSE exceeds :data:`DIVERGED_RMSE` or when any
    simulated temperature (of any member) left :data:`T_PLAUSIBLE`, which includes reaching
    absolute zero. A diverged run is reported as such and never scored as a valid simulation.
    """
    return bool(not np.isfinite(rmse) or rmse > DIVERGED_RMSE or implausible > 0.0)


def nocturnal_radiation_flags(df: pd.DataFrame) -> dict:
    """Months with nocturnal net radiation above :data:`R_NIGHT_MAX`, as {"YYYY-MM": count}."""
    night = night_mask(df.index) & (df["R"].to_numpy() > R_NIGHT_MAX)
    months = (df.index[night] - pd.Timedelta(minutes=30)).strftime("%Y-%m")
    return {str(k): int(v) for k, v in pd.Series(months).value_counts().sort_index().items()}


def check_block(df: pd.DataFrame, name: str = "", night_radiation: bool = True) -> pd.DataFrame:
    """Hard bounds on a forcing/observation block (T [K], R [W m-2], RH [%], e [kPa]).

    ``night_radiation=False`` skips the nocturnal bound; it is used only for whole-year files,
    whose flagged months are logged instead (``scripts/01_prepare_data.py``).
    """
    T = df["T"].to_numpy(float)
    check_temperature(T, name)
    lo, hi = QC_LIMITS["T"]
    require((_finite(T) >= lo) & (_finite(T) <= hi), f"{name}: temperature outside the QC range {lo}-{hi} K")
    if "RH" in df:
        rh = _finite(df["RH"])
        require((rh >= 0.0) & (rh <= 100.0), f"{name}: relative humidity outside 0-100 %")
    if "e" in df:
        e = df["e"].to_numpy(float)
        require(_finite(e) >= 0.0, f"{name}: negative vapour pressure")
        rh_e = _finite(100.0 * e / es_kpa(T))
        require(rh_e <= 100.0 + RH_TOL, f"{name}: vapour pressure above saturation (RH = {rh_e.max():.2f} %)")
    if "R" in df:
        lo, hi = QC_LIMITS["R"]
        R = df["R"].to_numpy(float)
        require((_finite(R) >= lo) & (_finite(R) <= hi), f"{name}: net radiation outside the QC range {lo}-{hi} W m-2")
        if night_radiation:
            flags = nocturnal_radiation_flags(df)
            require(not flags, f"{name}: nocturnal net radiation above {R_NIGHT_MAX} W m-2 in {flags}")
    return df


def check_params(p: dict, delta_max: float | None = None, name: str = "") -> None:
    """Physical coefficients: alpha, gamma > 0, beta >= 0 and |delta| < delta_max."""
    for k in ("alpha", "gamma"):
        if k in p:
            require(np.asarray(p[k], float) > 0.0, f"{name}: {k} must be positive")
    if "beta" in p:
        require(np.asarray(p["beta"], float) >= 0.0, f"{name}: beta must be non-negative")
    if "delta" in p and delta_max is not None:
        require(np.abs(np.asarray(p["delta"], float)) < delta_max, f"{name}: |delta| must stay below delta_max")


def check_skill(s: dict, name: str = "") -> dict:
    """Consistency of the scores of :func:`metrics.skill` (skipped when there are too few samples)."""
    if not np.isfinite(s.get("rmse", np.nan)):
        return s
    tol = 1e-9
    require(0.0 <= s["mae"] <= s["rmse"] + tol, f"{name}: scores violate 0 <= MAE <= RMSE")
    require(abs(s["bias"]) <= s["mae"] + tol, f"{name}: scores violate |bias| <= MAE")
    require(s["r2"] <= 1.0 + tol and s["kge"] <= 1.0 + tol, f"{name}: R2 or KGE above 1")
    require(-1.0 - tol <= s["r"] <= 1.0 + tol and -tol <= s["d"] <= 1.0 + tol, f"{name}: r or d out of range")
    return s
