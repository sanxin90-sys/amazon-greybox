"""Moist thermodynamics helpers (Alduchov & Eskridge, 1996, Magnus form over water)."""
from __future__ import annotations

import numpy as np

KELVIN = 273.15
_A, _B, _C = 0.61094, 17.625, 243.04  # kPa, -, degC


def es_kpa(T_k):
    """Saturation vapour pressure [kPa] at temperature ``T_k`` [K]."""
    tc = T_k - KELVIN
    return _A * np.exp(_B * tc / (tc + _C))


def dln_es_dT(T_k):
    """d ln(e_s) / dT [K-1] (Clausius-Clapeyron sensitivity, ~0.06 K-1 at 25 degC)."""
    tc = T_k - KELVIN
    return _B * _C / (tc + _C) ** 2


def vapour_pressure(T_k, rh_pct):
    """Actual vapour pressure [kPa] from temperature [K] and relative humidity [%]."""
    return rh_pct / 100.0 * es_kpa(T_k)


def relative_humidity(T_k, e_kpa):
    """Relative humidity [%] diagnosed from temperature [K] and vapour pressure [kPa]."""
    return 100.0 * e_kpa / es_kpa(T_k)
