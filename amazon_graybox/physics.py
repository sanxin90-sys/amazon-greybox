"""White-box models: dry (de Paulo et al., 2023) and moist (Maionchi et al., 2025).

Time is expressed in hours throughout; tendencies are returned in K h-1.

Dry model            rho c dT/dt = alpha R - gamma sigma T^4
Moist model (open)   rho c dT/dt = alpha R - gamma sigma T^4 + P (beta RH - delta dRH/dt)
    with RH and dRH/dt taken from observations, as in Maionchi et al. (2025).
Moist model (closed) same equation, but RH = 100 e / e_s(T) is diagnosed from the
    model temperature and the observed vapour pressure e, so that
    [rho c - P delta RH dln(e_s)/dT] dT/dt = alpha R - gamma sigma T^4
                                           + P (beta RH - 100 delta de/dt / e_s(T)).
    The bracket is an effective heat capacity; the model is well posed only while it
    stays positive, which bounds delta (see :func:`delta_max`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .thermo import dln_es_dT, es_kpa

SIGMA = 5.670374419e-8  # W m-2 K-4
RHO_AIR = 1.184  # kg m-3 (de Paulo et al., 2023)
CP_AIR = 1012.0  # J kg-1 K-1
RHO_C = RHO_AIR * CP_AIR  # J m-3 K-1
SEC_PER_H = 3600.0


@dataclass
class PhysParams:
    """Physical parameters in the units of de Paulo et al. (2023) / Maionchi et al. (2025).

    alpha, gamma [m-1]; beta [W m-3 %-1]; delta [J m-3 %-1].
    """

    alpha: float = 2.7e-3
    gamma: float = 6.5e-4
    beta: float = 0.0
    delta: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def silva_to_si(alpha_star: float, gamma_star: float, time_unit_s: float = 1e6) -> PhysParams:
    """Convert Silva et al. (2026) effective parameters to alpha, gamma [m-1].

    Their code integrates dT/dt = alpha* R - gamma* T^4 with time in units of
    ``time_unit_s`` seconds (minutes * 60e-6, i.e. 1e6 s).
    """
    return PhysParams(alpha=alpha_star * RHO_C / time_unit_s, gamma=gamma_star * RHO_C / (SIGMA * time_unit_s))


def si_to_silva(p: PhysParams, time_unit_s: float = 1e6) -> tuple[float, float]:
    return p.alpha * time_unit_s / RHO_C, p.gamma * SIGMA * time_unit_s / RHO_C


def interaction_probability(rh):
    """Switch P(RH) of Maionchi et al. (2025), Eq. (11)."""
    rh = np.clip(rh, 0.0, 99.999)
    return 1.0 / (1.0 + np.exp(-(rh - 80.0) / (100.0 - rh)))


def delta_max(rh_max: float = 100.0, T_min_k: float = 288.15, margin: float = 0.9) -> float:
    """Largest delta keeping the closed moist model's heat capacity positive."""
    return margin * RHO_C / (rh_max * dln_es_dT(T_min_k))


def dry_rhs(T, R, p: PhysParams):
    """dT/dt [K h-1] of the dry model."""
    return SEC_PER_H * (p.alpha * R - p.gamma * SIGMA * T**4) / RHO_C


def moist_open_rhs(T, R, rh_obs, drh_obs_dt_h, p: PhysParams, use_p: bool = False):
    """Maionchi et al. (2025) model forced by observed RH and dRH/dt [% h-1]."""
    P = interaction_probability(rh_obs) if use_p else 1.0
    q = p.alpha * R - p.gamma * SIGMA * T**4 + P * (p.beta * rh_obs - p.delta * drh_obs_dt_h / SEC_PER_H)
    return SEC_PER_H * q / RHO_C


def moist_closed_rhs(T, R, e_kpa, de_dt_h, p: PhysParams, use_p: bool = False):
    """Thermodynamically consistent moist model (RH diagnosed from the model state)."""
    es = es_kpa(T)
    rh = np.clip(100.0 * e_kpa / es, 0.0, 100.0)
    P = interaction_probability(rh) if use_p else 1.0
    c_eff = RHO_C - P * p.delta * rh * dln_es_dT(T)
    q = p.alpha * R - p.gamma * SIGMA * T**4 + P * (p.beta * rh - p.delta * 100.0 * de_dt_h / (SEC_PER_H * es))
    return SEC_PER_H * q / c_eff


def nocturnal_dry_solution(T0, t_h, gamma: float = 6.5e-4):
    """Analytical night-time (R = 0) solution of the dry model, de Paulo et al. (2023) Eq. (3)."""
    t_s = np.asarray(t_h) * SEC_PER_H
    return T0 / (1.0 + 3.0 * gamma * SIGMA * T0**3 * t_s / RHO_C) ** (1.0 / 3.0)
