"""Gray-box ODE models templated on RK4 (Malani et al., 2023), in PyTorch.

A model integrates dT/dt = F(T, forcing) with

    white box  W in {"none", "dry", "moist_open", "moist_closed"}
    correction C in {"none", "add", "mult", "hyb"}

    add   F = f_W + s * N(z)                               (Malani Eq. 2b)
    mult  F = f_W * exp(b * tanh N(z))                      (Malani Eq. 2c)
    hyb   F = [sum_k p_k exp(b * tanh N_k(z)) phi_k] / C_eff + s * N_0(z)
          i.e. state-dependent extinction coefficients (function composition,
          Malani Eq. 2d) combined with an additive residual (Malani Eq. 3)
    W = "none" with C = "add" is the black-box ODE-net (Malani Eq. 1).

Every tensor carries a leading ensemble dimension M: the M members (each with
its own network weights and physical parameters) are trained in parallel.
Time is in hours, temperature in K, tendencies in K h-1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .physics import RHO_C, SEC_PER_H, SIGMA, delta_max
from .sanity import require

TORCH_DTYPE = torch.float64
_A, _B, _C = 0.61094, 17.625, 243.04  # Magnus coefficients (see thermo.py)

WHITE_BOXES = ("none", "dry", "moist_open", "moist_closed")
CORRECTIONS = ("none", "add", "mult", "hyb")
INPUT_SETS = {"TR": ("T", "R"), "TRH": ("T", "R", "RH"), "TRHe": ("T", "R", "RH", "e")}


def es_t(T):
    tc = T - 273.15
    return _A * torch.exp(_B * tc / (tc + _C))


def dln_es_t(T):
    tc = T - 273.15
    return _B * _C / (tc + _C) ** 2


class EnsembleMLP(nn.Module):
    """M independent MLPs evaluated with batched matmuls: (M, B, d_in) -> (M, B, d_out)."""

    def __init__(self, M: int, d_in: int, d_out: int, hidden=(32, 16), out_scale: float = 0.1, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        dims = (d_in, *hidden, d_out)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            w = torch.randn(M, a, b, generator=g, dtype=TORCH_DTYPE) * math.sqrt(1.0 / a)
            if i == len(dims) - 2:
                # Small output layer so that training starts from the white box
                # (Malani et al. 2023 use the same trick against bad local minima).
                w = w * out_scale
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(M, 1, b, dtype=TORCH_DTYPE)))

    def forward(self, x):
        n = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.baddbmm(b, x, w)
            if i < n - 1:
                x = torch.tanh(x)
        return x

    def n_params_per_member(self) -> int:
        return sum(p[0].numel() for p in self.weights) + sum(p[0].numel() for p in self.biases)


@dataclass
class Scalers:
    """Fixed affine scaling of the network inputs (fitted on the training block)."""

    mean: dict = field(default_factory=dict)
    std: dict = field(default_factory=dict)
    out: float = 1.0  # K h-1, scale of additive corrections

    @classmethod
    def fit(cls, arrays: dict, dTdt_std: float) -> "Scalers":
        mean = {k: float(np.nanmean(v)) for k, v in arrays.items()}
        std = {k: float(np.nanstd(v)) or 1.0 for k, v in arrays.items()}
        return cls(mean, std, float(dTdt_std))


class GrayBoxODE(nn.Module):
    def __init__(
        self,
        white: str,
        correction: str,
        M: int = 10,
        inputs: str = "TRHe",
        hidden=(32, 16),
        p0: dict | None = None,
        scalers: Scalers | None = None,
        mult_bound: float = 1.5,
        train_phys: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        assert white in WHITE_BOXES and correction in CORRECTIONS
        if white == "none" and correction not in ("add",):
            raise ValueError("the black box needs correction='add'")
        self.white, self.correction, self.M = white, correction, M
        self.inputs = INPUT_SETS[inputs]
        self.scalers = scalers or Scalers()
        self.mult_bound = mult_bound
        p0 = dict(alpha=2.7e-3, gamma=6.5e-4, beta=1e-3, delta=0.0) | (p0 or {})
        self.p0 = p0
        self.delta_max = delta_max()
        # Physical parameters, one per member, in unconstrained coordinates.
        ones = torch.ones(M, 1, dtype=TORCH_DTYPE)
        self.log_alpha = nn.Parameter(ones * math.log(p0["alpha"]), requires_grad=train_phys)
        self.log_gamma = nn.Parameter(ones * math.log(p0["gamma"]), requires_grad=train_phys)
        moist = white.startswith("moist")
        self.log_beta = nn.Parameter(ones * math.log(max(p0["beta"], 1e-6)), requires_grad=train_phys and moist)
        if white == "moist_closed":
            d = np.clip(p0["delta"] / self.delta_max, -0.99, 0.99)
            self.raw_delta = nn.Parameter(ones * float(np.arctanh(d)), requires_grad=train_phys)
        else:  # open model: delta > 0 as in Maionchi et al. (2025), log-parameterised
            self.raw_delta = nn.Parameter(ones * math.log(max(p0["delta"], 1e-3)), requires_grad=train_phys and moist)
        n_terms = 3 if moist else 2
        d_out = {"none": 0, "add": 1, "mult": 1, "hyb": n_terms + 1}[correction]
        self.net = EnsembleMLP(M, len(self.inputs), d_out, hidden, seed=seed) if d_out else None

    # ------------------------------------------------------------------ params
    def phys_params(self) -> dict:
        p = {"alpha": self.log_alpha.exp(), "gamma": self.log_gamma.exp()}
        if self.white.startswith("moist"):
            p["beta"] = self.log_beta.exp()
            p["delta"] = self.delta_max * torch.tanh(self.raw_delta) if self.white == "moist_closed" else self.raw_delta.exp()
        return p

    def phys_params_numpy(self) -> dict:
        return {k: v.detach().cpu().numpy().ravel() for k, v in self.phys_params().items()}

    # --------------------------------------------------------------------- rhs
    def _features(self, T, F, RH):
        cols = []
        for name in self.inputs:
            v = {"T": T, "R": F["R"], "RH": RH, "e": F["e"]}[name]
            v = v.expand_as(T) if v.shape != T.shape else v
            cols.append((v - self.scalers.mean[name]) / self.scalers.std[name])
        return torch.stack(cols, dim=-1)

    def rhs(self, T, F, return_parts: bool = False):
        """Tendency [K h-1]. ``T``: (M, B); forcing tensors in ``F``: (1, B) or (M, B)."""
        T = T.clamp(250.0, 340.0)
        es = es_t(T)
        rh_model = (100.0 * F["e"] / es).clamp(0.0, 100.0)
        terms, c_eff = [], RHO_C
        if self.white != "none":
            p = self.phys_params()
            terms = [p["alpha"] * F["R"], -p["gamma"] * SIGMA * T**4]
            if self.white == "moist_open":
                rh = F["RH_obs"]
                terms.append(p["beta"] * rh)
                d_term = -p["delta"] * F["dRH_obs"] / SEC_PER_H
            elif self.white == "moist_closed":
                rh = rh_model
                terms.append(p["beta"] * rh)
                d_term = -p["delta"] * 100.0 * F["de"] / (SEC_PER_H * es)
                c_eff = RHO_C - p["delta"] * rh * dln_es_t(T)
                require((c_eff > 0).all().item(), "effective heat capacity must stay positive")
            else:
                d_term = 0.0
        nn_rh = F["RH_obs"] if self.white == "moist_open" else rh_model
        out = self.net(self._features(T, F, nn_rh)) if self.net is not None else None

        if self.white == "none":
            f = self.scalers.out * out[..., 0]
            return (f, {}) if return_parts else f
        if self.correction in ("none", "add", "mult"):
            phys = SEC_PER_H * (sum(terms) + d_term) / c_eff
            if self.correction == "none":
                f = phys
            elif self.correction == "add":
                f = phys + self.scalers.out * out[..., 0]
            else:
                f = phys * torch.exp(self.mult_bound * torch.tanh(out[..., 0]))
        else:  # hyb
            mods = torch.exp(self.mult_bound * torch.tanh(out[..., :-1]))
            mod_terms = sum(mods[..., i] * t for i, t in enumerate(terms))
            f = SEC_PER_H * (mod_terms + d_term) / c_eff + self.scalers.out * out[..., -1]
        if return_parts:
            return f, {"out": out, "c_eff": c_eff}
        return f

    def correction_penalty(self, T, F):
        """Mean squared size of the corrections at observed states (keeps the physics dominant)."""
        if self.net is None or self.white == "none":
            return torch.zeros((), dtype=TORCH_DTYPE)
        rh = F["RH_obs"] if self.white == "moist_open" else (100.0 * F["e"] / es_t(T)).clamp(0, 100)
        out = self.net(self._features(T, F, rh))
        if self.correction == "mult":
            return (self.mult_bound * torch.tanh(out[..., 0])).pow(2).mean()
        if self.correction == "add":
            return out[..., 0].pow(2).mean()
        return (self.mult_bound * torch.tanh(out[..., :-1])).pow(2).mean() + out[..., -1].pow(2).mean()

    # ----------------------------------------------------------------- rollout
    def rollout(self, T0, Ffine: dict, n_steps: int, n_sub: int = 2, dt_h: float = 0.5):
        """RK4 integration over ``n_steps`` data intervals with ``n_sub`` sub-steps each.

        ``Ffine[k]`` has shape (B, 2 * n_sub * n_steps + 1): forcing sampled every
        half sub-step, as needed by RK4. Returns (M, B, n_steps + 1).
        """
        h = dt_h / n_sub
        T = T0
        traj = [T0]
        for k in range(n_steps):
            for j in range(n_sub):
                i0 = 2 * (k * n_sub + j)
                F0 = {n: v[:, i0] for n, v in Ffine.items()}
                Fm = {n: v[:, i0 + 1] for n, v in Ffine.items()}
                F1 = {n: v[:, i0 + 2] for n, v in Ffine.items()}
                k1 = self.rhs(T, F0)
                k2 = self.rhs(T + 0.5 * h * k1, Fm)
                k3 = self.rhs(T + 0.5 * h * k2, Fm)
                k4 = self.rhs(T + h * k3, F1)
                T = T + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            traj.append(T)
        return torch.stack(traj, dim=-1)


def refine_forcing(F: dict, n_sub: int) -> dict:
    """Linearly interpolate forcing arrays (B, N+1) onto the RK4 half-sub-step grid."""
    out = {}
    for k, v in F.items():
        v = torch.as_tensor(v, dtype=TORCH_DTYPE)
        if v.ndim == 1:
            v = v[None]
        B, n1 = v.shape
        x = torch.linspace(0, n1 - 1, 2 * n_sub * (n1 - 1) + 1, dtype=TORCH_DTYPE)
        i0 = x.floor().long().clamp(max=n1 - 2)
        w = (x - i0)[None]
        out[k] = v[:, i0] * (1 - w) + v[:, i0 + 1] * w
    return out
