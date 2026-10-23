"""Trajectory-based training, simulation and forecasting of gray-box ensembles."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .graybox import TORCH_DTYPE, GrayBoxODE, refine_forcing

FORCING_COLS = {"R": "R", "e": "e", "de": "de", "RH_obs": "RH", "dRH_obs": "dRH"}


@dataclass
class TrainConfig:
    window_h: float = 48.0  # length of training trajectories
    stride_h: float = 24.0  # spacing of window starts
    n_sub: int = 1  # RK4 sub-steps per 30-min data interval (evaluation also checks 4)
    epochs: int = 600
    lr_net: float = 3e-3
    lr_phys: float = 1e-2
    weight_decay: float = 1e-5
    lam_corr: float = 1e-3  # penalty on the size of the corrections
    patience: int = 80  # epochs without validation improvement (per member)
    bootstrap: bool = True
    seed: int = 0
    verbose: int = 100


def block_arrays(df: pd.DataFrame) -> dict:
    return {k: df[c].to_numpy(dtype=float) for k, c in FORCING_COLS.items()} | {"T": df["T"].to_numpy(dtype=float)}


def make_windows(df: pd.DataFrame, window_h: float, stride_h: float, dt_h: float = 0.5):
    """Cut a block into (possibly overlapping) windows whose data are all finite."""
    arr = block_arrays(df)
    L = int(round(window_h / dt_h))
    s = int(round(stride_h / dt_h))
    ok = np.all([np.isfinite(v) for v in arr.values()], axis=0)
    starts = [i for i in range(0, len(df) - L, s) if ok[i : i + L + 1].all()]
    if not starts:
        raise ValueError("no complete window in block")
    W = {k: np.stack([v[i : i + L + 1] for i in starts]) for k, v in arr.items()}
    return W, np.array(starts), L


def _forcing_tensors(W: dict, n_sub: int) -> dict:
    return refine_forcing({k: W[k] for k in FORCING_COLS}, n_sub)


def window_loss(model: GrayBoxODE, W: dict, Ffine: dict, L: int, n_sub: int, weights=None):
    T0 = torch.as_tensor(W["T"][:, 0], dtype=TORCH_DTYPE)[None].expand(model.M, -1)
    pred = model.rollout(T0, Ffine, L, n_sub)
    obs = torch.as_tensor(W["T"], dtype=TORCH_DTYPE)[None]
    se = (pred - obs).pow(2).mean(-1)  # (M, B)
    if weights is None:
        return se.mean(-1), pred
    return (se * weights).sum(-1) / weights.sum(-1), pred


def train(model: GrayBoxODE, df_train: pd.DataFrame, df_val: pd.DataFrame, cfg: TrainConfig):
    """Train all ensemble members jointly; early stopping is applied per member."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    Wtr, _, L = make_windows(df_train, cfg.window_h, cfg.stride_h)
    Wva, _, Lv = make_windows(df_val, cfg.window_h, cfg.window_h)
    Ftr, Fva = _forcing_tensors(Wtr, cfg.n_sub), _forcing_tensors(Wva, cfg.n_sub)
    B = Wtr["T"].shape[0]
    if cfg.bootstrap:
        w = np.stack([np.bincount(rng.integers(0, B, B), minlength=B) for _ in range(model.M)]).astype(float)
    else:
        w = np.ones((model.M, B))
    weights = torch.as_tensor(w, dtype=TORCH_DTYPE)
    # observed states for the correction penalty
    arr = {k: v.copy() for k, v in block_arrays(df_train.dropna(subset=["T", "R", "RH", "e", "de", "dRH"])).items()}
    Fpen = {k: torch.as_tensor(arr[k], dtype=TORCH_DTYPE)[None] for k in FORCING_COLS}
    Tpen = torch.as_tensor(arr["T"], dtype=TORCH_DTYPE)[None].expand(model.M, -1)

    phys = [p for n, p in model.named_parameters() if not n.startswith("net.") and p.requires_grad]
    net = [p for n, p in model.named_parameters() if n.startswith("net.")]
    groups = []
    if net:
        groups.append({"params": net, "lr": cfg.lr_net, "weight_decay": cfg.weight_decay})
    if phys:
        groups.append({"params": phys, "lr": cfg.lr_phys, "weight_decay": 0.0})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=0.05 * cfg.lr_net)

    best = torch.full((model.M,), math.inf, dtype=TORCH_DTYPE)
    best_state = copy.deepcopy(model.state_dict())
    since = torch.zeros(model.M, dtype=torch.long)
    hist = []
    t0 = time.time()
    for ep in range(cfg.epochs):
        opt.zero_grad()
        loss_m, _ = window_loss(model, Wtr, Ftr, L, cfg.n_sub, weights)
        pen = model.correction_penalty(Tpen, Fpen) if cfg.lam_corr > 0 else 0.0
        loss = loss_m.sum() + cfg.lam_corr * model.M * pen
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {ep}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            val_m, _ = window_loss(model, Wva, Fva, Lv, cfg.n_sub)
            val_m = torch.nan_to_num(val_m, nan=math.inf)
            improved = val_m < best
            if improved.any():
                state = model.state_dict()
                for k, v in state.items():
                    if v.ndim and v.shape[0] == model.M:
                        best_state[k][improved] = v[improved].clone()
                best[improved] = val_m[improved]
            since = torch.where(improved, torch.zeros_like(since), since + 1)
        hist.append((ep, float(loss_m.detach().mean().sqrt()), float(val_m.mean().sqrt())))
        if cfg.verbose and ep % cfg.verbose == 0:
            print(f"  ep {ep:4d} train RMSE {hist[-1][1]:.3f} K  val RMSE {hist[-1][2]:.3f} K  ({time.time() - t0:.0f}s)", flush=True)
        if (since > cfg.patience).all():
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(hist, columns=["epoch", "train_rmse", "val_rmse"]), best.sqrt().numpy()


@torch.no_grad()
def simulate(model: GrayBoxODE, df: pd.DataFrame, n_sub: int = 2, min_len: int = 48) -> np.ndarray:
    """Free-run simulation over a block, re-initialised from observations after each gap.

    Returns (M, N) with NaN where data (or forcing) are missing.
    """
    arr = block_arrays(df)
    N = len(df)
    ok = np.all([np.isfinite(v) for v in arr.values()], axis=0)
    out = np.full((model.M, N), np.nan)
    i = 0
    while i < N:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < N and ok[j]:
            j += 1
        if j - i >= min_len:
            seg = {k: v[i:j][None] for k, v in arr.items()}
            F = refine_forcing({k: seg[k] for k in FORCING_COLS}, n_sub)
            T0 = torch.full((model.M, 1), seg["T"][0, 0], dtype=TORCH_DTYPE)
            out[:, i:j] = model.rollout(T0, F, j - i - 1, n_sub)[:, 0].numpy()
        i = j
    return out


@torch.no_grad()
def forecast(model: GrayBoxODE, df: pd.DataFrame, horizon_h: float = 24.0, init_hour: int = 0, n_sub: int = 2):
    """Forecasts of ``horizon_h`` hours started every day at ``init_hour`` (local time).

    Returns an (M, N) array with each sample predicted by the forecast that covers it.
    """
    L = int(round(horizon_h / 0.5))
    arr = block_arrays(df)
    ok = np.all([np.isfinite(v) for v in arr.values()], axis=0)
    idx = np.where((df.index.hour == init_hour) & (df.index.minute == 30))[0]
    starts = [i for i in idx if i + L < len(df) and ok[i : i + L + 1].all()]
    out = np.full((model.M, len(df)), np.nan)
    if not starts:
        return out
    W = {k: np.stack([v[i : i + L + 1] for i in starts]) for k, v in arr.items()}
    F = _forcing_tensors(W, n_sub)
    T0 = torch.as_tensor(W["T"][:, 0], dtype=TORCH_DTYPE)[None].expand(model.M, -1)
    pred = model.rollout(T0, F, L, n_sub).numpy()
    for b, i in enumerate(starts):
        out[:, i + 1 : i + L + 1] = pred[:, b, 1:]
    return out
