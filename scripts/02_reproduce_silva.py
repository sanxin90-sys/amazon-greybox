"""Re-implementation of the Silva et al. (2026) protocol (notebooks Code_Dry / Code_Wet).

Steps, following the notebooks cell by cell:
  1. alpha*, gamma* by free-run RK4 least squares over the whole season (time unit 1e6 s,
     forcing held constant over each step, L-BFGS-B with the notebook bounds).
  2. Additive NN(T, R) [32, 16] LeakyReLU trained on finite-difference dT/dt residuals
     (dry: random 70/15/15 split; wet: chronological split), StandardScaler, Adam.
  3. Free-run integration with explicit Euler, the network evaluated on the model state.
  4. Bootstrap ensemble as in "Unit 10": correction evaluated on the *observed* T and
     integrated open loop, plus percentiles of the correction integrated as bounds.
     The closed-loop alternative (each member integrated on its own state) is also run.

Outputs ``results/silva_protocol_<season>.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from amazon_graybox import sanity
from amazon_graybox.data import select_months
from amazon_graybox.experiments import SEASON_MONTHS, read
from amazon_graybox.metrics import skill
from amazon_graybox.physics import silva_to_si

DT = 30 * 60e-6  # notebook time step: minutes * 60e-6 (units of 1e6 s)
SETTINGS = {
    "dry": dict(guess=[1.2, 1e-8], fac=40.0, split="random", lr=1e-3, wd=1e-4, epochs=500),
    "wet": dict(guess=[2e-1, 1e-8], fac=4.0, split="chrono", lr=1e-4, wd=0.0, epochs=60),
}


def rk4_zoh(T, Rt, dt, a, g):
    f = lambda T: a * Rt - g * T**4  # noqa: E731
    k1 = dt * f(T); k2 = dt * f(T + 0.5 * k1); k3 = dt * f(T + 0.5 * k2); k4 = dt * f(T + k3)
    return T + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def free_run(T0, R, a, g, n):
    T = np.empty(n); T[0] = T0
    for i in range(1, n):
        T[i] = rk4_zoh(T[i - 1], R[i - 1], DT, a, g)
    return T


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(0.01), nn.Linear(32, 16), nn.LeakyReLU(0.01), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x)


def seeded_net(k):
    torch.manual_seed(k)
    return Net()


class BatchedNet(nn.Module):
    """M independent copies of :class:`Net` evaluated with batched matrix products."""

    def __init__(self, nets):
        super().__init__()
        lin = [[m for m in net.net if isinstance(m, nn.Linear)] for net in nets]
        self.W = nn.ParameterList([nn.Parameter(torch.stack([l[i].weight.T.detach() for l in lin])) for i in range(3)])
        self.b = nn.ParameterList([nn.Parameter(torch.stack([l[i].bias[None].detach() for l in lin])) for i in range(3)])

    def forward(self, x, M=None):  # x: [M, B, 2]
        M = M or x.shape[0]
        for i in range(3):
            x = torch.baddbmm(self.b[i][:M], x, self.W[i][:M])
            if i < 2:
                x = nn.functional.leaky_relu(x, 0.01)
        return x

    @torch.no_grad()
    def euler(self, T0, R, a, g, sx, sy, M, dt):
        """Closed-loop explicit-Euler runs of the first M members, each on its own state."""
        n = len(R); T = np.empty((M, n)); T[:, 0] = T0
        for i in range(1, n):
            z = (np.stack([T[:, i - 1], np.full(M, R[i - 1])], 1) - sx.mean_) / sx.scale_
            c = self(torch.tensor(z[:, None, :], dtype=torch.float32), M).numpy()[:, 0, 0] * sy.scale_[0] + sy.mean_[0]
            T[:, i] = T[:, i - 1] + (a * R[i - 1] - g * T[:, i - 1] ** 4 + c) * dt
        return T


def season_arrays(processed, season, year):
    d = sanity.check_block(select_months(read(processed, f"jaru_{year}"), *SEASON_MONTHS[season]), f"jaru_{year} {season}")
    d = d.dropna(subset=["T", "R"])
    return d, d["T"].to_numpy(np.float32).astype(float), d["R"].to_numpy(np.float32).astype(float)


def fd_derivative(T):
    d = np.zeros_like(T)
    d[1:-1] = (T[2:] - T[:-2]) / (2 * DT)
    d[0] = (T[1] - T[0]) / DT
    d[-1] = (T[-1] - T[-2]) / DT
    return d


def train_main(X, y, s, seed=42):
    if s["split"] == "random":
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, random_state=42)
        Xva, _, yva, _ = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=42)
    else:
        n = len(X); a, b = int(0.7 * n), int(0.85 * n)
        Xtr, ytr, Xva, yva = X[:a], y[:a], X[a:b], y[a:b]
    sx, sy = StandardScaler().fit(Xtr), StandardScaler().fit(ytr[:, None])
    t = lambda z: torch.tensor(z, dtype=torch.float32)  # noqa: E731
    Xtr_, ytr_, Xva_, yva_ = t(sx.transform(Xtr)), t(sy.transform(ytr[:, None])), t(sx.transform(Xva)), t(sy.transform(yva[:, None]))
    torch.manual_seed(seed)
    m = Net(); opt = torch.optim.Adam(m.parameters(), lr=s["lr"], weight_decay=s["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    best, count = np.inf, 0
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr_, ytr_), batch_size=128, shuffle=True)
    for ep in range(s["epochs"]):
        m.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = nn.functional.mse_loss(m(xb), yb); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(m(Xva_), yva_).item()
        sched.step(vl)
        if vl < best:
            best, count = vl, 0
        else:
            count += 1
            if count >= 10:
                break
    return m, sx, sy, ep


def correction_fn(m, sx, sy):
    def f(T, R):
        x = torch.tensor(sx.transform(np.array([[T, R]])), dtype=torch.float32)
        with torch.no_grad():
            return float(sy.inverse_transform(m(x).numpy())[0, 0])
    return f


def euler_corrected(T0, R, a, g, corr, n):
    T = np.empty(n); T[0] = T0
    for i in range(1, n):
        T[i] = T[i - 1] + (a * R[i - 1] - g * T[i - 1] ** 4 + corr(T[i - 1], R[i - 1])) * DT
    return T


def main(a):
    torch.set_num_threads(a.threads)
    s = SETTINGS[a.season]
    d, T_obs, R = season_arrays(a.processed, a.season, 2008)
    n = len(T_obs)
    obj = lambda p: np.mean((T_obs - free_run(T_obs[0], R, p[0], p[1], n)) ** 2)  # noqa: E731
    bounds = [(s["guess"][0] / s["fac"], s["guess"][0] * s["fac"]), (s["guess"][1] / s["fac"], s["guess"][1] * s["fac"])]
    res = minimize(obj, s["guess"], bounds=bounds)
    al, ga = res.x
    T_unc = free_run(T_obs[0], R, al, ga, n)
    out = {"season": a.season, "alpha_star": al, "gamma_star": ga, "alpha_si": silva_to_si(al, ga).alpha,
           "gamma_si": silva_to_si(al, ga).gamma, "rmse_uncorrected_rk4": skill(T_obs, T_unc)["rmse"]}

    X = np.vstack([T_obs, R]).T
    y = fd_derivative(T_obs) - (al * R - ga * T_obs**4)
    m, sx, sy, ep = train_main(X, y, s)
    corr = correction_fn(m, sx, sy)
    T_cor = euler_corrected(T_obs[0], R, al, ga, corr, n)
    out |= {"epochs_main": ep + 1, "main_model_season": skill(T_obs, T_cor),
            "uncorrected_euler": skill(T_obs, euler_corrected(T_obs[0], R, al, ga, lambda *_: 0.0, n))}
    # out-of-year transfer of the main model (parameters and network frozen)
    out["main_model_other_years"] = {}
    for yr in (2009, 2010):
        _, To, Ro = season_arrays(a.processed, a.season, yr)
        runs = {"uncorrected": free_run(To[0], Ro, al, ga, len(To)), "corrected": euler_corrected(To[0], Ro, al, ga, corr, len(To))}
        out["main_model_other_years"][yr] = {k: skill(To, v)["rmse"] for k, v in runs.items()}
        out["main_model_other_years"][yr]["implausible_fraction"] = {k: sanity.implausible_fraction(v) for k, v in runs.items()}
    # bootstrap ensemble, "Unit 10": members trained side by side (same algorithm as the notebook loop)
    sxe, sye = StandardScaler().fit(X), StandardScaler().fit(y[:, None])
    Xt = torch.tensor(sxe.transform(X), dtype=torch.float32); yt = torch.tensor(sye.transform(y[:, None]), dtype=torch.float32)
    rng = np.random.default_rng(0)
    boot = torch.tensor(np.stack([rng.integers(0, n, n) for _ in range(a.n_ens)]))
    ens = BatchedNet([seeded_net(k) for k in range(a.n_ens)])
    opt = torch.optim.Adam(ens.parameters(), lr=1e-3, weight_decay=1e-4)
    torch.manual_seed(0)
    for _ in range(100):
        perm = torch.gather(boot, 1, torch.argsort(torch.rand(boot.shape), dim=1))
        for j in range(0, n, 128):
            ids = perm[:, j:j + 128]
            opt.zero_grad()
            ((ens(Xt[ids]) - yt[ids]) ** 2).mean(dim=(1, 2)).sum().backward()  # sum of per-member mean losses
            opt.step()
    with torch.no_grad():
        P = ens(Xt.expand(a.n_ens, -1, -1)).squeeze(-1).numpy() * sye.scale_[0] + sye.mean_[0]
    closed_loop = list(ens.euler(T_obs[0], R, al, ga, sxe, sye, a.n_closed, DT))

    def integ(c):
        T = np.empty(n); T[0] = T_obs[0]
        for i in range(1, n):
            T[i] = T[i - 1] + (al * R[i - 1] - ga * T[i - 1] ** 4 + c[i - 1]) * DT
        return T

    T_mean, T_p5, T_p95 = integ(P.mean(0)), integ(np.percentile(P, 5, 0)), integ(np.percentile(P, 95, 0))
    C = np.array(closed_loop)
    out["implausible_fraction"] = {"uncorrected": sanity.implausible_fraction(T_unc), "corrected": sanity.implausible_fraction(T_cor),
                                   "closed_loop": sanity.implausible_fraction(C)}
    lo, hi = np.percentile(C, 5, 0), np.percentile(C, 95, 0)
    out["unit10_open_loop"] = {"ensemble_mean": skill(T_obs, T_mean),
                               "coverage90": float(np.mean((T_obs >= np.minimum(T_p5, T_p95)) & (T_obs <= np.maximum(T_p5, T_p95)))),
                               "mean_band_width_K": float(np.mean(np.abs(T_p95 - T_p5))),
                               "bounds_implausible_fraction": sanity.implausible_fraction(np.stack([T_p5, T_p95]))}
    out["closed_loop_ensemble"] = {"n_members": len(C), "ensemble_mean": skill(T_obs, C.mean(0)),
                                   "member_rmse": [skill(T_obs, c)["rmse"] for c in C],
                                   "coverage90": float(np.mean((T_obs >= lo) & (T_obs <= hi))),
                                   "mean_band_width_K": float(np.mean(hi - lo))}
    Path(a.results).mkdir(parents=True, exist_ok=True)
    (Path(a.results) / f"silva_protocol_{a.season}.json").write_text(json.dumps(out, indent=1, default=float))
    np.savez_compressed(Path(a.results) / f"silva_protocol_{a.season}_series.npz", T_obs=T_obs, T_unc=T_unc, T_cor=T_cor,
                        T_mean_open=T_mean, T_p5=T_p5, T_p95=T_p95, closed=C, time=d.index.values.astype("datetime64[m]").astype(float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("closed_loop_ensemble",)}, indent=1, default=float)[:3000])
    print("closed loop:", out["closed_loop_ensemble"]["ensemble_mean"], out["closed_loop_ensemble"]["coverage90"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=["dry", "wet"], required=True)
    ap.add_argument("--n-ens", dest="n_ens", type=int, default=100)
    ap.add_argument("--n-closed", dest="n_closed", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    main(ap.parse_args())
