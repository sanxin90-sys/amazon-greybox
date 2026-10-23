"""Train one gray-box ensemble and evaluate it on every test set.

Example:
    python scripts/03_run_experiment.py --season dry --white moist_closed --corr hyb --inputs TRHe

Physics-only runs (``--corr none``) must be done first: corrected models start
from the physical parameters they produce (``results/<season>/phys_<white>.json``).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from amazon_graybox import sanity
from amazon_graybox.experiments import calibration_split, climatology_baseline, test_sets
from amazon_graybox.graybox import GrayBoxODE, Scalers
from amazon_graybox.metrics import daily_extremes, skill
from amazon_graybox.training import TrainConfig, forecast, simulate, train

DEFAULT_P0 = {
    # Starting values: de Paulo et al. (2023) and Maionchi et al. (2025, Rebio Jaru, P = 1).
    "dry": {"alpha": 2.7e-3, "gamma": 6.5e-4},
    "moist_open": {"alpha": 0.55e-3, "gamma": 6.5e-4, "beta": 2.7e-3, "delta": 154.0},
    "moist_closed": {"alpha": 0.55e-3, "gamma": 6.5e-4, "beta": 2.7e-3, "delta": 0.0},
    "none": {},
}


def tag_of(a) -> str:
    return f"{a.white}_{a.corr}" + ("" if a.corr == "none" else f"_{a.inputs}")


def main(a):
    torch.set_num_threads(a.threads)
    out = Path(a.results) / a.season
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = calibration_split(a.processed, a.season)
    sc = Scalers.fit({"T": tr["T"], "R": tr["R"], "RH": tr["RH"], "e": tr["e"]}, float(np.nanstd(tr["dT"])))
    p0 = dict(DEFAULT_P0[a.white])
    phys_file = out / f"phys_{a.white}.json"
    if a.corr != "none" and a.white != "none":
        p0 |= {k: float(np.median(v)) for k, v in json.loads(phys_file.read_text())["params"].items()}
    model = GrayBoxODE(a.white, a.corr, M=a.M, inputs=a.inputs, hidden=tuple(a.hidden), p0=p0, scalers=sc, seed=a.seed)
    cfg = TrainConfig(epochs=a.epochs, seed=a.seed, lam_corr=a.lam_corr, lr_net=a.lr, verbose=a.verbose)
    if a.corr == "none":
        cfg.lr_phys = 2e-2
    t0 = time.time()
    hist, best_val = train(model, tr, va, cfg)
    train_s = time.time() - t0
    tag = tag_of(a)
    torch.save(model.state_dict(), out / f"{tag}.pt")
    hist.to_csv(out / f"{tag}_history.csv", index=False)

    params = {k: v.tolist() for k, v in model.phys_params_numpy().items()}
    sanity.check_params(params, model.delta_max if a.white == "moist_closed" else None, tag)
    if a.corr == "none" and a.white != "none":
        phys_file.write_text(json.dumps({"params": params, "val_rmse": best_val.tolist()}, indent=1))

    # residual scale for the predictive distribution, from the validation free run
    sim_va = simulate(model, va)
    sigma_res = float(np.sqrt(np.nanmean((sim_va.mean(0) - va["T"].to_numpy()) ** 2)))

    results = {"tag": tag, "season": a.season, "white": a.white, "corr": a.corr, "inputs": a.inputs,
               "M": a.M, "n_params_net": model.net.n_params_per_member() if model.net else 0,
               "train_seconds": train_s, "epochs_run": int(hist.epoch.max() + 1),
               "best_val_rmse": best_val.tolist(), "params": params, "sigma_res_val": sigma_res, "sets": {}}
    preds = {}
    for name, d in test_sets(a.processed, a.season).items():
        obs = d["T"].to_numpy()
        sim = simulate(model, d)
        sim4 = simulate(model, d, n_sub=4)
        fc = forecast(model, d, horizon_h=24.0)
        mu = np.nanmean(sim, axis=0)
        r = {
            "free_run": skill(obs, mu),
            "free_run_members_rmse": [skill(obs, s)["rmse"] for s in sim],
            "free_run_substep4_rmse": skill(obs, np.nanmean(sim4, axis=0))["rmse"],
            "forecast24": skill(obs, np.nanmean(fc, axis=0)),
            "extremes": daily_extremes(d.index, obs, mu),
            "spread_mean": float(np.nanmean(np.nanstd(sim, axis=0))),
        }
        r["implausible"] = {"free_run": sanity.implausible_fraction(sim), "forecast24": sanity.implausible_fraction(fc)}
        if name == "jaru2008_test":
            r["climatology"] = skill(obs, climatology_baseline(tr, d))
        results["sets"][name] = r
        preds[name] = {"sim": sim.astype(np.float32), "fc": fc.astype(np.float32)}
        print(f"{tag:32s} {name:20s} free-run RMSE {r['free_run']['rmse']:.3f} K  R2 {r['free_run']['r2']:.3f}  "
              f"24h-fc RMSE {r['forecast24']['rmse']:.3f} K  (sub4 {r['free_run_substep4_rmse']:.3f})", flush=True)
    np.savez_compressed(out / f"{tag}_pred.npz", **{f"{k}__{kk}": vv for k, v in preds.items() for kk, vv in v.items()})
    (out / f"{tag}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=["dry", "wet"], required=True)
    ap.add_argument("--white", choices=["none", "dry", "moist_open", "moist_closed"], required=True)
    ap.add_argument("--corr", choices=["none", "add", "mult", "hyb"], default="none")
    ap.add_argument("--inputs", choices=["TR", "TRH", "TRHe"], default="TRHe")
    ap.add_argument("--hidden", type=int, nargs="+", default=[32, 16])
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam-corr", dest="lam_corr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--verbose", type=int, default=100)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    main(ap.parse_args())
