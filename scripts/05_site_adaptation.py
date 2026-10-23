"""Few-shot adaptation of trained ensembles to another year or site.

The first ``--adapt-days`` days with data of a target set are used for adaptation
(the last two of them for early stopping) and the rest for evaluation. Strategies:

  transfer  the Jaru 2008 model as trained, no adaptation;
  phys      only the 2-4 physical parameters are re-estimated, network frozen;
  full      all parameters (network and physics) are fine-tuned.

Physics-only models have a single adaptation strategy (``phys``); the black box
only ``full``. Output: ``results/adaptation/<season>__<set>.json``.

Example:
    python scripts/05_site_adaptation.py --season dry --set sinop2003_08-09
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from amazon_graybox import sanity
from amazon_graybox.experiments import load_model, model_spec, persistence_baseline, test_sets, training_scalers
from amazon_graybox.metrics import skill
from amazon_graybox.training import TrainConfig, block_arrays, forecast, simulate, train

MODELS = ["dry_none", "moist_closed_none", "dry_add_TR", "dry_hyb_TRHe", "moist_closed_hyb_TRHe", "none_add_TRHe"]


def split_adaptation(d, adapt_days: float):
    """Index position after which ``adapt_days`` days of complete records have accumulated."""
    ok = np.all([np.isfinite(v) for v in block_arrays(d).values()], axis=0)
    need = int(adapt_days * 48)
    cut = int(np.searchsorted(np.cumsum(ok), need)) + 1
    return d.iloc[:cut], d.iloc[cut:]


def scores(model, d):
    obs = d["T"].to_numpy()
    sim, fcs = simulate(model, d), forecast(model, d, 24.0)
    fr, fc = skill(obs, np.nanmean(sim, 0)), skill(obs, np.nanmean(fcs, 0))
    return {"fr_rmse": fr["rmse"], "fr_bias": fr["bias"], "fc_rmse": fc["rmse"], "fc_bias": fc["bias"],
            "fr_implausible": sanity.implausible_fraction(sim), "fc_implausible": sanity.implausible_fraction(fcs)}


def main(a):
    torch.set_num_threads(a.threads)
    results = Path(a.results)
    sc = training_scalers(a.processed, a.season)
    d = test_sets(a.processed, a.season)[a.set]
    adapt, ev = split_adaptation(d, a.adapt_days)
    a_tr, a_va = split_adaptation(adapt, a.adapt_days - 2)
    obs = ev["T"].to_numpy()
    pers = persistence_baseline(ev)
    out = {"season": a.season, "set": a.set, "adapt_days": a.adapt_days,
           "n_adapt": int(len(adapt)), "n_eval": int(len(ev)),
           "persistence_rmse": skill(obs, pers)["rmse"], "models": {}}
    for tag in MODELS:
        if not (results / a.season / f"{tag}.pt").exists():
            continue
        white, corr, _ = model_spec(tag)
        res = {"transfer": scores(load_model(results, a.season, tag, sc), ev)}
        strategies = ["phys"] if corr == "none" else (["full"] if white == "none" else ["phys", "full"])
        for strat in strategies:
            m = load_model(results, a.season, tag, sc)
            if strat == "phys":
                for n, p in m.named_parameters():
                    if n.startswith("net."):
                        p.requires_grad_(False)
            cfg = TrainConfig(window_h=24.0, stride_h=6.0, epochs=a.epochs, patience=60, lr_net=1e-3,
                              lr_phys=1e-2, lam_corr=1e-3, verbose=0, seed=1)
            t0 = time.time()
            try:
                train(m, a_tr, a_va, cfg)
            except ValueError as err:  # no complete window in a short block
                res[strat] = {"error": str(err)}
                continue
            sanity.check_params(m.phys_params_numpy(), m.delta_max if white == "moist_closed" else None, f"{tag} {strat}")
            res[strat] = scores(m, ev) | {"seconds": time.time() - t0,
                                          "params": {k: float(np.median(v)) for k, v in m.phys_params_numpy().items()}}
        out["models"][tag] = res
        print(a.season, a.set, tag, {k: round(v.get("fr_rmse", np.nan), 2) for k, v in res.items()}, flush=True)
    od = results / "adaptation"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{a.season}__{a.set}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=["dry", "wet"], required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--adapt-days", dest="adapt_days", type=float, default=7.0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    main(ap.parse_args())
