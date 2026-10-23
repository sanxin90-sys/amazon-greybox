"""How much of the closed moist model's skill comes through the vapour-pressure forcing.

Two diagnostics:

1. Linearised restoring rate of the closed moist white box around the mean state of the
   2008 season, split into the radiative part 4 gamma sigma T^3 and the humidity part
   beta RH dln(e_s)/dT (RH is diagnosed from the model temperature, so a warmer model
   has a lower RH and receives less heating).
2. Free runs with the vapour pressure replaced by its 24-h centred running mean, which
   keeps the humidity level but removes its sub-daily co-variation with the observed
   temperature (e is computed from the observed RH and T of the same probe).

Output: ``results/humidity_diagnostics_<season>.json``.

    python scripts/09_humidity_diagnostics.py --season dry
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from amazon_graybox import sanity
from amazon_graybox.data import select_months
from amazon_graybox.experiments import SEASON_MONTHS, load_model, read, test_sets, training_scalers
from amazon_graybox.metrics import skill
from amazon_graybox.physics import RHO_C, SIGMA, delta_max
from amazon_graybox.thermo import dln_es_dT, es_kpa
from amazon_graybox.training import simulate

MODELS = ["dry_add_TR", "dry_add_TRHe", "moist_closed_none", "moist_closed_mult_TRHe", "moist_closed_hyb_TRHe", "none_add_TRHe"]


def smooth_vapour_pressure(d, hours: float = 24.0):
    """Centred running mean of e, capped at saturation at the observed temperature.

    On cool, humid nights the running mean exceeds e_s(T); without the cap the forcing would be
    supersaturated (RH up to about 120 %). The fraction of capped samples is returned as well.
    """
    p = d.copy()
    w = int(hours * 2)
    e = d["e"].rolling(w + 1, center=True, min_periods=w // 2).mean()
    es = es_kpa(d["T"])
    capped = float((e > es).sum() / max(int(e.notna().sum()), 1))
    p["e"] = e.where(~(e > es), es)
    p["de"] = (p["e"].shift(-1) - p["e"].shift(1)) / 1.0  # kPa h-1, centred over 1 h
    return sanity.check_block(p, "smoothed vapour pressure"), capped


def restoring(results, processed, season):
    P = json.loads((results / season / "phys_moist_closed.json").read_text())["params"]
    med = {k: float(np.median(v)) for k, v in P.items()}
    d = select_months(read(processed, "jaru_2008"), *SEASON_MONTHS[season])
    T, RH = float(d["T"].mean()), float(d["RH"].mean())
    sanity.check_params(med, delta_max(), "phys_moist_closed")
    rad = 4 * med["gamma"] * SIGMA * T**3                       # W m-3 K-1
    hum = med["beta"] * RH * float(dln_es_dT(T))                # W m-3 K-1
    c_eff = RHO_C - med["delta"] * RH * float(dln_es_dT(T))     # J m-3 K-1
    sanity.require(c_eff > 0, "effective heat capacity must stay positive")
    return {"T_mean": T, "RH_mean": RH, "params": med, "radiative_W_m3_K": rad, "humidity_W_m3_K": hum,
            "humidity_share": hum / (rad + hum), "c_eff": c_eff,
            "tau_radiative_h": c_eff / rad / 3600, "tau_total_h": c_eff / (rad + hum) / 3600}


def main(a):
    torch.set_num_threads(a.threads)
    results = Path(a.results)
    out = {"restoring": restoring(results, a.processed, a.season), "free_run": {}}
    print(json.dumps(out["restoring"], indent=1))
    sc = training_scalers(a.processed, a.season)
    sets = test_sets(a.processed, a.season)
    for tag in MODELS:
        m = load_model(results, a.season, tag, sc)
        out["free_run"][tag] = {}
        for name, d in sets.items():
            obs = d["T"].to_numpy()
            sim_b = simulate(m, d)
            base = skill(obs, np.nanmean(sim_b, 0))
            ds, capped = smooth_vapour_pressure(d)
            sim_s = simulate(m, ds)
            smo = skill(obs, np.nanmean(sim_s, 0))
            out["free_run"][tag][name] = {"rmse": base["rmse"], "bias": base["bias"],
                                          "rmse_e24h": smo["rmse"], "bias_e24h": smo["bias"], "capped_fraction": capped,
                                          "implausible": sanity.implausible_fraction(sim_b),
                                          "implausible_e24h": sanity.implausible_fraction(sim_s)}
            print(f"{a.season} {tag:24s} {name:20s} RMSE {base['rmse']:6.2f} -> {smo['rmse']:6.2f}  "
                  f"bias {base['bias']:+6.2f} -> {smo['bias']:+6.2f}", flush=True)
    (results / f"humidity_diagnostics_{a.season}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=["dry", "wet"], required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    main(ap.parse_args())
