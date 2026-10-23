"""Response of the trained ensembles to controlled perturbations of the forcing.

24-h forecasts over the whole Jaru 2008 season are repeated with
  R  x 1.1         (10 % more net radiation),
  e  x 0.9         (10 % less water vapour, i.e. a drier atmosphere),
and the changes of the mean temperature, of the daily maximum and minimum and of
the 20 h -> 23 h nocturnal cooling are reported for every ensemble member. The
spread across members measures how well the response is constrained by the
model structure, beyond the data the networks were trained on.

Output: ``results/sensitivity_<season>.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from amazon_graybox import sanity
from amazon_graybox.data import select_months
from amazon_graybox.experiments import SEASON_MONTHS, load_model, read, training_scalers
from amazon_graybox.metrics import nocturnal_cooling
from amazon_graybox.thermo import es_kpa
from amazon_graybox.training import forecast

MODELS = ["dry_none", "moist_closed_none", "dry_add_TR", "dry_add_TRHe", "dry_mult_TRHe", "dry_hyb_TRHe",
          "moist_closed_add_TRHe", "moist_closed_mult_TRHe", "moist_closed_hyb_TRHe", "none_add_TRHe"]


def perturb(d: pd.DataFrame, kind: str) -> pd.DataFrame:
    p = d.copy()
    if kind == "R+10%":
        p["R"] = d["R"] * 1.1
    elif kind == "e-10%":
        p["e"] = d["e"] * 0.9
        p["de"] = d["de"] * 0.9
        p["RH"] = (100 * p["e"] / es_kpa(d["T"])).clip(upper=100)
        p["dRH"] = p["RH"].diff().shift(-1).add(p["RH"].diff()).div(2 * 0.5)
    return p


def daily(index, T):
    s = pd.Series(T, index=index)
    g = s.groupby((s.index - pd.Timedelta(minutes=30)).normalize())
    return g.max(), g.min()


def main(a):
    torch.set_num_threads(a.threads)
    results = Path(a.results)
    sc = training_scalers(a.processed, a.season)
    d = sanity.check_block(select_months(read(a.processed, "jaru_2008"), *SEASON_MONTHS[a.season]), f"jaru_2008 {a.season}")
    perturbed = {k: sanity.check_block(perturb(d, k), f"jaru_2008 {a.season} {k}") for k in ("R+10%", "e-10%")}
    out = {}
    for tag in MODELS:
        if not (results / a.season / f"{tag}.pt").exists():
            continue
        m = load_model(results, a.season, tag, sc)
        base = forecast(m, d, 24.0)
        bmax, bmin = zip(*(daily(d.index, b) for b in base))
        bcool = [nocturnal_cooling(d.index, b) for b in base]
        out[tag] = {}
        for kind, dp in perturbed.items():
            pert = forecast(m, dp, 24.0)
            dT = np.nanmean(pert - base, axis=1)
            dmax = [float(np.nanmean(daily(d.index, p)[0] - bx)) for p, bx in zip(pert, bmax)]
            dmin = [float(np.nanmean(daily(d.index, p)[1] - bn)) for p, bn in zip(pert, bmin)]
            dcool = [float(np.nanmean(nocturnal_cooling(d.index, p) - bc)) for p, bc in zip(pert, bcool)]
            out[tag][kind] = {"dT_members": dT.tolist(), "dTmax_members": dmax, "dTmin_members": dmin,
                              "dcool_members": dcool}
            print(f"{a.season} {tag:24s} {kind:6s} dT {dT.mean():+.2f} (sd {dT.std():.2f})  dTmax {np.mean(dmax):+.2f} "
                  f"(sd {np.std(dmax):.2f})  dTmin {np.mean(dmin):+.2f} (sd {np.std(dmin):.2f})  "
                  f"dcool {np.mean(dcool):+.2f} (sd {np.std(dcool):.2f})", flush=True)
    (results / f"sensitivity_{a.season}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=["dry", "wet"], required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    main(ap.parse_args())
