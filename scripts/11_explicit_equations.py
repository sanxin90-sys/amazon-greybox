"""Explicit equations of the corrected models (degree-1 distillation, median coefficients).

Silva et al. (2026) wrote their corrected model as one explicit equation with fixed
coefficients (their Equations (12)-(14)). This script does the same for each gray-box model
of this study, from the degree-1 distillation of ``10_polynomial_distillation.py``:

  * every network output is replaced by a linear function of the inputs, written about the
    mean state of the training block, l = l_bar + l_T (T - T_bar) + l_R (R - R_bar) + l_e (e - e_bar),
    with the median coefficient of the ten members (an additive correction in K h-1, or the
    logarithm of a multiplicative or hybrid factor);
  * the physical parameters are the medians over the members of the trained model;
  * the resulting single equation is integrated with the same scheme and scored on the same
    sets as the ensembles, without clipping of the inputs.

The dry model with an additive correction and inputs (T, R) is Equation (7) of the article.

Output: ``results/polynomial/explicit_equations.json``.

    python scripts/11_explicit_equations.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from amazon_graybox import sanity
from amazon_graybox.experiments import test_sets, training_scalers
from amazon_graybox.graybox import INPUT_SETS, GrayBoxODE
from amazon_graybox.metrics import skill
from amazon_graybox.training import simulate

# the polynomial stand-in for the network is defined in the distillation script
_spec = importlib.util.spec_from_file_location("distil", Path(__file__).with_name("10_polynomial_distillation.py"))
distil = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(distil)

MODELS = [  # (white box, correction, network inputs)
    ("dry", "add", "TR"), ("dry", "mult", "TRHe"), ("dry", "hyb", "TRHe"),
    ("moist_closed", "add", "TRHe"), ("moist_closed", "mult", "TRHe"), ("moist_closed", "hyb", "TRHe"),
]
OUT_OF_SAMPLE = distil.OUT_OF_SAMPLE


def explicit_model(white, corr, inp, season, a, sc):
    """Single-member model with median physical parameters and median degree-1 polynomials."""
    tag = f"{white}_{corr}_{inp}"
    z = json.loads((a.results / "polynomial" / f"{season}_{tag}.json").read_text())
    deg = z["degrees"]["1"]
    outs, used, terms = z["outputs"], z.get("poly_inputs", z["inputs"]), deg["terms"]
    model_inputs = list(INPUT_SETS[inp])  # inputs not used by the polynomial (RH) get exponent 0
    powers = np.array([[1 if term == n else 0 for n in model_inputs] for term in terms])
    coef = np.array([[[deg["coef"][o][term]["median"] for o in outs] for term in terms]])
    kinds = ["ln" if o.startswith("ln") else "c" for o in outs]
    P = json.loads((a.results / season / f"{tag}.json").read_text())["params"]
    p0 = {k: float(np.median(P[k])) for k in ("alpha", "gamma", "beta", "delta") if k in P}
    m = GrayBoxODE(white, corr, M=1, inputs=inp, scalers=sc, p0=p0)
    sanity.check_params(p0, m.delta_max if white == "moist_closed" else None, tag)
    big = np.full(len(model_inputs), 1e9)
    m.net = distil.PolyNet(powers, coef, kinds, sc.out, m.mult_bound, -big, big, clip=False)
    # centred coefficients in physical units: intercept at the mean state, slope per unit of input
    terms_phys = {o: {"centre": deg["coef"][o]["1"]["median"],
                      **{n: deg["coef"][o][n]["median"] / z["scaler_std"][n] for n in used}} for o in outs}
    info = {"white": white, "correction": corr, "inputs": inp, "poly_inputs": used, "params": p0,
            "mean_state": {n: z["scaler_mean"][n] for n in used}, "terms": terms_phys,
            "ensemble_degree1": {k: deg[k] for k in ("fr_rmse_in", "fr_rmse_oos")}}
    return tag, m, info


def main(a):
    warnings.filterwarnings("ignore")
    torch.set_num_threads(a.threads)
    out = {}
    for season in ("dry", "wet"):
        sc = training_scalers(a.processed, season)
        sets = test_sets(a.processed, season)
        out[season] = {}
        for white, corr, inp in MODELS:
            tag, m, info = explicit_model(white, corr, inp, season, a, sc)
            sims = {n: simulate(m, d)[0] for n, d in sets.items()}
            r = {n: float(skill(d["T"].to_numpy(), sims[n])["rmse"]) for n, d in sets.items()}
            info["sets"] = r
            info["implausible"] = {n: sanity.implausible_fraction(v) for n, v in sims.items()}
            info["fr_rmse_in"] = r["jaru2008_test"]
            info["fr_rmse_oos"] = float(np.mean([v for n, v in r.items() if OUT_OF_SAMPLE(n)]))
            out[season][tag] = info
            print(f"{season} {tag}: test {info['fr_rmse_in']:.2f} K, out of sample {info['fr_rmse_oos']:.2f} K "
                  f"(ensemble of degree-1 polynomials: {info['ensemble_degree1']['fr_rmse_in']:.2f}, "
                  f"{info['ensemble_degree1']['fr_rmse_oos']:.2f} K)", flush=True)
    (a.results / "polynomial" / "explicit_equations.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    main(ap.parse_args())
