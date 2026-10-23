"""Polynomial distillation of the learned corrections (explicit, interpretable equations).

Silva et al. (2026) replaced their additive network N(T, R) by a fourth-degree polynomial of
T and R in order to write the corrected model as an explicit equation. This script does the
same for every trained correction and checks whether the explicit equation keeps the skill
of the network.

Part 1, trained ensembles (``results/<season>/<tag>.pt``):
  * the network output is mapped to "correction space": the additive correction s N(z) in
    K h-1, the logarithm b tanh N(z) of each multiplicative or hybrid factor, and the
    additive residual of the hybrid form;
  * polynomials of degree 1 to 3 (and 4 for the (T, R) inputs, as in Silva et al.) in the
    standardised inputs are fitted member by member, by least squares, to the correction at
    the observed states of the training block. For networks with inputs (T, R, RH, e) the
    polynomials use (T, R, e): RH is diagnosed from T and e, so it adds no information and
    makes the fit nearly collinear (``--poly-inputs all`` keeps all four);
  * each member's network is replaced by its polynomials and the distilled ensemble is
    integrated with the same RK4 scheme and scored on the same sets as the network. The
    inputs of the polynomials are clipped to the range of the training block (polynomials
    diverge outside it); free runs without clipping are also scored.
  The degree-1 fits of the additive corrections are converted into changes of the effective
  absorption coefficient alpha, of the linear restoring rate and of an effective beta.

Part 2, protocol of Silva et al.: the polynomial step of the notebook (degree 4 in the
unscaled T and R, ordinary least squares on the training samples, explicit Euler) is
repeated, fitted both to the finite-difference residuals (as in the notebook) and to the
network output, and the cancellation between its terms is quantified.

Output: ``results/polynomial/<season>_<tag>.json`` and ``results/polynomial/<season>_silva.json``.

    python scripts/10_polynomial_distillation.py --season dry
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch import nn

from amazon_graybox import sanity
from amazon_graybox.experiments import (
    calibration_split, climatology_baseline, load_model, persistence_baseline, test_sets, training_scalers,
)
from amazon_graybox.graybox import TORCH_DTYPE, es_t
from amazon_graybox.metrics import skill
from amazon_graybox.physics import RHO_C, SEC_PER_H, SIGMA
from amazon_graybox.thermo import dln_es_dT, es_kpa
from amazon_graybox.training import FORCING_COLS, block_arrays, forecast, simulate

MODELS = [
    "dry_add_TR", "dry_add_TRHe", "dry_mult_TRHe", "dry_hyb_TRHe",
    "moist_closed_add_TR", "moist_closed_add_TRHe", "moist_closed_mult_TRHe", "moist_closed_hyb_TRHe",
    "none_add_TRHe",
]
OUT_OF_SAMPLE = lambda name: name not in ("jaru2008_test", "jaru2009")  # noqa: E731  (as in Table 4)
REQUIRED = ["T", "R", "RH", "e", "de", "dRH"]


# --------------------------------------------------------------------------- part 1
def output_kinds(model) -> tuple[list[str], list[str]]:
    """Names and kinds ("c": additive, K h-1; "ln": log of a factor) of the network outputs."""
    if model.correction in ("add",):
        return ["c"], ["c"]
    if model.correction == "mult":
        return ["ln_m"], ["ln"]
    terms = ["alpha", "gamma", "beta"][: 3 if model.white.startswith("moist") else 2]
    return [f"ln_m_{t}" for t in terms] + ["c0"], ["ln"] * len(terms) + ["c"]


def to_correction_space(out, kinds, s, b):
    cols = [s * out[..., k] if kind == "c" else b * torch.tanh(out[..., k]) for k, kind in enumerate(kinds)]
    return torch.stack(cols, -1)


class PolyNet(nn.Module):
    """Polynomial stand-in for the ensemble network: scaled inputs (M, B, d) -> network outputs.

    The polynomial value y is mapped back to the network output space so that GrayBoxODE.rhs
    recovers y exactly: out = y / s for additive outputs, atanh(y / b) for log-factors.
    """

    def __init__(self, powers, coef, kinds, s, b, lo, hi, clip=True):
        super().__init__()
        self.register_buffer("powers", torch.as_tensor(powers, dtype=TORCH_DTYPE))  # (P, d)
        self.register_buffer("coef", torch.as_tensor(coef, dtype=TORCH_DTYPE))  # (M, P, K)
        self.register_buffer("lo", torch.as_tensor(lo, dtype=TORCH_DTYPE))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=TORCH_DTYPE))
        self.kinds, self.s, self.b, self.clip = kinds, s, b, clip

    def forward(self, x):
        if self.clip:
            x = torch.maximum(torch.minimum(x, self.hi), self.lo)
        phi = torch.prod(x[..., None, :] ** self.powers, dim=-1)  # (M, B, P)
        y = torch.bmm(phi, self.coef)
        cols = [y[..., k] / self.s if kind == "c" else torch.atanh((y[..., k] / self.b).clamp(-1 + 1e-9, 1 - 1e-9))
                for k, kind in enumerate(self.kinds)]
        return torch.stack(cols, -1)


def states(df):
    """Forcing tensors and observed temperature at the complete samples of a block."""
    arr = block_arrays(df.dropna(subset=REQUIRED))
    F = {k: torch.as_tensor(arr[k], dtype=TORCH_DTYPE)[None] for k in FORCING_COLS}
    return torch.as_tensor(arr["T"], dtype=TORCH_DTYPE)[None], F


@torch.no_grad()
def corrections_at(model, T, F, kinds):
    """Scaled inputs (N, d) and network corrections (M, N, K) at observed states."""
    Tm = T.expand(model.M, -1)
    rh = F["RH_obs"] if model.white == "moist_open" else (100.0 * F["e"] / es_t(Tm)).clamp(0.0, 100.0)
    x = model._features(Tm, F, rh)
    y = to_correction_space(model.net(x), kinds, model.scalers.out, model.mult_bound)
    return x[0].numpy(), y.numpy()


def design(x, powers, lo=None, hi=None):
    if lo is not None:
        x = np.clip(x, lo, hi)
    return np.prod(x[:, None, :] ** powers[None], axis=-1)


def r2(y, yhat):
    """R2 per member and output: y, yhat (M, N, K) -> (M, K)."""
    return 1 - ((y - yhat) ** 2).sum(1) / ((y - y.mean(1, keepdims=True)) ** 2).sum(1)


def scores(model, sets, train_df):
    """Free-run and forecast RMSE of the ensemble mean per set, and pooled forecast skill."""
    out = {}
    for name, d in sets.items():
        obs = d["T"].to_numpy()
        sim, fc = simulate(model, d), forecast(model, d)
        mu_s, mu_f = np.nanmean(sim, 0), np.nanmean(fc, 0)
        clim, pers = climatology_baseline(train_df, d), persistence_baseline(d)
        m = np.isfinite(obs) & np.isfinite(mu_f) & np.isfinite(clim) & np.isfinite(pers)
        out[name] = {"fr_rmse": skill(obs, mu_s)["rmse"], "fr_bias": skill(obs, mu_s)["bias"],
                     "fc_rmse": float(np.sqrt(np.mean((mu_f[m] - obs[m]) ** 2))),
                     "pers_rmse": float(np.sqrt(np.mean((pers[m] - obs[m]) ** 2))),
                     "fr_implausible": sanity.implausible_fraction(sim), "fc_implausible": sanity.implausible_fraction(fc)}
    return out


def summary(sc):
    oos = [v for k, v in sc.items() if OUT_OF_SAMPLE(k)]
    return {"fr_rmse_in": sc["jaru2008_test"]["fr_rmse"], "fc_rmse_in": sc["jaru2008_test"]["fc_rmse"],
            "fr_rmse_oos": float(np.mean([v["fr_rmse"] for v in oos])),
            "fc_rmse_oos": float(np.mean([v["fc_rmse"] for v in oos])),
            "ss_pers_oos": float(1 - np.mean([v["fc_rmse"] ** 2 for v in oos]) / np.mean([v["pers_rmse"] ** 2 for v in oos]))}


def quantiles(a):
    return {"median": float(np.median(a)), "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95))}


def physical_linear(model, coef, names, inputs, sc, train_df):
    """Degree-1 additive correction c = a0 + sum_i a_i x_i expressed in physical units.

    b_i = a_i / std_i is the change of the correction (K h-1) per unit of input i. For a
    white box with heat capacity C (rho c, or C_eff for the closed moist model):
        alpha_eff = alpha + C b_R / 3600                       [m-1]
        beta_eff  = (beta) + C b_RH / 3600                     [W m-3 %-1]
        restoring rate (h-1): white box 3600 * 4 gamma sigma T^3 / C (+ humidity part)
        and correction -(b_T - b_RH RH dln(e_s)/dT), since RH = 100 e / e_s(T) at fixed e.
    """
    P = model.phys_params_numpy()
    T_m, RH_m = float(train_df["T"].mean()), float(train_df["RH"].mean())
    g = float(dln_es_dT(T_m))
    C = RHO_C - P["delta"] * RH_m * g if model.white == "moist_closed" else np.full(model.M, RHO_C)
    lin = {n: coef[:, names.index(n), 0] / sc.std[n] for n in inputs}  # per member, K h-1 per unit
    res = {"a0_K_h": quantiles(coef[:, names.index("1"), 0]), "mean_state": {"T": T_m, "RH": RH_m, "R": float(train_df["R"].mean()), "e": float(train_df["e"].mean())}}
    for n, v in lin.items():
        res[f"b_{n}"] = quantiles(v)
    if model.white == "none":
        return res
    res["alpha"] = quantiles(P["alpha"])
    res["alpha_eff"] = quantiles(P["alpha"] + C * lin["R"] / SEC_PER_H)
    lam_wb = SEC_PER_H * 4 * P["gamma"] * SIGMA * T_m**3 / C
    if model.white == "moist_closed":
        lam_wb = lam_wb + SEC_PER_H * P["beta"] * RH_m * g / C
        res["beta"] = quantiles(P["beta"])
    lam_c = -lin["T"] + (lin["RH"] * RH_m * g if "RH" in lin else 0.0)
    res["restoring_white_h"] = quantiles(lam_wb)
    res["restoring_correction_h"] = quantiles(lam_c)
    res["tau_total_h"] = quantiles(1.0 / (lam_wb + lam_c))
    # humidity sensitivity at fixed T, per % of RH (from b_e when e is the input: dRH/de = 100 / e_s)
    b_rh = lin["RH"] if "RH" in lin else (lin["e"] * float(es_kpa(T_m)) / 100.0 if "e" in lin else None)
    if b_rh is not None:
        res["b_RH_fixed_T"] = quantiles(b_rh)
        res["beta_eff"] = quantiles(P.get("beta", 0.0) + C * b_rh / SEC_PER_H)
    return res


def poly_inputs(inputs, choice):
    """Inputs of the polynomials. RH = 100 e / e_s(T) is a function of T and e, so with the
    TRHe inputs the network output is a function of (T, R, e) alone; using all four inputs
    makes the least-squares problem nearly collinear (T and RH correlate at about -0.9)."""
    if choice == "all" or len(inputs) == 2:
        return inputs
    return {"TRe": ["T", "R", "e"], "TRRH": ["T", "R", "RH"]}[choice]


def distil(tag, season, a, sc, sets, tr, te):
    model = load_model(a.results, season, tag, sc)
    sanity.check_params(model.phys_params_numpy(), model.delta_max if model.white == "moist_closed" else None, tag)
    names_out, kinds = output_kinds(model)
    inputs = list(model.inputs)
    used = poly_inputs(inputs, a.poly_inputs)
    idx = [inputs.index(n) for n in used]
    xtr, ytr = corrections_at(model, *states(tr), kinds)
    xte, yte = corrections_at(model, *states(te), kinds)
    lo, hi = xtr.min(0), xtr.max(0)
    net_sc = scores(model, sets, tr)
    res = {"inputs": inputs, "poly_inputs": used, "outputs": names_out, "network": {"sets": net_sc, **summary(net_sc)},
           "scaler_mean": {n: sc.mean[n] for n in inputs}, "scaler_std": {n: sc.std[n] for n in inputs},
           "target_std_train": dict(zip(names_out, ytr.std(1).mean(0).tolist())), "degrees": {}}
    for deg in ((1, 2, 3, 4) if len(used) == 2 else (1, 2, 3)):
        pf = PolynomialFeatures(deg, include_bias=True).fit(np.zeros((1, len(used))))
        powers = np.zeros((len(pf.powers_), len(inputs)), dtype=int)
        powers[:, idx] = pf.powers_  # unused inputs enter with exponent 0
        names = list(pf.get_feature_names_out(used))
        A = design(xtr, powers)
        coef = np.stack([np.linalg.lstsq(A, ytr[m], rcond=None)[0] for m in range(model.M)])  # (M, P, K)
        fit_tr = np.einsum("np,mpk->mnk", A, coef)
        fit_te = np.einsum("np,mpk->mnk", design(xte, powers, lo, hi), coef)
        entry = {"n_terms": len(names), "terms": names,
                 "r2_train": dict(zip(names_out, np.median(r2(ytr, fit_tr), 0).tolist())),
                 "r2_test_states": dict(zip(names_out, np.median(r2(yte, fit_te), 0).tolist())),
                 "coef": {o: {t: quantiles(coef[:, i, k]) for i, t in enumerate(names)} for k, o in enumerate(names_out)}}
        # share of each term in the correction (mean absolute contribution over the training states,
        # centred inputs, so that it does not depend on the origin of the variables)
        contrib = np.abs(A[None, :, :, None] * coef[:, None]).mean(1)  # (M, P, K)
        entry["mean_abs_contribution"] = {o: dict(zip(names, np.median(contrib[..., k], 0).tolist())) for k, o in enumerate(names_out)}
        for clip in (True, False):
            dm = load_model(a.results, season, tag, sc)
            dm.net = PolyNet(powers, coef, kinds, sc.out, dm.mult_bound, lo, hi, clip=clip)
            if clip:
                s = scores(dm, sets, tr)
                entry["sets"] = s
                entry |= summary(s)
            else:
                sims = {name: simulate(dm, d) for name, d in sets.items()}
                entry["fr_rmse_noclip"] = {name: skill(d["T"].to_numpy(), np.nanmean(sims[name], 0))["rmse"]
                                           for name, d in sets.items()}
                entry["fr_implausible_noclip"] = {name: sanity.implausible_fraction(v) for name, v in sims.items()}
        if deg == 1 and model.correction == "add":
            entry["physical"] = physical_linear(model, coef, names, used, sc, tr)
        res["degrees"][str(deg)] = entry
        print(f"{season} {tag:24s} deg {deg}: R2 train {[round(v, 3) for v in entry['r2_train'].values()]} "
              f"FR in {entry['fr_rmse_in']:.2f} (net {res['network']['fr_rmse_in']:.2f}) "
              f"FR oos {entry['fr_rmse_oos']:.2f} (net {res['network']['fr_rmse_oos']:.2f}) "
              f"SS {entry['ss_pers_oos']:+.2f} (net {res['network']['ss_pers_oos']:+.2f})", flush=True)
    return res


# --------------------------------------------------------------------------- part 2
def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ranking(Xtr, target, t_offset, k=5):
    """Fit degree 4 with T shifted by ``t_offset``; return the k terms of largest mean |contribution|."""
    X = Xtr - np.array([t_offset, 0.0])
    pf = PolynomialFeatures(4, include_bias=False)
    F = pf.fit_transform(X)
    lr = LinearRegression().fit(F, target)
    c = np.abs(F * lr.coef_.ravel()).mean(0)
    order = np.argsort(c)[::-1][:k]
    return [str(pf.get_feature_names_out(["T", "R"])[i]) for i in order], lr


def silva_polynomial(a, season):
    silva = load_script("02_reproduce_silva")
    from scipy.optimize import minimize

    s = silva.SETTINGS[season]
    d, T_obs, R = silva.season_arrays(a.processed, season, 2008)
    n = len(T_obs)
    obj = lambda p: np.mean((T_obs - silva.free_run(T_obs[0], R, p[0], p[1], n)) ** 2)  # noqa: E731
    bounds = [(s["guess"][0] / s["fac"], s["guess"][0] * s["fac"]), (s["guess"][1] / s["fac"], s["guess"][1] * s["fac"])]
    al, ga = minimize(obj, s["guess"], bounds=bounds).x
    X = np.vstack([T_obs, R]).T
    y = silva.fd_derivative(T_obs) - (al * R - ga * T_obs**4)
    m, sx, sy, _ = silva.train_main(X, y, s)
    if s["split"] == "random":
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        Xtr, ytr = X[: int(0.7 * n)], y[: int(0.7 * n)]
    with torch.no_grad():
        net_tr = sy.inverse_transform(m(torch.tensor(sx.transform(Xtr), dtype=torch.float32)).numpy()).ravel()
    out = {"alpha_star": al, "gamma_star": ga, "network": {"season": skill(T_obs, silva.euler_corrected(T_obs[0], R, al, ga, silva.correction_fn(m, sx, sy), n))["rmse"]}}
    years = {yr: silva.season_arrays(a.processed, season, yr) for yr in (2009, 2010)}
    for tname, target in (("fit_to_residuals", ytr), ("fit_to_network", net_tr)):
        poly = make_pipeline(PolynomialFeatures(4, include_bias=False), LinearRegression()).fit(Xtr, target)
        corr = lambda T, R_, p=poly: float(p.predict(np.array([[T, R_]]))[0])  # noqa: E731
        F = poly[0].transform(Xtr)
        lr = poly[1]
        contrib = F * lr.coef_.ravel()
        total = contrib.sum(1) + lr.intercept_
        r = {"season_rmse": skill(T_obs, silva.euler_corrected(T_obs[0], R, al, ga, corr, n))["rmse"],
             "other_years_rmse": {yr: skill(To, silva.euler_corrected(To[0], Ro, al, ga, corr, len(To)))["rmse"] for yr, (_, To, Ro) in years.items()},
             "r2_train": float(1 - np.mean((total - target) ** 2) / np.var(target)),
             "intercept": float(lr.intercept_),
             "coef": dict(zip(poly[0].get_feature_names_out(["T", "R"]), lr.coef_.ravel().tolist())),
             # sum of the absolute term contributions relative to the size of the correction itself
             "cancellation_ratio": float((np.abs(contrib).sum(1) + abs(lr.intercept_)).mean() / np.abs(total).mean()),
             "cond_raw": float(np.linalg.cond(np.column_stack([np.ones(len(F)), F]))),
             "cond_standardised": float(np.linalg.cond(np.column_stack([np.ones(len(F)), PolynomialFeatures(4, include_bias=False).fit_transform(StandardScaler().fit_transform(Xtr))]))),
             "top5_T_in_K": ranking(Xtr, target, 0.0)[0], "top5_T_in_C": ranking(Xtr, target, 273.15)[0]}
        out[tname] = r
        print(f"{season} Silva poly ({tname}): season RMSE {r['season_rmse']:.3f}, other years {r['other_years_rmse']}, "
              f"R2 {r['r2_train']:.3f}, cancellation x{r['cancellation_ratio']:.0f}, cond raw {r['cond_raw']:.2e} "
              f"std {r['cond_standardised']:.2e}\n   top5 K {r['top5_T_in_K']}\n   top5 C {r['top5_T_in_C']}", flush=True)
    return out


def main(a):
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    torch.set_num_threads(a.threads)
    sc = training_scalers(a.processed, a.season)
    sets = test_sets(a.processed, a.season)
    tr, _, te = calibration_split(a.processed, a.season)
    out_dir = a.results / "polynomial"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not a.skip_silva:
        (out_dir / f"{a.season}_silva.json").write_text(json.dumps(silva_polynomial(a, a.season), indent=1, default=float))
    for tag in a.models or MODELS:
        if (a.results / a.season / f"{tag}.pt").exists():
            res = distil(tag, a.season, a, sc, sets, tr, te)
            (out_dir / f"{a.season}_{tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=["dry", "wet"], required=True)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--skip-silva", dest="skip_silva", action="store_true")
    ap.add_argument("--poly-inputs", dest="poly_inputs", choices=["TRe", "TRRH", "all"], default="TRe",
                    help="inputs of the polynomials for models with TRHe inputs")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    main(ap.parse_args())
