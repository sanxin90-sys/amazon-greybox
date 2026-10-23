"""Manuscript tables (JSON for the .docx builder, CSV for the supplement) from the saved results.

    python scripts/08_tables.py --out ../ms/tables.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from amazon_graybox import sanity
from amazon_graybox.experiments import calibration_split, test_sets
from amazon_graybox.physics import RHO_C

SEASONS = ("dry", "wet")
LABEL = {
    "dry_none": "Dry (physics)", "moist_open_none": "Moist open (physics)", "moist_closed_none": "Moist closed (physics)",
    "dry_add_TR": "Dry + additive (T, R)", "dry_add_TRHe": "Dry + additive", "dry_mult_TRHe": "Dry + multiplicative",
    "dry_hyb_TRHe": "Dry + hybrid", "moist_closed_add_TR": "Moist + additive (T, R)", "moist_closed_add_TRHe": "Moist + additive",
    "moist_closed_mult_TRHe": "Moist + multiplicative", "moist_closed_hyb_TRHe": "Moist + hybrid", "none_add_TRHe": "Black box",
}
SETNAME = {"jaru2008_test": "Jaru 2008 test", "jaru2009": "Jaru 2009", "jaru2010": "Jaru 2010", "sinop2002_06": "Sinop Jun 2002",
           "sinop2003_08-09": "Sinop Aug–Sep 2003", "cuieiras2003_08-10": "Cuieiras Aug–Oct 2003",
           "sinop2001_04-05": "Sinop Apr–May 2001", "cuieiras2004_01-03": "Cuieiras Jan–Mar 2004"}
f2 = lambda x: "–" if x is None or not np.isfinite(x) else f"{x:.2f}"  # noqa: E731


def table_data(processed):
    rows = []
    for s in SEASONS:
        tr, va, te = calibration_split(processed, s)
        sets = {"jaru2008_cal": pd.concat([tr, va, te])} | test_sets(processed, s)
        sets.pop("jaru2008_test")
        for k, d in sets.items():
            ok = d[["T", "R", "RH"]].notna().all(axis=1)
            day = (d.index - pd.Timedelta(minutes=30)).normalize()
            dtr = (d["T"].groupby(day).max() - d["T"].groupby(day).min()).mean()
            name = "Jaru 2008 (calibration)" if k == "jaru2008_cal" else SETNAME[k]
            rows.append([s.capitalize(), name, f"{int(ok.sum())}", f"{d['T'].mean() - 273.15:.1f}", f"{dtr:.1f}",
                         f"{d['R'].mean():.0f}", f"{d['RH'].mean():.0f}"])
    return {"label": "Table 1", "caption": "Data blocks used in this study: number of complete half-hourly records (*n*), mean air temperature (*T̄*), mean diurnal temperature range (DTR), mean net radiation (*R̄*) and mean relative humidity (RH). Rebio Jaru 2008 is split chronologically into training (70%), validation (15%) and test (15%) blocks; all other blocks are independent.",
            "header": ["Season", "Block", "*n*", "*T̄* (°C)", "DTR (K)", "*R̄* (W m⁻²)", "RH (%)"],
            "widths": [10, 30, 10, 12, 12, 14, 12], "rows": rows,
            "note": "Rebio Jaru 2009: temperature about 3 K lower than in the other years, suspected instrumental offset (Section 2.1)."}


def table_params(res):
    rows = []
    for s in SEASONS:
        for w, lab in (("dry", "Dry"), ("moist_open", "Moist open"), ("moist_closed", "Moist closed")):
            P = json.loads((res / s / f"phys_{w}.json").read_text())["params"]
            med = {k: float(np.median(v)) for k, v in P.items()}
            sd = {k: float(np.std(v)) for k, v in P.items()}
            r = json.loads((res / s / f"{w}_none.json").read_text())
            cell = lambda k, sc, nd=2: (f"{med[k] * sc:.{nd}f} ± {sd[k] * sc:.{nd}f}" if k in med else "–")  # noqa: E731
            rows.append([s.capitalize(), lab, cell("alpha", 1e3), cell("gamma", 1e4), cell("beta", 1e3), cell("delta", 1, 0),
                         f2(r["sets"]["jaru2008_test"]["free_run"]["rmse"])])
    rows.append(["Ref.", "Dry [depaulo2023]", "2.70", "6.50", "–", "–", "–"])
    rows.append(["Ref.", "Moist open, Jaru [maionchi2025]", "0.55", "6.50", "2.70", "154", "–"])
    return {"label": "Table 3", "caption": "Physical parameters of the physics-only models calibrated on Rebio Jaru 2008 (median ± standard deviation over the ten ensemble members) and free-run RMSE (K) on the 2008 test block. Published values are given for reference.",
            "header": ["Season", "Model", "α (10⁻³ m⁻¹)", "γ (10⁻⁴ m⁻¹)", "β (10⁻³ W m⁻³ %⁻¹)", "δ (J m⁻³ %⁻¹)", "RMSE (K)"],
            "widths": [9, 26, 14, 14, 15, 13, 9], "rows": rows,
            "note": "The closed-model δ is bounded by δ_max = 167.6 J m⁻³ %⁻¹ (Equation (4))."}


def table_silva(res):
    rows = []
    pub = {"dry": {"a": "1.1588", "g": "2.05", "u": "3.66 (3.62)", "c": "1.40", "e": "7.42", "cov": "0.94", "w": "26.9"},
           "wet": {"a": "0.2710", "g": "0.375", "u": "2.57 (2.54)", "c": "1.13", "e": "7.78", "cov": "0.96", "w": "79.1"}}
    for s in SEASONS:
        f = res / f"silva_protocol_{s}.json"
        if not f.exists():
            continue
        z = json.loads(f.read_text())
        rows.append([f"{s.capitalize()} season"])
        rows.append(["Published [silva2026]", pub[s]["a"], pub[s]["g"], pub[s]["u"], pub[s]["c"], pub[s]["e"], pub[s]["cov"], pub[s]["w"]])
        u, o, cl = z["main_model_season"], z["unit10_open_loop"], z["closed_loop_ensemble"]
        rows.append(["Re-implementation (ensemble open loop)", f"{z['alpha_star']:.4f}", f"{z['gamma_star'] * 1e8:.3g}",
                     f"{z['rmse_uncorrected_rk4']:.2f}", f"{u['rmse']:.2f}", f"{o['ensemble_mean']['rmse']:.2f}",
                     f"{o['coverage90']:.2f}", f"{o['mean_band_width_K']:.1f}"])
        rows.append(["Same members, closed loop", "–", "–", "–", "–", f"{cl['ensemble_mean']['rmse']:.2f}",
                     f"{cl['coverage90']:.2f}", f"{cl['mean_band_width_K']:.1f}"])
    return {"label": "Table 2", "caption": "Re-implementation of the protocol of Silva et al. [silva2026] on Rebio Jaru 2008 (whole season, as in the original study): fitted parameters (α^{*} in K m² W⁻¹ per 10⁶ s; γ^{*} in 10⁻⁸ K⁻³ per 10⁶ s), RMSE (K) of the uncorrected (RK4) and corrected (Euler) free runs, and RMSE of the mean, coverage of the 5–95% band and mean band width (K) of the 100-member bootstrap ensemble integrated open loop (as in the original study) and of 20 members integrated closed loop.",
            "header": ["Version", "α^{*}", "γ^{*} (10⁻⁸)", "Physics", "Corrected", "RMSE of mean", "Coverage", "Width (K)"],
            "group": ["", "Parameters", "", "Single-model RMSE", "", "Bootstrap ensemble", "", ""],
            "widths": [21, 8, 9, 10, 12, 10, 12, 10], "rows": rows,
            "note": "Published uncorrected RMSE from the parameter table (and from the text, in parentheses); published band widths from Tables X and XI of Silva et al. [silva2026]. The re-implementation uses the authors' notebook logic with the data prepared in Section 2.1, which differs slightly from the spreadsheets used in the original study."}


def table_scores(res):
    """In-sample (Jaru 2008 test) and mean out-of-sample scores."""
    rows = []
    for s in SEASONS:
        df = pd.read_csv(res / "tables" / f"{s}_long.csv")
        rows.append([f"{s.capitalize()} season"])
        for tag in LABEL:
            d = df[df.model == tag]
            if d.empty:
                continue
            sanity.require("fr_implausible" in d, "rerun scripts/04_evaluate.py: plausibility flags missing")
            ins = d[d.set == "jaru2008_test"].iloc[0]
            oos = d[~d.set.isin(["jaru2008_test", "jaru2009"])]
            j9 = d[d.set == "jaru2009"].iloc[0]
            mark = lambda x: "ᵈ" if (x.fr_implausible > 0).any() else ""  # noqa: E731
            rows.append([LABEL[tag], f2(ins.fr_rmse) + mark(d[d.set == "jaru2008_test"]), f2(ins.fc_rmse), f2(ins.fr_kge),
                         f2(oos.fr_rmse.mean()) + mark(oos), f2(oos.fc_rmse.mean()),
                         f2(1 - (oos.fc_rmse ** 2).mean() / (oos.pers_rmse ** 2).mean()), f2(j9.fr_crmse) + mark(d[d.set == "jaru2009"]),
                         f2(oos.cov90_tot.mean()), f2(oos.crps_tot.mean())])
    return {"label": "Table 4", "caption": "Skill of all models. In sample: Rebio Jaru 2008 test block (free-run RMSE, 24-h forecast RMSE and Kling–Gupta efficiency of the free run). Out of sample: means over Jaru 2010, Sinop and Cuieiras of the free-run RMSE, the 24-h forecast RMSE, the forecast skill score relative to 24-h persistence (pooled over the sets), the 90% coverage and the CRPS of the free run; Jaru 2009 is reported separately with the centred RMSE of the free run. RMSE and CRPS in K.",
            "header": ["Model", "FR RMSE", "FC RMSE", "FR KGE", "FR RMSE", "FC RMSE", "SS_{pers}", "J09 cRMSE", "Cov. 90%", "CRPS"],
            "group": ["", "In sample", "", "", "Out of sample", "", "", "", "", ""],
            "widths": [25, 8, 8, 8, 8, 8, 8, 9, 8, 8], "size": 16, "rows": rows,
            "note": "FR: free run; FC: 24-h forecasts initialised daily at 00:30 local time; SS_{pers} = 1 − MSE/MSE_{pers}. ᵈ The free run left the plausible range of 250–340 K in at least one set (diverged, RMSE above 10 K); that set also enters the out-of-sample coverage and CRPS. Mean baselines on the out-of-sample sets (dry/wet): persistence RMSE BASE_PERS K, climatology RMSE BASE_CLIM K."}


def table_proscons():
    """Qualitative synthesis; every statement is backed by a number in Section 3."""
    rows = [
        ["Dry physics [depaulo2023]", "Two parameters with a radiative meaning; analytical nocturnal solution",
         "No humidity dependence; relaxation time of 190–340 h at 300 K, against 3–7 h for the closed moist and corrected models, so free runs "
         "can drift by more than 10 K; 24-h forecasts less accurate than climatology on the test block"],
        ["Open moist physics [maionchi2025]", "Represents the humidity dependence of cooling; most accurate 24-h forecasts in the wet season, "
         "probably owing to the leakage",
         "Forced by the observed humidity tendency, which contains the observed temperature tendency; unstable in free run out of sample "
         "in the dry season; δ near or beyond the well-posedness bound"],
        ["Closed moist physics (this study)", "Four parameters; stable in every block; lowest mean out-of-sample free-run error of the "
         "physics-only models; reproduces most of the humidity dependence of nocturnal cooling in the wet season and a weaker version of it in the dry season; relaxation time similar to that learned by the "
         "corrections; parameters can be re-estimated from a few days of data at a new site",
         "Restoring relies mainly on the humidity feedback, so the accuracy depends on an accurate vapour pressure; δ poorly identified in the wet season; "
         "physical meaning of the humidity term unresolved"],
        ["Additive correction", "Simplest coupling; improved the mean out-of-sample skill of both white boxes; its degree-1 distillation "
         "is an explicit equation (linear relaxation and stronger radiative heating)",
         "The correction can override the physics; physical reading only after distillation"],
        ["Multiplicative correction", "Bounded modulation of the physical tendency; lowest mean out-of-sample free-run error in the wet "
         "season when applied to the closed moist model",
         "Did not prevent the drift of an unstable white box outside the training range (divergence of the dry model at Jaru 2009, "
         "wet season); its degree-1 distillation diverged for the dry model"],
        ["Hybrid correction", "State-dependent physical coefficients; the logarithm of the absorption factor is close to a linear function of "
         "the inputs, dominated by air temperature (R² of 0.91–0.95 at degree 1 in the dry season); physical parameters alone could be adapted to new sites",
         "Most outputs; no consistent accuracy gain over the additive form; modulation and residual terms may trade off (not tested here)"],
        ["Black-box ODE-net", "Accuracy similar to that of the gray boxes, without a physical prior; its distillation gives a "
         "relaxation rate similar to that of the corrected dry model",
         "No interpretable parameters; smallest response to a change of radiation; full fine-tuning degraded its free run at Cuieiras "
         "(wet season)"],
        ["Protocol of Silva et al. [silva2026]", "Simple and fast; reproduces the published parameters; its polynomial reproduces the "
         "network trajectories",
         "Scores computed on the training season; open-loop bootstrap bands tens of kelvin wide; polynomial terms in unscaled variables "
         "largely cancel, and their ranking depends on the origin of the temperature scale (kelvin or degrees Celsius)"],
    ]
    return {"label": "Table 8", "caption": "Strengths and weaknesses of the model classes and of the reference protocol, as found in this study.",
            "header": ["Model", "Strengths", "Weaknesses"], "widths": [22, 38, 40], "align": "left", "rows": rows}


POLY_MODELS = ["dry_add_TR", "dry_add_TRHe", "dry_mult_TRHe", "dry_hyb_TRHe", "moist_closed_add_TR", "moist_closed_add_TRHe",
               "moist_closed_mult_TRHe", "moist_closed_hyb_TRHe", "none_add_TRHe"]
DIVERGED = sanity.DIVERGED_RMSE  # K: free-run RMSE above which a run is reported as diverged


def poly_results(res):
    out = {}
    for s in SEASONS:
        for tag in POLY_MODELS:
            f = res / "polynomial" / f"{s}_{tag}.json"
            if f.exists():
                out[s, tag] = json.loads(f.read_text())
    return out


def poly_summary_csv(res, P):
    """Every degree, with and without clipping, for the supplement."""
    rows = []
    for (s, tag), z in P.items():
        rows.append({"season": s, "model": tag, "degree": "network", "n_terms": None, "r2_train_min": None,
                     **{k: z["network"][k] for k in ("fr_rmse_in", "fc_rmse_in", "fr_rmse_oos", "fc_rmse_oos", "ss_pers_oos")}})
        for deg, e in z["degrees"].items():
            rows.append({"season": s, "model": tag, "degree": deg, "n_terms": e["n_terms"], "r2_train_min": min(e["r2_train"].values()),
                         **{k: e[k] for k in ("fr_rmse_in", "fc_rmse_in", "fr_rmse_oos", "fc_rmse_oos", "ss_pers_oos")},
                         "n_sets_diverged_clipped": sum(sanity.diverged(v["fr_rmse"], v["fr_implausible"]) for v in e["sets"].values()),
                         "n_sets_diverged_unclipped": sum(sanity.diverged(v, e["fr_implausible_noclip"][k])
                                                          for k, v in e["fr_rmse_noclip"].items()),
                         "n_sets": len(e["sets"])})
    df = pd.DataFrame(rows)
    df.to_csv(res / "tables" / "polynomial_summary.csv", index=False)
    return df


def poly_diverged(e, which: str) -> bool:
    """sanity.diverged in the test block (which="in") or in any out-of-sample set (which="oos")."""
    sets = [v for k, v in e["sets"].items() if (k == "jaru2008_test" if which == "in" else k not in ("jaru2008_test", "jaru2009"))]
    return any(sanity.diverged(v["fr_rmse"], v["fr_implausible"]) for v in sets) or sanity.diverged(e[f"fr_rmse_{which}"])


def table_poly(res, P):
    def rm(e, which):
        return "div." if poly_diverged(e, which) else f2(e[f"fr_rmse_{which}"])

    def ss(e):
        bad = poly_diverged(e, "oos") or e["fc_rmse_oos"] > DIVERGED
        return "div." if bad else f2(e["ss_pers_oos"])

    rows = []
    for s in SEASONS:
        rows.append([f"{s.capitalize()} season"])
        for tag in POLY_MODELS:
            if (s, tag) not in P:
                continue
            z = P[s, tag]
            n, d1, d2 = z["network"], z["degrees"]["1"], z["degrees"]["2"]
            rows.append([LABEL[tag], f2(n["fr_rmse_in"]), f2(n["fr_rmse_oos"]), f2(n["ss_pers_oos"]),
                         f2(min(d1["r2_train"].values())), rm(d1, "in"), rm(d1, "oos"), ss(d1),
                         f2(min(d2["r2_train"].values())), rm(d2, "in"), rm(d2, "oos"), ss(d2)])
    return {"label": "Table 5", "caption": "Polynomial distillation of the learned corrections. Each member's network is replaced by polynomials of degree 1 or 2 in its standardised inputs ((*T*, *R*), or (*T*, *R*, *e*) for the networks with humidity inputs), fitted to the network output at the observed states of the training block; the distilled ensembles are integrated with the same scheme as the networks. R²: share of the variance of the network output explained by the polynomial on the training states (lowest value over the outputs of the hybrid form). FR RMSE: free-run RMSE (K) on the Rebio Jaru 2008 test block (in sample) and mean over Jaru 2010, Sinop and Cuieiras (out of sample); SS_{pers}: pooled skill score of the 24-h forecasts relative to persistence on the same out-of-sample sets.",
            "header": ["Model", "FR in", "FR out", "SS_{pers}", "R²", "FR in", "FR out", "SS_{pers}", "R²", "FR in", "FR out", "SS_{pers}"],
            "group": ["", "Network", "", "", "Degree 1", "", "", "", "Degree 2", "", "", ""],
            "widths": [22, 7, 7, 7.5, 6.5, 7, 7, 7.5, 6.5, 7, 7, 7.5], "size": 15, "rows": rows,
            "note": "div.: the free run diverged in at least one set (RMSE above 10 K, or a member left the plausible range of 250–340 K), or the mean forecast RMSE exceeded 10 K; the skill score is then not given. Inputs of the polynomials are clipped to the range of the training block. Results for degrees 3 and 4 and without clipping are given in the supplementary file polynomial_summary.csv."}


def table_polycoef(res, P):
    """Degree-1 additive corrections in physical units (medians over the ten members)."""
    rows, means = [], {}
    for s in SEASONS:
        rows.append([f"{s.capitalize()} season"])
        for tag in ("dry_add_TR", "dry_add_TRHe", "moist_closed_add_TR", "moist_closed_add_TRHe", "none_add_TRHe"):
            if (s, tag) not in P:
                continue
            ph = P[s, tag]["degrees"]["1"]["physical"]
            means[s] = ph["mean_state"]
            q = lambda k, sc=1.0, nd=2: f"{ph[k]['median'] * sc:.{nd}f}" if k in ph else "–"  # noqa: E731
            if "restoring_correction_h" in ph:
                lam_n, tau = ph["restoring_correction_h"], ph["tau_total_h"]
            else:  # black box: the whole tendency is distilled; its R coefficient defines an equivalent alpha
                lam_n = {k: -ph["b_T"][k] for k in ("median", "p05", "p95")}
                tau = {"median": 1.0 / lam_n["median"]}
                ph = ph | {"alpha_eff": {"median": RHO_C * ph["b_R"]["median"] / 3600.0}}
            rng = lambda d, nd: f"{d['median']:.{nd}f} ({min(d['p05'], d['p95']):.{nd}f}–{max(d['p05'], d['p95']):.{nd}f})"  # noqa: E731
            rows.append([LABEL[tag], q("a0_K_h"), q("b_R", 1e3), q("b_e") if "b_e" in ph else "–",
                         q("alpha", 1e3), q("alpha_eff", 1e3), q("restoring_white_h", 1.0, 3), rng(lam_n, 3), f"{tau['median']:.1f}"])
    ms = " ".join(f"{s}: T̄ = {m['T'] - 273.15:.1f} °C, R̄ = {m['R']:.0f} W m⁻², ē = {m['e']:.2f} kPa;" for s, m in means.items())
    return {"label": "Table 6", "caption": "Degree-1 distillation of the additive corrections (Equation (6)) in physical units (medians over the ten members). α: absorption coefficient of the white box trained with the correction; α_{eff} = α + C b_{R}/3600, with C = ρc for the dry model and the effective heat capacity of Equation (4) for the closed moist model; λ_{W}: linear restoring rate of the white box at the mean state, 3600 (4γσT̄³ + β RH d ln e_{s}/dT)/C, with the humidity part only for the closed moist model; λ_{N} = −b_{T}: restoring rate added by the correction at fixed vapour pressure (5–95% range over the members); τ = 1/(λ_{W} + λ_{N}): total relaxation time.",
            "header": ["Model", "a₀ (K h⁻¹)", "b_{R} (10⁻³ K h⁻¹ W⁻¹ m²)", "b_{e} (K h⁻¹ kPa⁻¹)", "α (10⁻³ m⁻¹)",
                       "α_{eff} (10⁻³ m⁻¹)", "λ_{W} (h⁻¹)", "λ_{N} (h⁻¹)", "τ (h)"],
            "widths": [21, 9, 12, 11, 9, 10, 9, 13, 6], "size": 15, "rows": rows,
            "note": "Mean states of the training blocks (" + ms.rstrip(";") + "). For the black box the whole tendency is distilled: a₀ is its value at the mean state, λ_{N} its restoring rate and α_{eff} = ρc b_{R}/3600."}


TERM = {"c": "c (K h⁻¹)", "ln_m": "ℓ_{m}", "ln_m_alpha": "ℓ_{α}", "ln_m_gamma": "ℓ_{γ}", "ln_m_beta": "ℓ_{β}", "c0": "c₀ (K h⁻¹)"}


def table_explicit(res):
    """Coefficients and free-run errors of the explicit equations (scripts/11_explicit_equations.py)."""
    E = json.loads((res / "polynomial" / "explicit_equations.json").read_text())
    f3 = lambda x: f"{x:.3f}"  # noqa: E731
    def err(e):
        bad = any(sanity.diverged(v, e["implausible"][n]) for n, v in e["sets"].items() if n != "jaru2009")
        return "div." if bad or sanity.diverged(e["fr_rmse_oos"]) else f"{e['fr_rmse_in']:.2f} / {e['fr_rmse_oos']:.2f}"

    rows = []
    for s in SEASONS:
        rows.append([f"{s.capitalize()} season"])
        for tag, e in E[s].items():
            sanity.require("implausible" in e, "rerun scripts/11_explicit_equations.py: plausibility flags missing")
            p = e["params"]
            phys = [f"{p['alpha'] * 1e3:.2f}", f"{p['gamma'] * 1e4:.2f}",
                    f"{p['beta'] * 1e3:.2f}" if "beta" in p and e["white"] != "dry" else "–",
                    f"{p['delta']:.0f}" if "delta" in p and e["white"] != "dry" else "–"]
            for k, (o, t) in enumerate(e["terms"].items()):
                lin = [f3(t["centre"]), f3(t["T"]), f3(t["R"] * 100.0), f3(t["e"]) if "e" in t else "–"]
                rows.append([LABEL[tag] if k == 0 else "", TERM[o], *lin, *(phys if k == 0 else [""] * 4), err(e) if k == 0 else ""])
    return {"label": "Table 7", "caption": "Coefficients of the explicit equations (7)–(12) (degree-1 distillation; medians over the ten members). ℓ: logarithm of a multiplicative or hybrid factor; c and c₀: additive correction and residual. Each is linear about the mean state of the training block (Equation (13)): ℓ̄ is its value at the mean state and the next three columns are its slopes per unit of T, R and e. α, γ, β and δ: physical parameters of the trained model. RMSE: free-run RMSE (K) of the single explicit equation on the 2008 test block / mean over the independent sets (Jaru 2009 excluded).",
            "header": ["Model", "Term", "ℓ̄ or c̄", "per K", "per 100 W m⁻²", "per kPa", "α (10⁻³ m⁻¹)", "γ (10⁻⁴ m⁻¹)",
                       "β (10⁻³ W m⁻³ %⁻¹)", "δ (J m⁻³ %⁻¹)", "RMSE (K)"],
            "widths": [15, 9, 8, 8, 9, 8, 8, 8, 9, 8, 10], "size": 14, "rows": rows,
            "note": "Mean states as in Table 6. div.: free-run RMSE above 10 K, or a simulated temperature outside 250–340 K, in the test block or an independent set. The inputs are not clipped. The dry model with the additive correction and inputs (T, R) is Equation (7)."}


def baselines(res):
    out = {}
    for s in SEASONS:
        df = pd.read_csv(res / "tables" / f"{s}_long.csv")
        d = df[(df.model == "moist_closed_none") & ~df.set.isin(["jaru2008_test", "jaru2009"])]
        out[s] = (d.pers_rmse.mean(), d.clim_rmse.mean())
    return out


def main(a):
    res = Path(a.results)
    T = {"data": table_data(a.processed), "params": table_params(res), "scores": table_scores(res), "proscons": table_proscons()}
    b = baselines(res)
    T["scores"]["note"] = T["scores"]["note"].replace("BASE_PERS", f"{b['dry'][0]:.2f}/{b['wet'][0]:.2f}").replace(
        "BASE_CLIM", f"{b['dry'][1]:.2f}/{b['wet'][1]:.2f}")
    if (res / "silva_protocol_dry.json").exists():
        T["silva"] = table_silva(res)
    P = poly_results(res)
    if P:
        poly_summary_csv(res, P)
        T["poly"], T["polycoef"] = table_poly(res, P), table_polycoef(res, P)
    if (res / "polynomial" / "explicit_equations.json").exists():
        T["explicit"] = table_explicit(res)
    for t in T.values():  # typographic minus in numeric cells
        t["rows"] = [[re.sub(r"(?<![\w])-(?=\d)", "−", str(c)) for c in r] for r in t["rows"]]
    Path(a.out).write_text(json.dumps(T, ensure_ascii=False, indent=1))
    for k, t in T.items():
        print(f"== {k}: {t['caption'][:60]}")
        print(" | ".join(t["header"]))
        for r in t["rows"]:
            print(" | ".join(map(str, r)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables.json")
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    main(ap.parse_args())
