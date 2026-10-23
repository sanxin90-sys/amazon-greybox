"""Collect every trained ensemble of one season into comparison tables.

For each model and test set the script reports, from the saved predictions:
  * free-run and 24-h forecast skill (RMSE, centred RMSE, bias, r, KGE, Willmott d);
  * skill of the 24-h forecasts relative to climatology and 24-h persistence,
    evaluated on exactly the samples the forecasts cover;
  * calibration of the Gaussian predictive distribution (ensemble spread with and
    without the validation residual term): 90 % coverage and CRPS;
  * the mean nocturnal cooling 20 h -> 23 h (de Paulo et al. 2023 statistic).

Outputs ``results/tables/<season>_long.csv`` (one row per model x set) and prints
compact tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from amazon_graybox import sanity
from amazon_graybox.experiments import calibration_split, climatology_baseline, persistence_baseline, test_sets
from amazon_graybox.metrics import crps_gaussian, ensemble_predictive, gaussian_interval_coverage, nocturnal_cooling, skill

ORDER = [
    "dry_none", "moist_open_none", "moist_closed_none",
    "dry_add_TR", "dry_add_TRHe", "dry_mult_TRHe", "dry_hyb_TRHe",
    "moist_closed_add_TR", "moist_closed_add_TRHe", "moist_closed_mult_TRHe", "moist_closed_hyb_TRHe",
    "none_add_TRHe",
]


def centred_rmse(obs, pred):
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    m = np.isfinite(o) & np.isfinite(p)
    e = p[m] - o[m]
    return float(np.sqrt(np.mean((e - e.mean()) ** 2)))


def mse_on(obs, pred, mask):
    return float(np.mean((pred[mask] - obs[mask]) ** 2))


def evaluate(season: str, processed: Path, results: Path) -> pd.DataFrame:
    tr, _, _ = calibration_split(processed, season)
    sets = test_sets(processed, season)
    rows = []
    for tag in ORDER:
        jf = results / season / f"{tag}.json"
        if not jf.exists():
            continue
        meta = json.loads(jf.read_text())
        pred = np.load(results / season / f"{tag}_pred.npz")
        sig = meta["sigma_res_val"]
        for name, d in sets.items():
            obs = d["T"].to_numpy()
            sim, fc = pred[f"{name}__sim"].astype(float), pred[f"{name}__fc"].astype(float)
            mu_s, mu_f = np.nanmean(sim, 0), np.nanmean(fc, 0)
            fr, f24 = skill(obs, mu_s), skill(obs, mu_f)
            clim, pers = climatology_baseline(tr, d), persistence_baseline(d)
            mask = np.isfinite(obs) & np.isfinite(mu_f) & np.isfinite(clim) & np.isfinite(pers)
            mse_f = mse_on(obs, mu_f, mask)
            sd_ens = np.nanstd(sim, 0)
            _, sd_tot = ensemble_predictive(sim, sig)
            cool_o = nocturnal_cooling(d.index, obs)
            cool_m = nocturnal_cooling(d.index, mu_s)
            cool_f = nocturnal_cooling(d.index, mu_f)
            j = cool_o.index.intersection(cool_f.index)
            rows.append({
                "season": season, "model": tag, "set": name,
                "fr_implausible": sanity.implausible_fraction(sim), "fc_implausible": sanity.implausible_fraction(fc),
                "fr_rmse": fr["rmse"], "fr_crmse": centred_rmse(obs, mu_s), "fr_bias": fr["bias"], "fr_r": fr["r"],
                "fr_kge": fr["kge"], "fr_d": fr["d"], "fr_nse": fr["r2"],
                "fc_rmse": f24["rmse"], "fc_crmse": centred_rmse(obs, mu_f), "fc_bias": f24["bias"], "fc_r": f24["r"],
                "fc_kge": f24["kge"],
                "fc_ss_clim": 1 - mse_f / mse_on(obs, clim, mask),
                "fc_ss_pers": 1 - mse_f / mse_on(obs, pers, mask),
                "clim_rmse": float(np.sqrt(mse_on(obs, clim, mask))), "pers_rmse": float(np.sqrt(mse_on(obs, pers, mask))),
                "cov90_ens": gaussian_interval_coverage(obs, mu_s, np.maximum(sd_ens, 1e-6)),
                "cov90_tot": gaussian_interval_coverage(obs, mu_s, sd_tot),
                "crps_tot": crps_gaussian(obs, mu_s, sd_tot),
                "spread": float(np.nanmean(sd_ens)),
                "cool_obs": float(cool_o.mean()), "cool_fc": float(cool_f.loc[j].mean()) if len(j) else np.nan,
                "cool_fc_rmse": float(np.sqrt(np.mean((cool_f.loc[j] - cool_o.loc[j]) ** 2))) if len(j) else np.nan,
                "cool_fr": float(cool_m.mean()),
                "tmax_bias": meta["sets"][name]["extremes"]["tmax_bias"], "tmin_bias": meta["sets"][name]["extremes"]["tmin_bias"],
                "n_params_net": meta["n_params_net"], "train_s": meta["train_seconds"],
                "substep4_rmse": meta["sets"][name]["free_run_substep4_rmse"],
            })
    return pd.DataFrame(rows)


def main(a):
    out = Path(a.results) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df = evaluate(a.season, a.processed, Path(a.results))
    df.to_csv(out / f"{a.season}_long.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for col in ("fr_rmse", "fr_crmse", "fc_rmse", "fc_ss_clim", "fc_ss_pers", "cov90_tot", "cool_fc"):
        t = df.pivot(index="model", columns="set", values=col).reindex([m for m in ORDER if m in set(df.model)])
        print(f"\n[{a.season}] {col}\n" + t.round(2).to_string())
    base = df.drop_duplicates("set").set_index("set")[["clim_rmse", "pers_rmse", "cool_obs"]]
    print(f"\n[{a.season}] baselines (forecast-covered samples)\n" + base.round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=["dry", "wet"], required=True)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    main(ap.parse_args())
