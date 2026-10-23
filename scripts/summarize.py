"""Print a one-line summary per trained ensemble of one season (development helper).

    python scripts/summarize.py dry
"""
import glob
import json
import sys

import numpy as np

from amazon_graybox import sanity

for f in sorted(glob.glob(f"results/{sys.argv[1]}/*.json")):
    if "phys_" in f:
        continue
    r = json.load(open(f))
    p = {k: np.median(v) for k, v in r["params"].items()}
    sanity.check_params(r["params"], name=r["tag"])
    ps = " ".join(f"{k}={v:.3g}" for k, v in p.items())
    print(f"== {r['tag']} ({r['epochs_run']} ep, {r['train_seconds']:.0f}s, valRMSE {np.mean(r['best_val_rmse']):.2f}, "
          f"sig_res {r['sigma_res_val']:.2f}) {ps}")
    for s, v in r["sets"].items():
        fr, fc, ex = (sanity.check_skill(v["free_run"], s), sanity.check_skill(v["forecast24"], s), v["extremes"])
        print(f"   {s:20s} FR rmse {fr['rmse']:6.2f} bias {fr['bias']:6.2f} r2 {fr['r2']:6.2f} | FC24 rmse {fc['rmse']:5.2f} "
              f"r2 {fc['r2']:5.2f} | Tmax {ex['tmax_bias']:5.2f} Tmin {ex['tmin_bias']:5.2f} | spread {v['spread_mean']:.2f}"
              + (f" | clim {v['climatology']['rmse']:.2f}" if "climatology" in v else ""))
