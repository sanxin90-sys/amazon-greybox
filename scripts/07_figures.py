"""Figures of the manuscript (PNG, 300 dpi; optionally PDF/TIFF/SVG/EPS with --formats) from the saved results.

Comparisons of simulations with observations show a 95% interval and print RMSE, MAE and
R2 = 1 - SSE/SST in the legend; residuals are plotted below the time series. Every stored
trajectory is checked against the physical bounds of ``amazon_graybox.sanity`` when loaded.

    python scripts/07_figures.py --out figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.lines  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from amazon_graybox import sanity  # noqa: E402
from amazon_graybox.experiments import load_model, test_sets, training_scalers, calibration_split  # noqa: E402
from amazon_graybox.graybox import TORCH_DTYPE  # noqa: E402
from amazon_graybox.metrics import ensemble_predictive, skill  # noqa: E402
from amazon_graybox.training import FORCING_COLS, block_arrays  # noqa: E402

# Light-mode reference palette (dataviz skill); colours follow the entity in every figure.
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
C = {"obs": INK, "dry_none": "#eb6834", "moist_open_none": "#eda100", "moist_closed_none": "#1baf7a",
     "graybox": "#2a78d6", "none_add_TRHe": "#e87ba4", "silva": "#6250d6"}
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0b2a8", "#f0efec", "#9ec5f4", "#256abf"])
LABEL = {
    "dry_none": "Dry (physics)", "moist_open_none": "Moist open (physics)", "moist_closed_none": "Moist closed (physics)",
    "dry_add_TR": "Dry + add (T,R)", "dry_add_TRHe": "Dry + add", "dry_mult_TRHe": "Dry + mult", "dry_hyb_TRHe": "Dry + hybrid",
    "moist_closed_add_TR": "Moist + add (T,R)", "moist_closed_add_TRHe": "Moist + add", "moist_closed_mult_TRHe": "Moist + mult",
    "moist_closed_hyb_TRHe": "Moist + hybrid", "none_add_TRHe": "Black box",
}
SETLABEL = {"jaru2008_test": "Jaru 2008 (test)", "jaru2009": "Jaru 2009*", "jaru2010": "Jaru 2010",
            "sinop2002_06": "Sinop Jun 2002", "sinop2003_08-09": "Sinop Aug–Sep 2003", "cuieiras2003_08-10": "Cuieiras Aug–Oct 2003",
            "sinop2001_04-05": "Sinop Apr–May 2001", "cuieiras2004_01-03": "Cuieiras Jan–Mar 2004"}
SEASON = {"dry": "Dry season (Jul–Sep)", "wet": "Wet season (Jan–Mar)"}

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 100, "lines.linewidth": 1.2,
})


def load_pred(res: Path, season: str, tag: str, key: str) -> np.ndarray:
    """Stored ensemble trajectories (M, N) [K]."""
    return np.load(res / season / f"{tag}_pred.npz")[key].astype(float)


def flags(obs, pred) -> str:
    """RMSE, MAE and R2 for a legend, or "diverged" (sanity.diverged)."""
    s = skill(obs, pred)
    if sanity.diverged(s["rmse"], sanity.implausible_fraction(pred)):
        return "diverged"
    return f"RMSE {s['rmse']:.2f} K, MAE {s['mae']:.2f} K, R$^2$ {s['r2']:.2f}".replace("-", "\u2212")


FORMATS = ["png"]  # --formats; journal submission: pdf (vector) and tiff (600 dpi, LZW)


def save(fig, out: Path, name: str):
    for fmt in FORMATS:
        extra = {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}} if fmt == "tiff" else {}
        fig.savefig(out / f"{name}.{fmt}", bbox_inches="tight", facecolor=SURF, **extra)
    plt.close(fig)
    print("wrote", ", ".join(str(out / f"{name}.{fmt}") for fmt in FORMATS))


# ---------------------------------------------------------------- Figure 1
def fig_schematic(out):
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_axis_off(); ax.set_xlim(0, 12.4); ax.set_ylim(-0.6, 4.4)

    def box(x, y, w, h, title, body, edge=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=SURF, ec=edge, lw=1.0))
        ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=7.5, weight="bold")
        ax.text(x + w / 2, y + h - 0.52, body, ha="center", va="top", fontsize=6.6, color=INK2, linespacing=1.45)

    def arrow(x0, y0, x1, y1, **kw):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=INK2, **kw))

    box(0.05, 0.6, 2.2, 3.2, "Inputs", "Forcing (observed)\nnet radiation R\nvapour pressure e\nand its tendency\n\nState (simulated)\nair temperature T\nRH = 100 e / e_s(T)")
    box(2.85, 2.35, 3.3, 1.75, "White box f_W(T, F; p)",
        "dry:  (αR − γσT⁴) / ρc\nclosed moist:  (αR − γσT⁴ + βRH\n− 100δ ė/e_s) / C_eff", edge=C["moist_closed_none"])
    box(2.85, 0.3, 3.3, 1.75, "Network N(z)", "z = (T, R) or (T, R, RH, e)\ntwo tanh layers (32, 16)\nsmall initial output", edge=C["graybox"])
    box(6.75, 0.6, 2.55, 3.2, "Coupling", "additive\nf_W + s·N\n\nmultiplicative\nf_W · exp(1.5 tanh N)\n\nhybrid\nα, γ, β × exp(1.5 tanh N_k)\n+ s·N₀")
    box(9.9, 0.6, 2.45, 3.2, "RK4 ODE-net", "dT/dt integrated\nevery 30 min\n\n48-h trajectory loss\nphysics and network\ntrained jointly\n\nensemble of 10\n(bootstrap weights)")
    arrow(2.25, 2.9, 2.85, 3.2); arrow(2.25, 1.5, 2.85, 1.2)
    arrow(6.15, 3.2, 6.75, 2.7); arrow(6.15, 1.2, 6.75, 1.7); arrow(9.3, 2.2, 9.9, 2.2)
    ax.add_patch(FancyArrowPatch((11.1, 0.6), (1.15, 0.6), connectionstyle="bar,fraction=-0.06", arrowstyle="-|>",
                                 mutation_scale=9, lw=0.9, color=INK2))
    ax.text(6.2, -0.45, "T(t + Δt) fed back as the new state (closed loop, no observed temperature after initialisation)",
            ha="center", fontsize=6.6, color=INK2)
    save(fig, out, "Figure1_schematic")


# ---------------------------------------------------------------- Figure 3
def fig_timeseries(out, res, processed):
    fig = plt.figure(figsize=(7.0, 8.2))
    outer = fig.add_gridspec(2, 1, hspace=0.42, left=0.09, right=0.99, top=0.9, bottom=0.04)
    best = "moist_closed_hyb_TRHe"
    for k, season in enumerate(("dry", "wet")):
        inner = outer[k].subgridspec(2, 1, height_ratios=[3, 1.2], hspace=0.06)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)
        d = test_sets(processed, season)["jaru2008_test"]
        t, obs = d.index, d["T"].to_numpy()
        ax.plot(t, obs - 273.15, color=C["obs"], lw=1.4, label="Observed")
        for tag in ("dry_none", "moist_closed_none", best):
            col = C["graybox"] if tag == best else C[tag]
            sim = load_pred(res, season, tag, "jaru2008_test__sim")
            mu = np.nanmean(sim, 0)
            if tag == best:  # Gaussian predictive distribution of Section 2.7: ensemble spread + validation residual
                sig = json.loads((res / season / f"{tag}.json").read_text())["sigma_res_val"]
                _, sd = ensemble_predictive(sim, sig)
                lo, hi = mu - 1.96 * sd, mu + 1.96 * sd
                cov = np.mean(((obs >= lo) & (obs <= hi))[np.isfinite(obs) & np.isfinite(mu)])
                ax.fill_between(t, lo - 273.15, hi - 273.15, color=col, alpha=0.22, lw=0,
                                label=f"95% prediction interval, {LABEL[best]} (coverage {cov:.2f})")
            ax.plot(t, mu - 273.15, color=col, lw=1.0, label=f"{LABEL[tag]}: {flags(obs, mu)}")
            axr.plot(t, mu - obs, color=col, lw=0.8)
        axr.axhline(0, color=INK2, lw=0.6)
        ax.set_ylabel("Air temperature (°C)")
        axr.set_ylabel("Residual (K)")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6.5, handlelength=1.6, borderaxespad=0.2,
                  title=f"{SEASON[season]}, Rebio Jaru 2008 test block, free run", title_fontsize=8, alignment="left")
        ax.margins(x=0)
        axr.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=2))
        axr.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
        ax.tick_params(labelbottom=False)
    save(fig, out, "Figure3_timeseries")


# ---------------------------------------------------------------- Figure 4
def fig_heatmaps(out, res):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), gridspec_kw={"width_ratios": [1, 1]})
    for row, season in enumerate(("dry", "wet")):
        df = pd.read_csv(res / "tables" / f"{season}_long.csv")
        models = [m for m in LABEL if m in set(df.model)]
        sets = list(dict.fromkeys(df.set))
        for col, (var, title, cmap, norm) in enumerate((
            ("fr_crmse", "Free run: centred RMSE (K)", SEQ, None),
            ("fc_ss_pers", "24-h forecast: skill vs persistence", DIV, TwoSlopeNorm(0, -1.5, 1.0)),
        )):
            ax = axes[row, col]
            M = df.pivot(index="model", columns="set", values=var).reindex(index=models, columns=sets)
            vals = M.to_numpy()
            if norm is None:
                im = ax.imshow(np.clip(vals, 0, 5), cmap=cmap, vmin=1, vmax=5, aspect="auto")
            else:
                im = ax.imshow(np.clip(vals, -1.5, 1.0), cmap=cmap, norm=norm, aspect="auto")
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    v = vals[i, j]
                    dark = (norm is None and v > 3.2) or (norm is not None and v > 0.55)
                    ax.text(j, i, f"{v:.1f}" if norm is None else f"{v:.2f}", ha="center", va="center", fontsize=6.3,
                            color="white" if dark else INK)
            ax.set_xticks(range(len(sets)), [SETLABEL[s] for s in sets], rotation=35, ha="right", fontsize=6.5)
            ax.set_yticks(range(len(models)), [LABEL[m] for m in models] if col == 0 else [], fontsize=6.8)
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{SEASON[season]}: {title}", loc="left", fontsize=7.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
    fig.text(0.01, 0.005, "* Jaru 2009: suspected temperature offset (Section 2.1). Values clipped to the colour range; numbers are exact.",
             fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, out, "Figure4_skill_matrix")


# ---------------------------------------------------------------- Figure 5
def fig_cooling_vs_rh(out, res, processed):
    tags = ["dry_none", "moist_closed_none", "moist_closed_hyb_TRHe"]
    bins = np.arange(55, 101, 5)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, season in zip(axes, ("dry", "wet")):
        sets = test_sets(processed, season)
        rows = {k: [] for k in ["obs"] + tags}
        rh_all, night_id = [], []
        for name, d in sets.items():
            night = np.asarray((d.index.hour >= 19) | (d.index.hour < 5))
            obs = d["T"].to_numpy()
            fcs = {t: np.nanmean(load_pred(res, season, t, f"{name}__fc"), 0) for t in tags}
            # tendency over the next half hour, from the forecast trajectories and from observations
            dobs = (np.roll(obs, -1) - obs) * 2
            rh = d["RH"].to_numpy()
            ok = night & np.isfinite(dobs) & np.isfinite(rh) & np.all([np.isfinite(np.roll(fcs[t], -1) - fcs[t]) for t in tags], 0)
            ok[-1] = False
            rh_all.append(rh[ok]); rows["obs"].append(dobs[ok])
            night_id.append(np.char.add(name, (d.index[ok] - pd.Timedelta(hours=12)).strftime("%Y%m%d").to_numpy().astype(str)))
            for t in tags:
                rows[t].append(((np.roll(fcs[t], -1) - fcs[t]) * 2)[ok])
        rh = np.concatenate(rh_all)
        idx = np.digitize(rh, bins)
        centres = (bins[:-1] + bins[1:]) / 2
        # 95% CI of the bin means from 1000 bootstrap resamples of whole nights (samples within a night are correlated)
        _, night = np.unique(np.concatenate(night_id), return_inverse=True)
        n_bins, n_nights = len(bins) - 1, night.max() + 1
        w = rng.multinomial(n_nights, np.full(n_nights, 1 / n_nights), size=1000)
        inbin = (idx >= 1) & (idx <= n_bins)
        cnt = np.zeros((n_nights, n_bins))
        np.add.at(cnt, (night[inbin], idx[inbin] - 1), 1)
        for k, v in rows.items():
            v = np.concatenate(v)
            tot = np.zeros((n_nights, n_bins))
            np.add.at(tot, (night[inbin], idx[inbin] - 1), v[inbin])
            enough = cnt.sum(0) >= 30
            m = np.where(enough, tot.sum(0) / np.maximum(cnt.sum(0), 1), np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = (w @ tot) / (w @ cnt)
            lo, hi = (np.where(enough, np.nanpercentile(boot, q, 0), np.nan) for q in (2.5, 97.5))
            col = C["obs"] if k == "obs" else (C["graybox"] if "hyb" in k else C[k])
            ax.fill_between(centres, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(centres, m, marker="o", ms=4, color=col, lw=1.2 if k == "obs" else 1.0,
                    label="Observed" if k == "obs" else LABEL[k], mec=SURF, mew=0.8)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xlabel("Relative humidity (%)")
        ax.set_title(SEASON[season] + ", all test sets, 19–05 h", loc="left", fontsize=8)
    axes[0].set_ylabel("Nocturnal tendency dT/dt (K h$^{-1}$)")
    axes[1].legend(loc="lower right", fontsize=7, title="Bin mean and 95% CI", title_fontsize=7)
    fig.tight_layout()
    save(fig, out, "Figure5_cooling_vs_RH")


# ---------------------------------------------------------------- Figure 6
@torch.no_grad()
def fig_hybrid_factors(out, res, processed):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    names = ["α (absorption)", "γ (emission)", "β (humidity)"]
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    for ax, season in zip(axes, ("dry", "wet")):
        sc = training_scalers(processed, season)
        m = load_model(res, season, "moist_closed_hyb_TRHe", sc)
        tr, va, te = calibration_split(processed, season)
        d = pd.concat([tr, va, te]).dropna(subset=["T", "R", "RH", "e", "de"])
        arr = block_arrays(d)
        F = {k: torch.as_tensor(arr[k], dtype=TORCH_DTYPE)[None] for k in FORCING_COLS}
        T = torch.as_tensor(arr["T"], dtype=TORCH_DTYPE)[None].expand(m.M, -1)
        _, parts = m.rhs(T, F, return_parts=True)
        fac = torch.exp(m.mult_bound * torch.tanh(parts["out"][..., :3])).numpy()  # (M, N, 3)
        hour = (d.index - pd.Timedelta(minutes=15)).hour.to_numpy()
        for k in range(3):
            prof = np.array([[fac[i, hour == h, k].mean() for h in range(24)] for i in range(m.M)])
            ax.fill_between(range(24), prof.min(0), prof.max(0), color=cols[k], alpha=0.22, lw=0)
            ax.plot(range(24), prof.mean(0), color=cols[k], label=names[k], marker="o", ms=3, mec=SURF, mew=0.6)
        ax.axhline(1, color=INK2, lw=0.6)
        ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("Local hour")
        ax.set_yscale("log"); ax.set_yticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(SEASON[season] + ", Jaru 2008", loc="left", fontsize=8)
    axes[0].set_ylabel("Learned modulation factor")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, out, "Figure6_hybrid_factors")


# ---------------------------------------------------------------- Figure 7
def fig_adaptation(out, res):
    tags = ["dry_none", "moist_closed_none", "dry_add_TR", "dry_hyb_TRHe", "moist_closed_hyb_TRHe", "none_add_TRHe"]
    strat_style = {"transfer": ("o", "none", "Transfer (no adaptation)"), "phys": ("o", "fill", "Physical parameters re-estimated"),
                   "full": ("D", "fill", "All parameters fine-tuned")}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, season in zip(axes, ("dry", "wet")):
        files = sorted((res / "adaptation").glob(f"{season}__*.json"))
        recs = [json.loads(f.read_text()) for f in files]
        recs = [r for r in recs if r["set"] != "jaru2009"]
        pers = np.mean([r["persistence_rmse"] for r in recs])
        for i, tag in enumerate(tags):
            for s, (mk, fill, lab) in strat_style.items():
                v = [r["models"][tag][s]["fc_rmse"] for r in recs if tag in r["models"] and s in r["models"][tag]
                     and "fc_rmse" in r["models"][tag][s]]
                if not v:
                    continue
                col = C.get(tag, C["graybox"])
                ax.plot(np.mean(v), i, marker=mk, ms=6, mfc=col if fill == "fill" else SURF, mec=col, mew=1.2, ls="none")
        ax.axvline(pers, color=INK2, lw=0.8)
        ax.text(pers, len(tags) - 0.55, " persistence", fontsize=6.5, color=INK2, va="top")
        ax.set_yticks(range(len(tags)), [LABEL[t] for t in tags])
        ax.set_xlabel("24-h forecast RMSE (K), mean over sets")
        ax.set_title(SEASON[season] + " (Jaru 2010, Sinop, Cuieiras)", loc="left", fontsize=8)
        ax.invert_yaxis()
    h = [matplotlib.lines.Line2D([], [], marker=mk, ms=6, ls="none", mec=INK2, mew=1.2, mfc=INK2 if fill == "fill" else SURF)
         for mk, fill, _ in strat_style.values()]
    fig.legend(h, [lab for *_, lab in strat_style.values()], ncol=3, loc="upper center", bbox_to_anchor=(0.55, 1.06), fontsize=7)
    fig.tight_layout()
    save(fig, out, "Figure7_adaptation")


# ---------------------------------------------------------------- Figure 8
def fig_sensitivity(out, res):
    tags = ["dry_none", "moist_closed_none", "dry_add_TR", "dry_add_TRHe", "dry_mult_TRHe", "dry_hyb_TRHe", "moist_closed_add_TRHe",
            "moist_closed_mult_TRHe", "moist_closed_hyb_TRHe", "none_add_TRHe"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8), sharey=True)
    for r, season in enumerate(("dry", "wet")):
        S = json.loads((res / f"sensitivity_{season}.json").read_text())
        for c, kind in enumerate(("R+10%", "e-10%")):
            ax = axes[r, c]
            for i, tag in enumerate(tags):
                if tag not in S:
                    continue
                col = C.get(tag, C["graybox"])
                for key, mk, dy in (("dTmax_members", "^", -0.17), ("dTmin_members", "v", 0.17)):
                    v = np.array(S[tag][kind][key])
                    ax.errorbar(v.mean(), i + dy, xerr=v.std(), fmt=mk, ms=5, color=col, mec=col, elinewidth=1, capsize=0)
            ax.axvline(0, color=INK2, lw=0.6)
            ax.set_yticks(range(len(tags)), [LABEL[t] for t in tags])
            ax.set_title(f"{SEASON[season]}: {'R × 1.1' if kind == 'R+10%' else 'e × 0.9'}", loc="left", fontsize=8)
            ax.set_xlabel("Change in daily extreme (K), mean ± s.d. over members")
            ax.invert_yaxis() if c == 0 and r == 0 else None
    h = [matplotlib.lines.Line2D([], [], marker=mk, ms=5, ls="none", color=INK2) for mk in ("^", "v")]
    fig.tight_layout()
    fig.legend(h, ["Change in daily maximum", "Change in daily minimum"], ncol=2, loc="upper center",
               bbox_to_anchor=(0.55, 1.04), fontsize=7)
    save(fig, out, "Figure8_sensitivity")


# ---------------------------------------------------------------- Figure 9
def fig_distillation(out, res, processed):
    """Mean diurnal cycle of the additive correction of the dry model (inputs T, R) and of its
    degree-1 and degree-2 polynomial distillations, with the two linear terms of the degree-1 form."""
    import importlib.util

    from sklearn.preprocessing import PolynomialFeatures

    spec = importlib.util.spec_from_file_location("p10", Path(__file__).with_name("10_polynomial_distillation.py"))
    p10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p10)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, season in zip(axes, ("dry", "wet")):
        sc = training_scalers(processed, season)
        m = load_model(res, season, "dry_add_TR", sc)
        tr, _, _ = calibration_split(processed, season)
        d = tr.dropna(subset=p10.REQUIRED)
        x, y = p10.corrections_at(m, *p10.states(d), ["c"])
        hour = (d.index - pd.Timedelta(minutes=15)).hour.to_numpy()
        prof = lambda v: np.array([[v[i, hour == h].mean() for h in range(24)] for i in range(v.shape[0])])  # noqa: E731
        net = prof(y[..., 0])
        fits = {}
        for deg in (1, 2):
            pw = PolynomialFeatures(deg).fit(np.zeros((1, 2))).powers_
            A = p10.design(x, pw)
            coef = np.stack([np.linalg.lstsq(A, y[i, :, 0], rcond=None)[0] for i in range(m.M)])  # (M, P)
            fits[deg] = (A @ coef.T).T, coef
        lin_coef = fits[1][1]  # columns: 1, T, R
        terms = {r"Radiative term $b_R\,(R-\bar{R})$": (lin_coef[:, 2:3] * x[None, :, 1], "#eda100"),
                 r"Restoring term $b_T\,(T-\bar{T})$": (lin_coef[:, 1:2] * x[None, :, 0], "#1baf7a")}
        ax.fill_between(range(24), net.min(0), net.max(0), color=C["graybox"], alpha=0.2, lw=0)
        ax.plot(range(24), net.mean(0), color=C["graybox"], lw=1.6, label="Network (mean and range of members)")
        ax.plot(range(24), prof(fits[1][0]).mean(0), color=INK, lw=1.2, ls="--", label="Degree-1 polynomial")
        ax.plot(range(24), prof(fits[2][0]).mean(0), color=INK, lw=1.0, ls=":", label="Degree-2 polynomial")
        for lab, (v, col) in terms.items():
            ax.plot(range(24), prof(v).mean(0), color=col, lw=1.2, label=lab + " (degree 1)")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("Local hour")
        ax.set_title(SEASON[season] + ", Jaru 2008 training block", loc="left", fontsize=8)
    axes[0].set_ylabel(r"Correction to d$T$/d$t$ (K h$^{-1}$)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, lab, ncol=3, loc="upper center", bbox_to_anchor=(0.53, 1.12), fontsize=7)
    save(fig, out, "Figure9_distillation")


# ---------------------------------------------------------------- Figure 2
def fig_silva(out, res):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0))
    for ax, season in zip(axes, ("dry", "wet")):
        f = res / f"silva_protocol_{season}_series.npz"
        if not f.exists():
            continue
        z = np.load(f)
        sanity.check_temperature(z["T_obs"], f"Silva protocol {season} observations")
        js = json.loads((res / f"silva_protocol_{season}.json").read_text())
        op, cl = js["unit10_open_loop"], js["closed_loop_ensemble"]
        t = pd.to_datetime(z["time"], unit="m")
        sl = slice(0, 14 * 48)
        ax.plot(t[sl], z["T_obs"][sl] - 273.15, color=C["obs"], lw=1.3, label="Observed")
        ax.fill_between(t[sl], np.minimum(z["T_p5"], z["T_p95"])[sl] - 273.15, np.maximum(z["T_p5"], z["T_p95"])[sl] - 273.15,
                        color=C["silva"], alpha=0.18, lw=0,
                        label=f"Open-loop 5–95% band (100 members, original protocol): coverage {op['coverage90']:.2f}, "
                              f"mean width {op['mean_band_width_K']:.0f} K")
        C_ = z["closed"]
        ax.fill_between(t[sl], np.percentile(C_, 5, 0)[sl] - 273.15, np.percentile(C_, 95, 0)[sl] - 273.15,
                        color=C["graybox"], alpha=0.6, lw=0,
                        label=f"Closed-loop 5–95% band (20 members): coverage {cl['coverage90']:.2f}, "
                              f"mean width {cl['mean_band_width_K']:.1f} K")
        ax.plot(t[sl], z["T_cor"][sl] - 273.15, color=C["silva"], lw=0.8, ls="--",
                label=f"Main corrected model (Euler), whole season: {flags(z['T_obs'], z['T_cor'])}")
        ax.set_ylabel("Air temperature (°C)")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6.5, handlelength=1.6, borderaxespad=0.2,
                  title=f"{SEASON[season]}, Rebio Jaru 2008, first 14 days", title_fontsize=8, alignment="left")
        ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
        ax.margins(x=0)
    fig.tight_layout(h_pad=1.0)
    save(fig, out, "Figure2_silva_bands")


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res, proc = Path(a.results), Path(a.processed)
    for f in a.only or ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        {"1": lambda: fig_schematic(out), "2": lambda: fig_silva(out, res), "3": lambda: fig_timeseries(out, res, proc),
         "4": lambda: fig_heatmaps(out, res), "5": lambda: fig_cooling_vs_rh(out, res, proc),
         "6": lambda: fig_hybrid_factors(out, res, proc), "7": lambda: fig_adaptation(out, res),
         "8": lambda: fig_sensitivity(out, res), "9": lambda: fig_distillation(out, res, proc)}[f]()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "tiff", "svg", "eps"])
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    FORMATS[:] = args.formats
    main(args)
