"""Dataset splits and test sets shared by all experiment scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import select_months
from .sanity import check_block

SEASON_MONTHS = {"dry": (7, 9), "wet": (1, 3)}


def read(processed: Path, tag: str) -> pd.DataFrame:
    return pd.read_csv(Path(processed) / f"{tag}.csv", index_col=0, parse_dates=True)


def calibration_split(processed: Path, season: str, year: int = 2008, frac=(0.70, 0.15)):
    """Chronological train / validation / test blocks within one Jaru season."""
    d = select_months(read(processed, f"jaru_{year}"), *SEASON_MONTHS[season])
    n = len(d)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    parts = {"train": d.iloc[:a], "val": d.iloc[a:b], "test": d.iloc[b:]}
    return tuple(check_block(p, f"jaru_{year} {season} {k}") for k, p in parts.items())


def test_sets(processed: Path, season: str) -> dict:
    """Independent evaluation sets: other years at Jaru and the other two towers."""
    m = SEASON_MONTHS[season]
    sets = {}
    _, _, sets["jaru2008_test"] = calibration_split(processed, season)
    for y in (2009, 2010):
        sets[f"jaru{y}"] = select_months(read(processed, f"jaru_{y}"), *m)
    if season == "dry":
        sets["sinop2002_06"] = read(processed, "sinop_2002_06")
        sets["sinop2003_08-09"] = pd.concat([read(processed, "sinop_2003_08"), read(processed, "sinop_2003_09")])
        c = read(processed, "cuieiras_2003")
        sets["cuieiras2003_08-10"] = c[(c.index - pd.Timedelta(minutes=30)).month <= 10]
    else:
        sets["sinop2001_04-05"] = pd.concat([read(processed, "sinop_2001_04"), read(processed, "sinop_2001_05")])
        sets["cuieiras2004_01-03"] = select_months(read(processed, "cuieiras_2004"), 1, 3)
    return {k: check_block(v, k) for k, v in sets.items()}


def climatology_baseline(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Mean diurnal cycle of the training block, repeated over the target period."""
    key = lambda d: d.index.hour * 2 + d.index.minute // 30  # noqa: E731
    clim = train["T"].groupby(key(train)).mean()
    return clim.reindex(key(target)).to_numpy()


def persistence_baseline(target: pd.DataFrame) -> np.ndarray:
    """Yesterday-at-the-same-time forecast."""
    return target["T"].shift(48).to_numpy()


def model_spec(tag: str) -> tuple[str, str, str]:
    """(white box, correction, inputs) from a result tag such as ``moist_closed_hyb_TRHe``."""
    for white in ("moist_closed", "moist_open", "dry", "none"):
        if tag.startswith(white + "_"):
            rest = tag[len(white) + 1 :].split("_")
            return white, rest[0], rest[1] if len(rest) > 1 else "TRHe"
    raise ValueError(tag)


def training_scalers(processed: Path, season: str):
    """Input scalers exactly as fitted by ``scripts/03_run_experiment.py``."""
    from .graybox import Scalers

    tr, _, _ = calibration_split(processed, season)
    return Scalers.fit({"T": tr["T"], "R": tr["R"], "RH": tr["RH"], "e": tr["e"]}, float(np.nanstd(tr["dT"])))


def load_model(results: Path, season: str, tag: str, scalers, M: int = 10):
    """Rebuild a trained ensemble from ``results/<season>/<tag>.pt``."""
    import torch

    from .graybox import GrayBoxODE

    white, corr, inputs = model_spec(tag)
    model = GrayBoxODE(white, corr, M=M, inputs=inputs, scalers=scalers)
    model.load_state_dict(torch.load(Path(results) / season / f"{tag}.pt"))
    return model
