"""Build the clean 30-min datasets used by all experiments.

Output: one CSV per site and year in ``data/processed`` with columns
T [K], R [W m-2], RH [%], e [kPa], de [kPa h-1], dRH [% h-1] and a ``filled`` flag.
Gaps of at most 2 h are linearly interpolated; longer gaps stay NaN.
Every file must pass the hard physical bounds of ``amazon_graybox.sanity``; months with
nocturnal net radiation above the bound are listed in the QC log and must not be used
(the calibration and test blocks are checked again when they are read).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from amazon_graybox import sanity
from amazon_graybox.data import fill_short_gaps, load_cuieiras, load_jaru, load_sinop_xls
from amazon_graybox.thermo import vapour_pressure

warnings.filterwarnings("ignore", category=UserWarning)

SINOP_FILES = {  # file -> year; Nov/Dec 2003 excluded (net radiometer fault, see QC log)
    "Apr01-Met.xls": 2001,
    "May01-Met.xls": 2001,
    "Cópia de Met-CO2 Jun-02.xls": 2002,
    "Met Aug-03.xls": 2003,
    "Met Sep-03.xls": 2003,
}
SINOP_EXCLUDED = {"Met Nov-03.xls": 2003, "Met Dec-03.xls": 2003}


def finish(df: pd.DataFrame) -> pd.DataFrame:
    raw_nan = df[["T", "R", "RH"]].isna().any(axis=1)
    df = fill_short_gaps(df, max_steps=4)
    df["filled"] = raw_nan & df[["T", "R", "RH"]].notna().all(axis=1)
    df["e"] = vapour_pressure(df["T"], df["RH"])
    # centred differences on the regular 30-min grid, per hour
    df["de"] = (df["e"].shift(-1) - df["e"].shift(1)) / 1.0
    df["dRH"] = (df["RH"].shift(-1) - df["RH"].shift(1)) / 1.0
    df["dT"] = (df["T"].shift(-1) - df["T"].shift(1)) / 1.0
    return df


def qc_entry(d: pd.DataFrame, tag: str) -> dict:
    sanity.check_block(d, tag, night_radiation=False)
    return {"n": len(d), "complete": float(d[["T", "R", "RH"]].notna().all(axis=1).mean()), "filled": int(d.filled.sum()),
            "months_nocturnal_R_above_bound": sanity.nocturnal_radiation_flags(d)}


def main(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    log = {}
    jaru = root / "Dados Rebio Jaru 2006 a 2008.xlsx"
    for y in (2008, 2009, 2010):
        d = finish(load_jaru(jaru, y))
        d.to_csv(out / f"jaru_{y}.csv", index_label="time")
        log[f"jaru_{y}"] = qc_entry(d, f"jaru_{y}")
    for f, y in SINOP_FILES.items():
        d = finish(load_sinop_xls(root / f, y))
        tag = f"sinop_{d.index[0]:%Y_%m}"
        d.to_csv(out / f"{tag}.csv", index_label="time")
        log[tag] = qc_entry(d, tag)
    for f, y in SINOP_EXCLUDED.items():
        d = load_sinop_xls(root / f, y)
        night = d.R[(d.index.hour >= 0) & (d.index.hour < 4)].mean()
        log[f"excluded_{f}"] = f"mean net radiation 00-04 h = {night:.0f} W m-2 (sensor fault)"
    for y in (2003, 2004):
        d = finish(load_cuieiras(root / "Dados para Análises", y))
        d.to_csv(out / f"cuieiras_{y}.csv", index_label="time")
        log[f"cuieiras_{y}"] = qc_entry(d, f"cuieiras_{y}")
    (out / "qc_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    a = ap.parse_args()
    main(a.root, a.out)
