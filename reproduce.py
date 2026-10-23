"""Run the whole study, or part of it, in the order used for the manuscript.

    python reproduce.py outputs            # tables and figures from the stored results (minutes)
    python reproduce.py evaluate outputs   # re-evaluate the stored ensembles, then tables and figures
    python reproduce.py all --jobs 4       # data preparation, training, evaluation, tables and figures

Stages (each one needs the results of the previous ones):
  data      raw tower spreadsheets in --raw -> data/processed (script 01)
  train     every ensemble of both seasons (script 03; physics-only models first)
  evaluate  scripts 04, 02, 05, 06, 09, 10 and 11
  outputs   script 08 (tables) and script 07 (figures)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEASONS = ("dry", "wet")
PHYSICS = ("dry", "moist_open", "moist_closed")
# (white box, correction, network inputs); "none" as white box is the black-box ODE-net
CORRECTED = [("dry", "add", "TR"), ("dry", "add", "TRHe"), ("dry", "mult", "TRHe"), ("dry", "hyb", "TRHe"),
             ("moist_closed", "add", "TR"), ("moist_closed", "add", "TRHe"), ("moist_closed", "mult", "TRHe"),
             ("moist_closed", "hyb", "TRHe"), ("none", "add", "TRHe")]
ADAPTATION_SETS = {"dry": ["jaru2009", "jaru2010", "sinop2002_06", "sinop2003_08-09", "cuieiras2003_08-10"],
                   "wet": ["jaru2009", "jaru2010", "sinop2001_04-05", "cuieiras2004_01-03"]}


def run(args: list[str]) -> None:
    cmd = [sys.executable, *args]
    print("+", " ".join(cmd), flush=True)
    env = os.environ | {"PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def run_all(jobs: list[list[str]], n: int) -> None:
    with ThreadPoolExecutor(max_workers=n) as pool:
        for f in [pool.submit(run, j) for j in jobs]:
            f.result()


def stage_data(a):
    run(["scripts/01_prepare_data.py", "--root", str(a.raw), "--out", "data/processed"])


def stage_train(a):
    run_all([["scripts/03_run_experiment.py", "--season", s, "--white", w, "--corr", "none"]
             for s in SEASONS for w in PHYSICS], a.jobs)
    run_all([["scripts/03_run_experiment.py", "--season", s, "--white", w, "--corr", c, "--inputs", i]
             for s in SEASONS for w, c, i in CORRECTED], a.jobs)


def stage_evaluate(a):
    run_all([["scripts/04_evaluate.py", "--season", s] for s in SEASONS]
            + [["scripts/02_reproduce_silva.py", "--season", s] for s in SEASONS]
            + [["scripts/05_site_adaptation.py", "--season", s, "--set", t] for s in SEASONS for t in ADAPTATION_SETS[s]]
            + [["scripts/06_sensitivity.py", "--season", s] for s in SEASONS]
            + [["scripts/09_humidity_diagnostics.py", "--season", s] for s in SEASONS], a.jobs)
    run_all([["scripts/10_polynomial_distillation.py", "--season", s] for s in SEASONS], a.jobs)
    run(["scripts/11_explicit_equations.py"])


def stage_outputs(a):
    run(["scripts/08_tables.py", "--out", "results/tables/tables.json"])
    run(["scripts/07_figures.py", "--out", "figures"])


STAGES = {"data": stage_data, "train": stage_train, "evaluate": stage_evaluate, "outputs": stage_outputs}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=[*STAGES, "all"])
    ap.add_argument("--jobs", type=int, default=1, help="independent runs executed in parallel (one CPU thread each)")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw", help="folder with the raw tower spreadsheets")
    a = ap.parse_args()
    for name in STAGES if "all" in a.stages else [s for s in STAGES if s in a.stages]:
        print(f"== stage {name}", flush=True)
        STAGES[name](a)
