# amazon_graybox

Code, processed data and trained models for the article

> [Authors] (2026). *Physics-guided neural corrections of dry and humidity-aware energy-balance models for
> near-surface air temperature in Amazonian forests.* Submitted to *Agricultural and Forest Meteorology*.

The package couples the dry energy-balance model of de Paulo et al. (2023) and the humidity-aware model of
Maionchi et al. (2025), in an open and a closed (state-consistent) form, with additive, multiplicative and hybrid
neural corrections (gray-box ODE-nets; Malani et al., 2023), trained on simulated trajectories and evaluated at
three Amazonian flux towers (Rebio Jaru, Sinop, Cuieiras). It reorganises and extends the notebooks
`Code_Dry.ipynb` and `Code_Wet.ipynb` of da Silva et al. (2026, *Chaos* 36, 033134).

## Quick start

```bash
git clone https://github.com/[USER]/amazon-graybox.git
cd amazon-graybox
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-exact.txt                    # versions used for the article
pip install -e .
python -m pytest                                         # unit tests and physical checks of the data
python reproduce.py outputs                              # all tables and Figures 1-9 from the stored results
```

`reproduce.py outputs` rewrites `figures/` and `results/tables/` from the trained ensembles stored in `results/`
in less than a minute; with the versions in `requirements-exact.txt` the figures and tables are identical to the
stored ones.
With conda: `conda env create -f environment.yml`. For journal submission, vector and high-resolution
versions of the figures are written with `python scripts/07_figures.py --out figures --formats png pdf tiff`.

## Reproducing the study from scratch

| Command | What it does | Approximate time (4-core CPU) |
|---|---|---|
| `python reproduce.py data` | quality control of the raw tower files (`data/raw`, not included) into `data/processed` | minutes |
| `python reproduce.py train --jobs 4` | trains the 24 ensembles (10 members each; physics-only models first) | 2–7 min per ensemble, about 30–60 min in all |
| `python reproduce.py evaluate --jobs 4` | scripts 04, 02, 05, 06, 09, 10 and 11 | about 1 h |
| `python reproduce.py outputs` | script 08 (tables) and script 07 (figures) | under 1 min |
| `python reproduce.py all --jobs 4` | all of the above, in order | |

Each stage needs the results of the previous ones, and the processed data and trained ensembles are included,
so any stage can be repeated on its own. Every script can also be called directly (run from the repository root
after `pip install -e .`, or with `PYTHONPATH=.`); for example

```bash
python scripts/03_run_experiment.py --season dry --white moist_closed --corr hyb --inputs TRHe
#   white in {dry, moist_open, moist_closed, none}, corr in {none, add, mult, hyb}, inputs in {TR, TRHe}
python scripts/05_site_adaptation.py --season dry --set sinop2002_06
```

Retraining uses fixed seeds, but small numerical differences from the stored results are expected across
hardware, operating systems and library versions. The results in the article were produced on CPU with
Python 3.13, PyTorch 2.14, NumPy 2.5, pandas 3.0, SciPy 1.18, scikit-learn 1.9 and matplotlib 3.11
(`requirements-exact.txt`).

## Where each result of the article comes from

| Article | Produced by | Stored in |
|---|---|---|
| Table 1 (data blocks) | `08_tables.py` | `results/tables/tables.json` |
| Table 2 (physical parameters) | `03_run_experiment.py`, `08_tables.py` | `results/<season>/phys_*.json` |
| Table 3 (skill of all models), Fig. 4 | `04_evaluate.py` | `results/tables/<season>_long.csv` |
| Tables 4–5, Fig. 9 (polynomial distillation) | `10_polynomial_distillation.py` | `results/polynomial/`, `results/tables/polynomial_summary.csv` |
| Fig. 2, Table S1 (protocol of da Silva et al.) | `02_reproduce_silva.py` | `results/silva_protocol_<season>.json` |
| Fig. 5, Fig. 6, Section 3.6 | `07_figures.py`, `09_humidity_diagnostics.py` | `results/humidity_diagnostics_<season>.json` |
| Fig. 7 (adaptation) | `05_site_adaptation.py` | `results/adaptation/` |
| Fig. 8 (forcing sensitivity) | `06_sensitivity.py` | `results/sensitivity_<season>.json` |
| Eq. (7), Eqs. (S1)–(S6), Table S2 | `11_explicit_equations.py` | `results/polynomial/explicit_equations.json` |

## Repository layout

```
amazon_graybox/          package
  data.py                loaders and quality control (Rebio Jaru, Sinop, Cuieiras)
  thermo.py              saturation vapour pressure (Magnus form, Alduchov & Eskridge 1996)
  physics.py             white boxes: dry, moist open (Maionchi), moist closed (this study)
  graybox.py             RK4 ODE-net with additive / multiplicative / hybrid corrections
  training.py            48-h trajectory training, free runs and 24-h forecasts, ensembles
  metrics.py             RMSE, centred RMSE, KGE, Willmott d, coverage, CRPS, diurnal diagnostics
  experiments.py         calibration split, test sets, result-tag parser, model loading
  sanity.py              physical bounds and invariants checked by every script (see below)
scripts/
  01_prepare_data.py     raw spreadsheets (data/raw) -> data/processed/*.csv (+ qc_log.json)
  02_reproduce_silva.py  re-implementation of the da Silva et al. (2026) protocol
  03_run_experiment.py   train one ensemble (white box x correction x inputs) and evaluate it
  04_evaluate.py         comparison tables per season (results/tables/<season>_long.csv)
  05_site_adaptation.py  few-shot adaptation to another year or site (7 days)
  06_sensitivity.py      response to R x 1.1 and e x 0.9
  07_figures.py          manuscript figures (PNG 300 dpi; --formats pdf tiff for submission)
  08_tables.py           manuscript tables (JSON for the .docx builder, and polynomial_summary.csv)
  09_humidity_diagnostics.py  restoring-rate split and free runs with 24-h mean vapour pressure
  10_polynomial_distillation.py  polynomial fits (degree 1-4) of every trained correction, free runs of
                         the distilled ensembles, physical coefficients, and the da Silva et al. polynomial step
  11_explicit_equations.py  each corrected model as one explicit equation (degree 1, median
                         coefficients), integrated and scored (Eq. (7), Eqs. (S1)-(S6), Table S2)
  summarize.py           one-line summary of each trained ensemble (development helper)
tests/
  test_core.py           unit tests (physics, unit conversions, RK4 rollout, tag parser, physical bounds)
  test_processed_data.py the released series pass the physical bounds
reproduce.py             runs the stages above in the order used for the manuscript
```

## Physical sanity checks

`amazon_graybox/sanity.py` holds the bounds that every script enforces. A violation raises
`PhysicalBoundError` (an `AssertionError` that `python -O` does not remove) and stops the script.

* Data, whenever a calibration or test block is read (`experiments.calibration_split`,
  `experiments.test_sets`) and after preparation (script 01): temperature above absolute zero and
  within the QC range (283.15-313.15 K), 0 <= RH <= 100 %, vapour pressure not above saturation,
  net radiation within -150 to 1100 W m-2 and, at night (20:00-04:00 local time), not above
  50 W m-2. Months that break the nocturnal bound are listed in `data/processed/qc_log.json`
  (four months of Jaru 2010, none of them in the seasons analysed).
* Model: the tendency is evaluated with T limited to 250-340 K, the effective heat capacity of the
  closed moist model must stay positive (checked at every call of `GrayBoxODE.rhs`), and the
  physical coefficients must have their admissible signs (|delta| < delta_max).
* Outputs: a free run or forecast that leaves 250-340 K (so also any run that reaches absolute
  zero) must be reported as diverged (RMSE above 10 K) and is never scored as a valid run
  (`sanity.check_run`, used wherever a run is scored); skill scores must be consistent
  (0 <= MAE <= RMSE, |bias| <= MAE, R2 <= 1, KGE <= 1, -1 <= r <= 1, 0 <= d <= 1).
* Script 09 caps the 24-h mean vapour pressure at saturation at the observed temperature;
  without the cap the smoothed forcing was supersaturated in up to 31 % of the samples.

## Main changes with respect to the original notebooks

* One package instead of two notebooks of 5,562 and 3,962 lines with repeated imports,
  functions redefined up to three times and hard-coded local Windows paths.
* Corrections are trained on simulated 48-h trajectories through the RK4 integrator
  (Malani et al., 2023) instead of on finite-difference tendencies with a random split of
  half-hourly samples, which mixes neighbouring, strongly autocorrelated samples between the
  training and test sets.
* Bootstrap ensembles are integrated closed loop: each member is run on its own simulated
  temperature. The original ensemble evaluates the corrections on the observed temperature
  and integrates percentiles of the corrections, which is not a distribution of trajectories.
* The same integrator (RK4) is used for calibration, training and evaluation; the notebooks
  calibrate with RK4 and simulate the corrected model with explicit Euler.
* Added: the humidity-aware white boxes (open and closed), multiplicative and hybrid
  corrections, a black-box ODE-net, out-of-sample evaluation over years and sites,
  climatology and persistence baselines, probabilistic scores, few-shot adaptation and
  forcing-sensitivity experiments, and unit tests.
* The polynomial step is fitted in standardised inputs, per ensemble member, to the network
  output (in correction units), and the distilled ensembles are integrated and scored like the
  networks. The notebooks fit a degree-4 polynomial in unscaled T and R to the finite-difference
  residuals; in those units the design matrix has a condition number of about 1e18 and the terms
  largely cancel, so their ranking depends on the temperature unit (see
  `results/polynomial/<season>_silva.json`).

## Data

`data/processed` holds the quality-controlled 30-min series produced by `01_prepare_data.py` (columns T [K],
R [W m-2], RH [%], e [kPa], de [kPa h-1], dRH [% h-1] and a gap-filling flag), with `qc_log.json`. `results/`
holds the trained ensembles (`.pt`), their predictions and every table used in the article.
`results/polynomial/<season>_<model>.json` hold the distilled polynomials (coefficient quantiles over the members,
R², scores with and without input clipping and, for the degree-1 additive fits, the physical coefficients α_eff,
λ_N, λ_W and τ); `results/polynomial_4inputs/` holds an earlier run with (T, R, RH, e) as polynomial inputs, which
was nearly collinear and is kept for reference only.

The raw tower spreadsheets are not included: they belong to the LBA programme and its data providers
(see `data/raw/README.md` for the files that `01_prepare_data.py` expects). [Before publishing, the authors must
confirm that the derived series in `data/processed` may be redistributed; otherwise remove that folder and state
how the data can be requested.] The notebooks of da Silva et al. read `Jaru_Dry.xlsx` and `Jaru_Wet.xlsx`, which
were not available; the 2008 seasons were cut from `Dados Rebio Jaru 2006 a 2008.xlsx` instead (fitted parameters
within 1% of the published ones).

## How to cite

See `CITATION.cff` (GitHub shows a "Cite this repository" button). A versioned DOI for the release that
accompanies the article can be obtained by archiving the repository on Zenodo.

## License

Code: MIT (see `LICENSE`). The data in `data/processed` remain subject to the data-use conditions of the LBA
programme.
