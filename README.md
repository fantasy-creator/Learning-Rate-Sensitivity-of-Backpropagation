# Learning-Rate Sensitivity of Backpropagation

Code for the ECS 171 (UC Davis) final project *Learning-Rate Sensitivity of Backpropagation: A One-Step
Bound and an Empirical Study of Proximal Updates*, by Zekai Yu, Tingshuo Fan and Jiacheng Chen (equal
contribution).

The project starts from Rumelhart, Hinton & Williams (1986) and asks whether Proximal Backpropagation
(Frerix et al., 2018) widens the range of learning rates that train well, or only moves it. It checks a
one-step bound on the parameter response to a learning-rate change, then sweeps 13 learning rates on small
MLPs over three UCI datasets.

## Update rules

All rules share BP's forward and backward pass and are written out by hand in `src/core.py`.

| Method | Hidden layers | Output layer |
| --- | --- | --- |
| `bp_sgd` | `-lr * g` | `-lr * g` |
| `prox_sgd` | `-lr * g M^-1` | `-lr * g` |
| `prox_all` | `-lr * g M^-1` | `-lr * g M^-1` |
| `nm_static` | `-lr * c_l * g`, with `c_l` frozen at initialization | `-lr * g` |
| `nm_dynamic` | `-lr * s_l(t) * g`, with `s_l(t)` recomputed every step | `-lr * g` |
| `bp_adam` | Adam on `g` | Adam on `g` |

Here `g` is the BP gradient, `M = I + A_hat A_hat^T` for the layer's input batch with a bias row, and `c_l`
and `s_l(t)` are ratios of ProxProp's step norm to BP's.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The datasets are not stored in this repository. Download them from the UCI repository and unpack them under
`data/raw` so that these paths exist:

| Dataset | Source | Paths under `data/raw` |
| --- | --- | --- |
| Dry Bean | [UCI 602](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | `drybean/DryBeanDataset/Dry_Bean_Dataset.arff` |
| Optdigits | [UCI 80](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) | `optdigits/optdigits.tra`, `optdigits/optdigits.tes` |
| HAR | [UCI 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | `har/UCI HAR Dataset/` (a zip inside the download) |

Then build the fixed splits:

```bash
python src/data_prep.py --raw data/raw --out data/processed
```

## Running

```bash
python src/test_core.py     # correctness checks for the update rules
SMOKE=1 bash run.sh         # unit tests plus 2-epoch runs of every method
bash run.sh                 # full pipeline
bash src/postprocess.sh     # analysis, tables and figures
```

`run.sh` reads `PYTHON` (default `python3`) and `WORKERS` (default 30, each a single-threaded run). It is CPU
only by design and skips finished work, so it can be restarted. Each worker needs a few hundred MB of memory.
With 30 workers the 3,510-run main sweep took about 24 minutes.

| Step | Runs | Output |
| --- | --- | --- |
| One-step bound checks | 1,260 layer records | `results/onestep/` |
| Main sweep | 3,510 | `results/main/` |
| Input-scale, momentum and `tau` ablations (3 seeds) | 702, 234, 234 | `results/{scale,momentum,tau}/` |
| Test stage at validation-selected learning rates | 270 | `results/test_main/` |
| Logistic-regression baseline, per-step cost | | `results/logreg.json`, `results/bench_step.json` |

`src/postprocess.sh` writes tables to `paper/tables/` and figures to `paper/figures/`.

## Code layout

| File | Purpose |
| --- | --- |
| `src/core.py` | Data loading, MLPs, update rules, training loop |
| `src/data_prep.py` | Fixed train/validation/test splits |
| `src/test_core.py` | Correctness checks for the update rules |
| `src/sweep.py` | Parallel experiment grids |
| `src/onestep.py`, `src/dump_modes.py` | One-step bound checks and per-mode responses |
| `src/final_test.py` | Test-stage re-runs of the validation-selected configurations |
| `src/logreg.py`, `src/bench_step.py` | Logistic-regression baseline and per-step cost |
| `src/analyze.py`, `src/universal_lr.py` | Aggregation and paired tests |
| `src/make_tables.py`, `src/make_results_tables.py` | LaTeX tables |
| `src/fig_*.py` | Figures |

## Reproducibility

Everything runs in float64 on CPU, because ProxProp's smallest updates would be rounded away in float32. A
seed fixes the initialization and the mini-batch order, which all methods share. Every result stores the
SHA-1 prefix of `src/core.py` as `code_version`, and `src/sweep.py` only skips a run whose stored version
matches the current file, so editing `src/core.py` invalidates earlier results. The reported results have
`code_version` `5d565f754e9a` and used Python 3.10, PyTorch 2.3.0 and NumPy 1.24.4; other versions may give
slightly different numbers.
