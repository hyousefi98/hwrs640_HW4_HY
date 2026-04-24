# camelflow — Streamflow Prediction with LSTM

A small, reproducible Python package that trains an LSTM to predict daily streamflow for 50 CAMELS-US basins, using the [`minicamels`](https://github.com/BennettHydroLab/minicamels) dataset. Built for HWRS640 Assignment 4.

- **Input per day:** 5 Daymet forcings (`prcp`, `tmax`, `tmin`, `srad`, `vp`) over a 365-day window, plus 14 static basin attributes.
- **Output:** next-day observed streamflow `qobs` (mm/day).
- **Typical skill:** median test **NSE ≈ 0.79**, KGE ≈ 0.71 across the 50 basins.

Author: Hossein Yousefi Sohi · `hyousefi@arizona.edu`

---

## Quick start

```bash
# 1. install (creates .venv automatically)
uv sync

# 2. look at the data
uv run camelflow summarize-data

# 3. train (about 25 min on a single GPU, stops early on val plateau)
uv run camelflow train

# 4. evaluate + make all plots
uv run camelflow plot --checkpoint outputs/best.pt
```

That's it. Results land in [outputs/](outputs/).

---

## Requirements

- **Python ≥ 3.12** (managed automatically by `uv`)
- **GPU optional** (recommended) — any CUDA-capable NVIDIA card with driver ≥ 530. Falls back to CPU if none is found.
- **[uv](https://docs.astral.sh/uv/)** — one-line install: `pipx install uv` or `pip install --user uv`.

Everything else (PyTorch, xarray, geopandas, matplotlib, `minicamels`, …) is pulled in automatically by `uv sync`.

---

## The 4 CLI commands

All commands share the same `camelflow` entry point. Run any with `--help` for details.

### 1. `summarize-data` — explore the dataset

```bash
uv run camelflow summarize-data
```

Writes to [outputs/figs/summary/](outputs/figs/summary/):

| file | what it shows |
|---|---|
| `qobs_timeseries.png` | daily streamflow for a handful of basins (30-year span) |
| `qobs_histogram.png` | log-transformed streamflow distribution, all basins |
| `precip_vs_qobs.png` | precipitation + streamflow panel for one basin, 2 water years |
| `map_q_mean.png` | CONUS map of the 50 basins colored by mean flow |
| `map_aridity.png` | same map, colored by aridity |
| `static_scatter.png` | aridity vs runoff ratio scatter |
| `attributes_summary.csv` | table of static-attribute statistics |

### 2. `train` — train the LSTM

```bash
# defaults (match configs/default.yaml)
uv run camelflow train

# full control
uv run camelflow train \
  --seq-len 365 --epochs 50 --batch-size 256 \
  --hidden 128 --dropout 0.5 \
  --lr 1e-3 --weight-decay 1e-5 \
  --loss nse --patience 8 --seed 42 \
  --out outputs
```

What you get in `--out` (default `outputs/`):

| file | meaning |
|---|---|
| `best.pt` | checkpoint at the highest val NSE (use this for eval) |
| `last.pt` | final-epoch checkpoint |
| `history.json` | per-epoch train/val loss, val NSE, learning rate |

Training prints one line per epoch and early-stops automatically if val NSE stops improving for `--patience` epochs.

### 3. `evaluate` — score a trained model

```bash
uv run camelflow evaluate --checkpoint outputs/best.pt
```

Writes to `outputs/`:

| file | meaning |
|---|---|
| `test_metrics.csv` | one row per basin: NSE, KGE, RMSE, Pearson r, peak-flow bias |
| `test_summary.json` | median & mean of the metrics across basins |
| `test_predictions.npz` | raw per-sample predictions + observations (for plotting) |

Prints a summary block with the top-3 and bottom-3 basins.

### 4. `plot` — produce every results figure

```bash
uv run camelflow plot --checkpoint outputs/best.pt
```

Runs `evaluate` if needed, then writes:

**[outputs/figs/train/](outputs/figs/train/)**
- `loss_curves.png` — train and val loss vs epoch
- `val_nse.png` — validation NSE vs epoch

**[outputs/figs/pred/](outputs/figs/pred/)**
- `hydrograph_best_<id>.png` — predicted vs observed for best basin
- `hydrograph_median_<id>.png` — predicted vs observed for median-skill basin
- `hydrograph_worst_<id>.png` — predicted vs observed for worst basin
- `parity.png` — predicted vs observed scatter with 1:1 line
- `nse_hist.png` — histogram of per-basin test NSE
- `nse_map.png` — CONUS map colored by test NSE
- `fdc.png` — flow-duration curves (best/median/worst), log-scale
- `metrics_boxplot.png` — NSE / KGE / r distribution across basins
- `skill_vs_attrs.png` — 6-panel: test NSE against aridity, area, precip, baseflow index, snow fraction, elevation

---

## Hyperparameter reference

Flags for `camelflow train` (defaults are in `configs/default.yaml`):

| flag | default | what it does |
|---|---|---|
| `--model` | `lstm` | model architecture (only LSTM registered) |
| `--seq-len` | 365 | number of past days used as input |
| `--epochs` | 30 | max number of epochs |
| `--batch-size` | 256 | mini-batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--hidden` | 128 | LSTM hidden units |
| `--num-layers` | 1 | stacked LSTM layers |
| `--dropout` | 0.5 | dropout before the output head |
| `--weight-decay` | 1e-5 | L2 regularization |
| `--loss` | `nse` | `nse` = basin-averaged NSE\* (Kratzert 2019), or `mse` |
| `--num-workers` | 0 | DataLoader worker threads |
| `--seed` | 42 | RNG seed |
| `--patience` | 8 | early stop after N epochs without val-NSE gain |
| `--min-delta` | 1e-4 | minimum val-NSE improvement to reset patience |
| `--out` | `outputs` | directory for checkpoints, history, metrics |

---

## How it works

**Data split (per basin, temporal — the standard CAMELS benchmark):**

| split | water years | dates |
|---|---|---|
| train | WY 1981 – 2000 | 1980-10-01 → 2000-09-30 |
| val   | WY 2001 – 2005 | 2000-10-01 → 2005-09-30 |
| test  | WY 2006 – 2010 | 2005-10-01 → 2010-09-30 |

Every basin appears in every split. The 365-day input window for the first val/test samples legitimately uses forcings from the previous split (forcings only, never labels), which mirrors real deployment.

**Preprocessing:**
- Per-variable z-score using **train-only** mean/std (no leakage).
- Target: `log1p(qobs)` then z-score. Predictions are inverse-transformed before metrics.
- Static features: 14 columns, median-imputed and z-scored.
- Samples with NaN in the input window or label are dropped at indexing time.

**Model** ([src/camelflow/models.py](src/camelflow/models.py)):

```
input (B, 365, 5 dynamic + 14 static)
   ↓
LSTM (hidden=128, 1 layer, dropout=0.5)
   ↓
last hidden state → dropout → linear(128 → 1)
   ↓
predicted standardized log-flow  →  inverse transform  →  qobs (mm/day)
```

~76k parameters.

**Loss** (default `--loss nse`): the basin-averaged NSE* loss from Kratzert 2019,

    L = mean over samples of ( (y_hat − y)² / (σ_b + ε)² )

where σ_b is the train-period std of the target for basin `b` and ε = 0.1. This prevents low-variance (arid, flashy) basins from being drowned out by wet-basin errors, which was the main failure mode of a plain MSE loss.

---

## Project layout

```
.
├── main.py                 # thin wrapper → camelflow.cli:main
├── pyproject.toml          # build config + deps
├── configs/default.yaml    # default hyperparameters
├── src/camelflow/
│   ├── cli.py              # argparse subcommands
│   ├── data.py             # MiniCamels loader, split, normalize, Dataset
│   ├── models.py           # LSTMRegressor
│   ├── train.py            # train loop, NSE loss, early stopping
│   ├── evaluate.py         # per-basin test metrics
│   ├── utils.py            # NSE / KGE / RMSE / r / peak_bias / checkpoint I/O
│   └── visualization.py    # every plotting function
├── tests/
│   ├── test_data.py        # dataset shape, no-leak
│   └── test_model.py       # forward / backward
├── notebooks/              # exploratory notebook
└── outputs/                # gitignored — checkpoints, metrics, figures
```

---

## Development

Run tests:

```bash
uv run pytest -q
```

Build a wheel (for distribution):

```bash
uv build
# → dist/camelflow-0.1.0-py3-none-any.whl
```

The wheel is a normal installable package:

```bash
pip install dist/camelflow-0.1.0-py3-none-any.whl
camelflow --help
```

---

## Reproducing the reported results

```bash
uv sync
uv run camelflow train --epochs 50 --seq-len 365 --batch-size 256 \
                 --hidden 128 --dropout 0.5 --weight-decay 1e-5 \
                 --loss nse --lr 1e-3 --seed 42
uv run camelflow plot --checkpoint outputs/best.pt
```

On an RTX 6000 Ada, training early-stops around epoch ~25 and the full pipeline finishes in ~30 minutes.

---

## Rebuilding the PDF report

The report source is a notebook at [report/report.ipynb](report/report.ipynb), generated from [report/_build_notebook.py](report/_build_notebook.py). To rebuild after any text or figure change:

```bash
# one-time setup
uv sync --extra report
uv run playwright install chromium

# rebuild the PDF (no LaTeX needed — uses headless Chromium)
bash report/make.sh
```

The script rebuilds `report.ipynb` from the source script, then exports it to `report/report.pdf` via `nbconvert --to webpdf`. Do **not** use VSCode's built-in "Export to PDF" button — it requires TeX/xelatex, which is not installed.

---

## Credits

- [`minicamels`](https://github.com/BennettHydroLab/minicamels) — pared-down CAMELS-US dataset (Bennett Hydro Lab).
- Underlying data: [CAMELS-US](https://ral.ucar.edu/solutions/products/camels) (Newman et al. 2015; Addor et al. 2017).
- LSTM-for-rainfall-runoff approach follows Kratzert et al., *HESS* 2018 / 2019.
