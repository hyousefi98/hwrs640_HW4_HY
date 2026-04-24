# HWRS640 - Assignment 4: Streamflow Prediction with Sequence Models

## Due date: TBD

---

## Background

In this assignment you will build a sequence model to predict daily streamflow from meteorological forcings and basin attributes. Rather than working in a single notebook, you will create a small but well-structured Python project with a command-line interface (CLI). The goal is to practice both modern machine learning for hydrology and basic software engineering: modular design, reproducible training, and clear separation of responsibilities across files.

You will use the [`minicamels`](https://github.com/BennettHydroLab/minicamels) package, which provides a pared-down version of CAMELS-US for teaching and learning. The dataset contains:

- 50 basins
- 30 water years (WY1981-WY2010)
- Daymet-derived daily forcings: `prcp`, `tmax`, `tmin`, `srad`, `vp`
- Observed streamflow: `qobs`
- Static basin attributes

The package exposes a `MiniCamels` class that can load individual basins, multiple basins, static attributes, and water-year slices. You are expected to build your data pipeline on top of this package rather than manually downloading and parsing raw CAMELS files.

You must develop **one** sequence model approach. For example, you could implement:

- An **LSTM-based** model
- A **Transformer-based** model

But, the model architecture is up to you if you would like to explore another approach.

Your model should take a sequence of daily meteorological forcings, combined with static basin attributes, and predict streamflow.

---

## Learning goals

By the end of this assignment, you should be able to:

- Prepare multivariate hydrologic time series for supervised learning.
- Design and train a sequence model for rainfall-runoff prediction.
- Evaluate model skill using appropriate hydrologic metrics and visual diagnostics.
- Organize a machine learning project into reusable Python modules.
- Expose training, evaluation, and plotting functionality through a CLI.

---

## Repository requirements

Create a **new Git repository** for this assignment. Your repository should contain, at minimum, the following files:

- `README.md` - project overview, instructions for setup and reproduction, and CLI documentation
- `main.py` - the main CLI entry point
- `data.py` - dataset loading, preprocessing, and dataloaders
- `model.py` - model definitions
- `train.py` - training and validation routines
- `utils.py` - helper functions
- `visualization.py` - plotting functions

You may add other files and directories if needed, for example:

- `requirements.txt` or `pyproject.toml`
- `tests/`
- `outputs/`
- `configs/`

Your code should be modular. `main.py` should orchestrate work by importing functions and classes from the other modules rather than containing the entire project logic inline.

---

## CLI requirements

Your repository must expose a CLI through `main.py`. At a minimum, your CLI must support the following tasks:

1. Inspect or summarize the dataset
2. Train a model
3. Evaluate a trained model
4. Generate plots

One possible interface is:

```bash
# Provide example visualizations/statistics about the dataset
python main.py summarize-data
# Train a model with specified hyperparameters
python main.py train --model lstm --seq-len 30 --epochs 20
# Evaluate the best model on the test set
python main.py evaluate --checkpoint outputs/best_model.pt
# Generate plots for the best model
python main.py plot --checkpoint outputs/best_model.pt
```

You may use `argparse`, `click`, or another standard Python CLI library. Whatever you choose, your commands and arguments should be clearly documented in your `README.md`.

---

## Problem 1: Data access and exploratory analysis 

Use `minicamels` to inspect the dataset and build your supervised learning problem.

1. Load the basin index, static attributes, and at least several basin time series using `MiniCamels`.
2. Report the dataset structure:
   - number of basins
   - time span
   - dynamic input variables
   - target variable
   - number of static attributes used
3. Choose a train/validation/test split strategy and justify it in 3-4 sentences.
   - You may split by basin, by time, or by a combination of both.
   - Your split must avoid leakage.
4. Define the supervised learning samples.
   - Choose a sequence length (for example, 30, 60, or 90 days).
   - Explain what the model input and target are.
   - State whether you predict one day ahead or another forecast horizon.
5. Produce at least three exploratory plots, such as:
   - streamflow time series for multiple basins
   - precipitation and streamflow for one basin over the same period
   - histograms of `qobs`
   - a map or scatterplot using static attributes
6. Describe any preprocessing steps:
   - missing-value handling
   - normalization or standardization
   - treatment of static attributes
   - sequence window construction

Your `data.py` module should contain the core code for dataset access, sample generation, and PyTorch `Dataset`/`DataLoader` creation.

---

## Problem 2: Model design

Implement at least one sequence model in `model.py`.

Your model must:

- Accept a batch of input sequences of shape `(batch, seq_len, num_features)` or an equivalent documented format.
- Produce a streamflow prediction for each sample.

If you include static basin attributes, clearly explain how they enter the model. For example, you might concatenate static attributes to each timestep, encode them separately and fuse them with the sequence representation, or use another reasonable design.

Document and discuss:

1. The full model architecture
2. Why your chosen architecture is appropriate for streamflow prediction
3. One expected strength and one expected weakness of your architecture

---

## Problem 3: Training pipeline and CLI implementation (25 points)

Implement the training workflow in `train.py` and expose it through `main.py`.

Your training pipeline might include:

- training and validation loops
- loss computation
- optimizer setup
- model checkpointing
- metric tracking
- command-line arguments for key hyperparameters

At minimum, your CLI training command should allow the user to specify:

- model hyper parameters
- sequence length
- learning rate
- batch size
- number of epochs
- output/checkpoint path

Use a regression loss appropriate for streamflow prediction, such as MSE or MAE. You may additionally report Nash-Sutcliffe Efficiency (NSE), Kling-Gupta Efficiency (KGE), or correlation-based metrics during validation.

Produce the following plots:

- training loss vs. epoch
- validation loss vs. epoch
- at least one validation metric vs. epoch

Your code should be organized so that plotting functions live in `visualization.py` and reusable helper functions live in `utils.py`.

---

## Problem 4: Evaluation and interpretation (25 points)

Evaluate your trained model on the held-out test set.

1. Report at least three quantitative metrics. At least one must be a hydrology-relevant metric such as NSE/KGE or peak flow/low flow.
2. Plot predicted vs. observed streamflow time series for at least one test basin.
3. Create at least one parity plot or scatterplot of predicted vs. observed values.
5. Identify one basin where the model performs well and one where it performs poorly.
6. Discuss possible reasons for the performance differences, using basin behavior, climate, seasonality, or catchment attributes where relevant.

Your discussion should connect the results back to both the data and the model design.

---

## Deliverables

Submit the following:

1. A link to your Git repository
2. A short report in PDF format with figures and written answers
3. Instructions for reproducing your best model run

Your report should include:

- brief methods overview
- requested plots and tables
- answers to all discussion questions
- a short conclusion summarizing what worked and what did not

## Suggested workflow

1. Create a new repository and set up your environment via `uv`.
2. Explore `minicamels` and decide on your split strategy.
3. Implement `data.py` and verify your tensors and dataloaders.
4. Implement `model.py` and test a forward pass.
5. Implement `train.py` and connect it through `main.py`.
6. Add plotting functions in `visualization.py`.
7. Run experiments, evaluate results, and write your report.
