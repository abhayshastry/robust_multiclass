# Robust Multiclass SVM Experiments

This repository compares a standard nominal multiclass SVM against a robust
one-vs-one SVM under finite-shot kernel noise. It is set up as a small
experiment codebase: run a dataset, sweep measurement shot counts, and save
accuracy, reliability, reliability-confusion diagnostics, and plots.

## What the Code Does

- Builds an RBF precomputed kernel for a selected dataset.
- Optionally replaces the training kernel with a shared finite-shot stochastic
  kernel.
- Trains two classifiers on the same training kernel:
  - a nominal `sklearn.svm.SVC`
  - a robust one-vs-one SVM from `robust_multiclass.py`
- Simulates finite-shot test kernels for each value in `--shots`.
- Measures average accuracy and reliability across repeated trials.
- Saves arrays, CSV summaries, config files, and diagnostic plots under
  `results/`.

## Repository Layout

```text
.
|-- run_robust_vs_nominal.py      # Main experiment runner
|-- robust_multiclass.py          # Robust one-vs-one multiclass model
|-- math_utils.py                 # Kernel noise and robust optimization helpers
|-- reliability_diagnostics.py    # Reliability matrices and plotting helpers
|-- plot_accuracy_reliability.py  # Replot accuracy/reliability from saved arrays
|-- data/                         # Cached OpenML data
`-- results/                      # Timestamped experiment outputs
```

## Requirements

Use Python 3.10+ if possible. The scripts import:

```bash
pip install numpy scipy scikit-learn cvxpy matplotlib liac-arff
```

The default robust optimization path uses CVXPY with SCS. Some optional helper
functions in `math_utils.py` call MOSEK directly, but the main experiment with
the default settings does not require MOSEK.

## Quick Start

Run a small experiment:

```bash
python run_robust_vs_nominal.py \
  --dataset synthetic_overlap \
  --shots 2,32,256 \
  --N-trials 10
```

Run on the digits dataset with the same settings as the open config:

```bash
python run_robust_vs_nominal.py \
  --dataset digits \
  --train-kernel-shots 4000 \
  --shots 2,32,256 \
  --N-trials 50
```

Each run creates a directory like:

```text
results/20260626_235636_digits_rbf_train-T4000_eval-2-32-256/
```

Important outputs:

- `config.json`: full experiment settings
- `summary.csv`: compact metrics by shot count
- `results_arrays.npz`: NumPy arrays for deeper analysis
- `plots/`: reliability-confusion plots and, for newer runs, the
  accuracy/reliability curve plot

## Datasets

The runner supports:

- `synthetic_overlap`: generated multiclass classification data
- `digits`: scikit-learn handwritten digits
- `digits_noisy`: digits with Gaussian feature noise
- `vehicle`: OpenML/UCI Statlog Vehicle Silhouettes

The `vehicle` dataset is fetched through scikit-learn/OpenML and cached under
`data/openml/`.

## Main Experiment Options

```bash
python run_robust_vs_nominal.py --help
```

Common options:

- `--dataset`: one of `digits`, `digits_noisy`, `synthetic_overlap`, `vehicle`
- `--shots`: comma-separated finite-shot evaluation budgets, for example
  `2,4,8,16,32,64,128,256`
- `--train-kernel-shots`: finite-shot training kernel budget; use `0` for exact
  training kernel
- `--robust-uncertainty-mode`: `eval`, `train`, or `inf`
- `--N-trials`: number of finite-shot test-kernel trials per shot count
- `--C`, `--delta-1`, `--delta-2`, `--conf-int`: robust SVM parameters
- `--results-root`: output directory root, default `results`

## Replotting Saved Results

To regenerate the accuracy/reliability curve from a saved run:

```bash
python plot_accuracy_reliability.py \
  results/<run-directory>/results_arrays.npz
```

By default, the plot is saved to:

```text
results/<run-directory>/plots/accuracy_reliability_vs_shots.png
```

## Interpreting Reliability Diagnostics

The reliability-confusion matrix is not the usual accuracy confusion matrix.
Rows are ideal/noiseless predictions and columns are finite-shot predictions.
A strong diagonal means finite-shot predictions are stable relative to the
ideal prediction. Off-diagonal entries show which labels the classifier drifts
toward under finite-shot kernel noise.
