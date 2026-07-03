# How to Use This Code

## 1. Install Dependencies

From the repository root:

```bash
pip install numpy scipy scikit-learn cvxpy matplotlib liac-arff
```

## 2. Run an Experiment

Fast smoke test:

```bash
python run_robust_vs_nominal.py --dataset synthetic_overlap --shots 2,32 --N-trials 3
```

Digits experiment:

```bash
python run_robust_vs_nominal.py \
  --dataset digits \
  --train-kernel-shots 4000 \
  --shots 2,32,256 \
  --N-trials 50
```

Use an exact training kernel by passing `--train-kernel-shots 0`:

```bash
python run_robust_vs_nominal.py \
  --dataset digits \
  --train-kernel-shots 0 \
  --shots 2,32,256
```

## 3. Find the Outputs

After the run finishes, look in the printed run directory under `results/`.
For example:

```text
results/20260626_235636_digits_rbf_train-T4000_eval-2-32-256/
```

The useful files are:

- `config.json`: settings used for the run
- `summary.csv`: accuracy and reliability summary by shot count
- `results_arrays.npz`: full saved NumPy arrays
- `plots/*.png`: generated diagnostic plots

## 4. Recreate the Main Plot

```bash
python plot_accuracy_reliability.py results/<run-directory>/results_arrays.npz
```

To choose the output path:

```bash
python plot_accuracy_reliability.py \
  results/<run-directory>/results_arrays.npz \
  --output my_plot.png
```

## 5. Useful Settings

- Increase `--N-trials` for smoother reliability estimates.
- Add more shot values with `--shots 2,4,8,16,32,64,128,256`.
- Use `--dataset vehicle` to run the OpenML vehicle dataset.
- Use `--results-root other_results` to keep a separate output folder.
- Use `--return-predictions` if you want trial-level predictions returned from
  `train_test_rob` in code.

## 6. Programmatic Use

The main callable pieces are:

```python
from run_robust_vs_nominal import run_experiment, train_test_rob
from robust_multiclass import fit_robust_ovo
```

For direct model fitting, pass a precomputed square training kernel and labels:

```python
model = fit_robust_ovo(K_train, y_train, C=1.0, delta_1=0.01, delta_2=0.01)
y_pred = model.predict(K_test)
```

`K_test` must have shape `(n_test, n_train)` and its columns must match the
training order used for `K_train`.

