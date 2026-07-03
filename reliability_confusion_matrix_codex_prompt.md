# Codex task: Add reliability confusion matrices to the robust multiclass quantum-kernel code

## Context

This repository implements one-vs-one multiclass classification for robust quantum-kernel SVMs.
The current code already supports:

- nominal multiclass `sklearn.svm.SVC(kernel="precomputed")`;
- robust multiclass classification through a one-vs-one wrapper;
- repeated finite-shot/noisy test-kernel trials;
- ordinary accuracy curves;
- scalar reliability curves comparing noisy predictions to an ideal/noiseless reference prediction;
- ordinary confusion matrices against true labels.

The goal of this task is to add a **reliability confusion matrix** diagnostic.

A usual confusion matrix compares

\[
    y_{\rm true}(x)
    \quad \text{vs.} \quad
    f(x).
\]

The reliability confusion matrix should instead compare

\[
    f^\ast(x)
    \quad \text{vs.} \quad
    f^{(N)}(x),
\]

where:

- \(f^\ast(x)\) is the ideal/noiseless classifier prediction;
- \(f^{(N)}(x)\) is the finite-shot classifier prediction at shot budget \(N\).

This matrix tells us how finite-shot measurement noise moves predictions between labels.
It is a multiclass diagnostic for **noise-induced label drift**.

---

## Existing files and relevant functions

The repository currently has the following important files:

```text
robust_multiclass.py
math_utils.py
run_robust_vs_nominal.py
```

Relevant code structure:

### `robust_multiclass.py`

- `fit_robust_ovo(K_train, y_train, ...)`
  - Trains robust one-vs-one binary classifiers for all class pairs.
  - Returns a `RobustOVOModel`.

- `RobustOVOModel.predict(K_vt)`
  - Predicts multiclass labels using one-vs-one voting.
  - This mirrors the scikit-learn SVC multiclass convention.

### `math_utils.py`

- `skm(K_val, shots=shots, N_trials=N_trials)`
  - Produces noisy finite-shot versions of a kernel matrix.
  - Shape: `(N_trials, n_test, n_train)`.

- `robustness(y_pred, y_test, fraction=1)`
  - Existing scalar reliability-style function.
  - `y_pred` has shape `(N_trials, n_test)`.
  - `y_test` is currently used as the reference labels.
  - In the existing pipeline, this reference is the ideal/noiseless prediction, not necessarily the true label.

### `run_robust_vs_nominal.py`

- `train_val_rob(...)`
  - Runs shot sweeps.
  - Stores arrays:
    - `y_pred_nominal` with shape `(n_shots, N_trials, n_val)`;
    - `y_pred_robust` with shape `(n_shots, N_trials, n_val)`.
  - Currently returns only accuracy and scalar reliability arrays.
  - The prediction arrays are local variables, so the new diagnostics either need to be computed inside this function or returned optionally.

---

## Definitions to implement

Let:

- `y_ref` be the ideal/noiseless reference prediction, shape `(n_samples,)`;
- `y_pred_trials` be finite-shot predictions, shape `(n_trials, n_samples)`;
- `labels` be the ordered list of class labels.

Define the reliability confusion matrix

\[
    M^{(N)}_{ab}
    =
    \Pr\!\left(
        f^{(N)}(x)=b
        \mid
        f^\ast(x)=a
    \right).
\]

Empirically, aggregate over both samples and trials:

\[
    C^{(N)}_{ab}
    =
    \sum_{t=1}^{T}
    \sum_{i=1}^{m}
    \mathbf{1}\{y_{\rm ref,i}=a\}
    \mathbf{1}\{y^{(t)}_{{\rm pred},i}=b\}.
\]

The row-normalized matrix is

\[
    M^{(N)}_{ab}
    =
    \frac{C^{(N)}_{ab}}{\sum_b C^{(N)}_{ab}}.
\]

Interpretation:

- row `a`: ideal/noiseless predicted class is `a`;
- column `b`: finite-shot prediction is `b`;
- diagonal entries: class-wise stability under finite-shot noise;
- off-diagonal entries: most likely noise-induced label drift.

Also implement a conditional failure matrix

\[
    E^{(N)}_{ab}
    =
    \Pr\!\left(
        f^{(N)}(x)=b
        \mid
        f^\ast(x)=a,
        f^{(N)}(x)\neq a
    \right),
    \qquad b\neq a.
\]

Empirically, for `b != a`,

\[
    E^{(N)}_{ab}
    =
    \frac{C^{(N)}_{ab}}{\sum_{b\neq a} C^{(N)}_{ab}}.
\]

Set the diagonal of `E` to zero.
If a row has no failures, return zeros for that row and avoid division-by-zero warnings.

---

## Important distinction from the existing scalar reliability

The current scalar reliability function uses a thresholded pointwise criterion:

\[
    R(\mathcal D)
    =
    \frac{1}{|\mathcal D|}
    \sum_{x\in \mathcal D}
    \mathbf{1}_{R(x)\geq 1-\delta}.
\]

The row-normalized reliability confusion matrix instead aggregates all trial predictions.
Its diagonal gives the **mean class-wise agreement probability**, not the thresholded fraction of perfectly or nearly perfectly stable points.

Therefore, do **not** replace the existing scalar reliability with the diagonal average.
Add the reliability confusion matrix as an additional diagnostic.

Optionally, also implement per-class thresholded reliability:

\[
    R_a
    =
    \frac{1}{|\{i:y_{{\rm ref},i}=a\}|}
    \sum_{i:y_{{\rm ref},i}=a}
    \mathbf{1}\left[
        \frac{1}{T}\sum_t
        \mathbf{1}\{y^{(t)}_{{\rm pred},i}=y_{{\rm ref},i}\}
        \geq 1-\delta
    \right].
\]

This gives a class-wise version of the existing scalar reliability.

---

## Required implementation

### 1. Add helper functions

Add these functions either to `math_utils.py` or to a new module such as `reliability_diagnostics.py`.
Prefer a new module if it keeps the code cleaner.

Recommended function signatures:

```python
def reliability_confusion_counts(y_pred_trials, y_ref, labels=None):
    """
    Count matrix C[a,b] for reliability confusion.

    Parameters
    ----------
    y_pred_trials : array-like, shape (n_trials, n_samples)
        Finite-shot predictions.
    y_ref : array-like, shape (n_samples,)
        Ideal/noiseless reference predictions.
    labels : array-like or None
        Ordered class labels. If None, infer from union of y_ref and y_pred_trials.

    Returns
    -------
    counts : ndarray, shape (n_classes, n_classes)
        counts[row, col] counts how often reference class row is predicted as col.
    labels : ndarray
        Ordered labels used for rows/columns.
    """
```

```python
def normalize_rows(counts):
    """
    Row-normalize a count matrix. Rows with zero total should remain zero.
    """
```

```python
def reliability_confusion_matrix(y_pred_trials, y_ref, labels=None, normalize=True):
    """
    Return either raw counts or row-normalized reliability confusion matrix.
    """
```

```python
def conditional_failure_matrix(y_pred_trials, y_ref, labels=None):
    """
    Return row-normalized off-diagonal failure matrix.
    Diagonal should be zero.
    Rows with no failures should remain zero.
    """
```

```python
def per_class_reliability(y_pred_trials, y_ref, labels=None, threshold=0.99):
    """
    Class-wise thresholded reliability.
    For each reference class a, compute the fraction of samples with reference a
    whose trial-wise agreement probability is at least threshold.
    """
```

### 2. Add plotting utility

Add a simple heatmap plotting function. Use matplotlib only.

```python
def plot_reliability_confusion_matrix(
    matrix,
    labels,
    title,
    output_path=None,
    fmt=".2f",
):
    """
    Plot a heatmap for a row-normalized reliability confusion matrix.
    If output_path is provided, save the figure.
    Return fig, ax.
    """
```

Requirements:

- x-axis label: `finite-shot prediction $f^{(N)}(x)$`;
- y-axis label: `ideal prediction $f^*(x)$`;
- include colorbar;
- show numeric entries inside cells for small numbers of classes;
- allow saving to PNG.

### 3. Modify `train_val_rob`

Modify `train_val_rob(...)` in `run_robust_vs_nominal.py` to optionally compute and return reliability confusion diagnostics.

Recommended API:

```python
def train_val_rob(
    K_train,
    K_val,
    y_train,
    y_test,
    C=1.0,
    delta_1=0.01,
    delta_2=0.01,
    conf_int=0,
    return_predictions=False,
    compute_reliability_confusion=False,
    reliability_threshold=0.99,
):
    ...
```

If `return_predictions=True`, return the prediction arrays as well:

```python
return {
    "shots_array": shots_array,
    "accuracy_nominal": accuracy_nominal,
    "accuracy_robust": accuracy_robust,
    "reliability_nominal": reliability_nominal,
    "reliability_robust": reliability_robust,
    "y_ref": y_pred_nominal_exact,
    "y_pred_nominal": y_pred_nominal,
    "y_pred_robust": y_pred_robust,
}
```

If `compute_reliability_confusion=True`, also return:

```python
"reliability_confusion_nominal": list_or_array_by_shot,
"reliability_confusion_robust": list_or_array_by_shot,
"failure_confusion_nominal": list_or_array_by_shot,
"failure_confusion_robust": list_or_array_by_shot,
"per_class_reliability_nominal": list_or_array_by_shot,
"per_class_reliability_robust": list_or_array_by_shot,
"labels": labels,
```

Here each `reliability_confusion_*[idx]` corresponds to `shots_array[idx]`.

### 4. Preserve current behavior

Do not break the existing call:

```python
accuracy_nom, accuracy_rob, reliability_nom, reliability_rob = train_val_rob(...)
```

Either:

1. keep this return behavior when no new flags are passed; or
2. update the script consistently and clearly.

Prefer option 1 for backward compatibility.

### 5. Add a small demo in `run_robust_vs_nominal.py`

After the shot sweep, add an example that prints the reliability confusion matrix for one selected shot budget.

For example:

```python
results = train_val_rob(
    K_train,
    K_test,
    y_train,
    y_test,
    C=1,
    conf_int=0.0,
    return_predictions=True,
    compute_reliability_confusion=True,
)

shots_array = results["shots_array"]
idx = 0  # lowest shot budget, or choose another value
print("Shots:", shots_array[idx])
print("Nominal reliability confusion matrix:")
print(results["reliability_confusion_nominal"][idx])
print("Robust reliability confusion matrix:")
print(results["reliability_confusion_robust"][idx])
```

Also save heatmaps if matplotlib is available:

```python
plot_reliability_confusion_matrix(
    results["reliability_confusion_nominal"][idx],
    results["labels"],
    title=f"Nominal reliability confusion, shots={shots_array[idx]}",
    output_path=f"results/nominal_reliability_confusion_shots_{shots_array[idx]}.png",
)

plot_reliability_confusion_matrix(
    results["reliability_confusion_robust"][idx],
    results["labels"],
    title=f"Robust reliability confusion, shots={shots_array[idx]}",
    output_path=f"results/robust_reliability_confusion_shots_{shots_array[idx]}.png",
)
```

---

## Acceptance criteria

The task is complete when:

1. The code computes the ordinary confusion matrix as before.
2. The code computes reliability confusion matrices for both nominal and robust classifiers.
3. The reliability confusion matrix rows correspond to ideal/noiseless labels.
4. The reliability confusion matrix columns correspond to finite-shot labels.
5. Row-normalized matrices have rows summing to one whenever that reference class appears.
6. The diagonal entries can be interpreted as class-wise mean stability under finite-shot noise.
7. A conditional failure matrix is available to identify the most likely wrong label when a class drifts.
8. The existing scalar reliability curves remain unchanged.
9. Existing behavior of `train_val_rob` is preserved unless explicitly using new flags.
10. At least one printed or saved example is produced for nominal and robust classifiers at a chosen shot count.

---

## Suggested sanity checks

Add quick checks in the demo or in a small test function:

```python
M, labels = reliability_confusion_matrix(y_pred_trials, y_ref, normalize=True)
row_sums = M.sum(axis=1)
assert np.allclose(row_sums[row_sums > 0], 1.0)
```

For a noiseless case:

```python
y_pred_trials = np.tile(y_ref, (n_trials, 1))
M, labels = reliability_confusion_matrix(y_pred_trials, y_ref, normalize=True)
assert np.allclose(M, np.eye(len(labels)))
```

For the conditional failure matrix:

```python
E, labels = conditional_failure_matrix(y_pred_trials, y_ref)
assert np.allclose(np.diag(E), 0.0)
```

---

## Reporting language

Use this description in comments or documentation:

> The reliability confusion matrix is not a standard accuracy confusion matrix. A standard confusion matrix compares true labels with predicted labels. The reliability confusion matrix compares the ideal/noiseless classifier prediction with the finite-shot classifier prediction. It therefore measures noise-induced label drift rather than classifier error against ground truth.

