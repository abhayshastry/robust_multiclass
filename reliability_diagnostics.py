import os

import numpy as np


def _use_headless_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")


def _as_trial_predictions(y_pred_trials):
    y_pred_trials = np.asarray(y_pred_trials)
    if y_pred_trials.ndim != 2:
        raise ValueError("y_pred_trials must have shape (n_trials, n_samples)")
    return y_pred_trials


def _as_reference_predictions(y_ref):
    y_ref = np.asarray(y_ref)
    if y_ref.ndim != 1:
        raise ValueError("y_ref must have shape (n_samples,)")
    return y_ref


def _resolve_labels(y_pred_trials, y_ref, labels):
    if labels is None:
        labels = np.unique(np.concatenate([y_ref.ravel(), y_pred_trials.ravel()]))
    else:
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
    return labels


def reliability_confusion_counts(y_pred_trials, y_ref, labels=None):
    """
    Count finite-shot label drift relative to ideal/noiseless predictions.

    counts[row, col] is the number of times a sample whose ideal prediction is
    labels[row] is predicted as labels[col] across all finite-shot trials.
    """
    y_pred_trials = _as_trial_predictions(y_pred_trials)
    y_ref = _as_reference_predictions(y_ref)
    if y_pred_trials.shape[1] != y_ref.shape[0]:
        raise ValueError("y_pred_trials and y_ref disagree on n_samples")

    labels = _resolve_labels(y_pred_trials, y_ref, labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=int)

    for ref_label, pred_label in zip(
        np.repeat(y_ref, y_pred_trials.shape[0]),
        y_pred_trials.T.ravel(),
    ):
        if ref_label not in label_to_idx:
            raise ValueError(f"reference label {ref_label!r} is not in labels")
        if pred_label not in label_to_idx:
            raise ValueError(f"predicted label {pred_label!r} is not in labels")
        counts[label_to_idx[ref_label], label_to_idx[pred_label]] += 1

    return counts, labels


def normalize_rows(counts):
    """
    Row-normalize a count matrix. Rows with zero total remain zero.
    """
    counts = np.asarray(counts)
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.zeros(counts.shape, dtype=float)
    np.divide(counts, row_totals, out=normalized, where=row_totals != 0)
    return normalized


def reliability_confusion_matrix(y_pred_trials, y_ref, labels=None, normalize=True):
    """
    Return raw counts or row-normalized reliability confusion matrix.

    Rows are ideal/noiseless predictions f*(x); columns are finite-shot
    predictions f^(N)(x). This is a label-drift diagnostic, not an accuracy
    confusion matrix against the true labels.
    """
    counts, labels = reliability_confusion_counts(y_pred_trials, y_ref, labels)
    if normalize:
        return normalize_rows(counts), labels
    return counts, labels


def conditional_failure_matrix(y_pred_trials, y_ref, labels=None):
    """
    Return row-normalized off-diagonal failure matrix.

    Entry [a, b] is the probability of drifting to label b, conditioned on the
    ideal prediction being label a and the finite-shot prediction being wrong.
    Rows with no failures remain zero and the diagonal is always zero.
    """
    counts, labels = reliability_confusion_counts(y_pred_trials, y_ref, labels)
    failure_counts = counts.astype(float)
    np.fill_diagonal(failure_counts, 0.0)
    return normalize_rows(failure_counts), labels


def per_class_reliability(y_pred_trials, y_ref, labels=None, threshold=0.99):
    """
    Class-wise thresholded reliability.

    For each ideal/noiseless reference class, compute the fraction of samples
    whose trial-wise agreement probability is at least threshold.
    """
    y_pred_trials = _as_trial_predictions(y_pred_trials)
    y_ref = _as_reference_predictions(y_ref)
    if y_pred_trials.shape[1] != y_ref.shape[0]:
        raise ValueError("y_pred_trials and y_ref disagree on n_samples")

    labels = _resolve_labels(y_pred_trials, y_ref, labels)
    agreement_rate = (y_pred_trials == y_ref[None, :]).mean(axis=0)
    reliability = np.zeros(len(labels), dtype=float)

    for idx, label in enumerate(labels):
        mask = y_ref == label
        if np.any(mask):
            reliability[idx] = np.mean(agreement_rate[mask] >= threshold)

    return reliability, labels


def plot_reliability_confusion_matrix(
    matrix,
    labels,
    title,
    output_path=None,
    fmt=".2f",
):
    """
    Plot a heatmap for a row-normalized reliability confusion matrix.
    """
    _use_headless_matplotlib()
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix)
    labels = np.asarray(labels)

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel(r"finite-shot prediction $f^{(N)}(x)$")
    ax.set_ylabel(r"ideal prediction $f^*(x)$")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    if matrix.size <= 400:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = format(matrix[i, j], fmt)
                color = "white" if matrix[i, j] > 0.5 else "black"
                ax.text(j, i, value, ha="center", va="center", color=color)

    fig.tight_layout()
    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    return fig, ax


def plot_accuracy_reliability_curves(
    shots_array,
    accuracy_nominal,
    accuracy_robust,
    reliability_nominal,
    reliability_robust,
    output_path=None,
):
    """
    Plot average accuracy and scalar reliability against measurement shots.
    """
    _use_headless_matplotlib()
    import matplotlib.pyplot as plt

    shots_array = np.asarray(shots_array)
    accuracy_nominal = np.asarray(accuracy_nominal)
    accuracy_robust = np.asarray(accuracy_robust)
    reliability_nominal = np.asarray(reliability_nominal)
    reliability_robust = np.asarray(reliability_robust)

    nominal_accuracy_range = None
    robust_accuracy_range = None
    if accuracy_nominal.ndim == 2:
        nominal_accuracy_range = (
            accuracy_nominal.min(axis=1),
            accuracy_nominal.max(axis=1),
        )
        accuracy_nominal = accuracy_nominal.mean(axis=1)
    if accuracy_robust.ndim == 2:
        robust_accuracy_range = (
            accuracy_robust.min(axis=1),
            accuracy_robust.max(axis=1),
        )
        accuracy_robust = accuracy_robust.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    curves = [
        (
            axes[0],
            accuracy_nominal,
            accuracy_robust,
            "Average accuracy",
            "accuracy",
        ),
        (
            axes[1],
            reliability_nominal,
            reliability_robust,
            "Scalar reliability",
            "reliability",
        ),
    ]

    for ax, nominal_values, robust_values, title, ylabel in curves:
        nominal_line = ax.plot(shots_array, nominal_values, marker="o", label="Nominal")[0]
        robust_line = ax.plot(shots_array, robust_values, marker="s", label="Robust")[0]
        if ylabel == "accuracy":
            if nominal_accuracy_range is not None:
                ax.fill_between(
                    shots_array,
                    nominal_accuracy_range[0],
                    nominal_accuracy_range[1],
                    color=nominal_line.get_color(),
                    alpha=0.18,
                    linewidth=0,
                    label="Nominal range",
                )
            if robust_accuracy_range is not None:
                ax.fill_between(
                    shots_array,
                    robust_accuracy_range[0],
                    robust_accuracy_range[1],
                    color=robust_line.get_color(),
                    alpha=0.18,
                    linewidth=0,
                    label="Robust range",
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(shots_array, labels=shots_array)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(title)
        ax.set_xlabel("shots")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    return fig, axes


def close_figure(fig):
    """
    Close a Matplotlib figure returned by a plotting helper.
    """
    import matplotlib.pyplot as plt

    plt.close(fig)
