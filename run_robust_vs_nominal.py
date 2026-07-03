# run_robust_vs_nominal_with_shots.py
import argparse
import json
import os
from datetime import datetime

import numpy as np
from sklearn.datasets import fetch_openml, load_digits, make_classification
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import math_utils as mu
from robust_multiclass import fit_robust_ovo
from reliability_diagnostics import (
    close_figure,
    conditional_failure_matrix,
    per_class_reliability,
    plot_accuracy_reliability_curves,
    plot_reliability_confusion_matrix,
    reliability_confusion_matrix,
)


def rbf_precomputed(X_train, X_test=None, gamma=None):
    if gamma is None:
        D2 = pairwise_distances(X_train, metric="euclidean", squared=True)
        med = np.median(D2[D2 > 0])
        gamma = 1.0 / (med + 1e-12)

    def kmat(A, B):
        A2 = (A * A).sum(axis=1)[:, None]
        B2 = (B * B).sum(axis=1)[None, :]
        sq = A2 + B2 - 2 * (A @ B.T)
        return np.exp(-gamma * sq)

    K_train = kmat(X_train, X_train)
    if X_test is None:
        return K_train, None, gamma
    return K_train, kmat(X_test, X_train), gamma


def make_shared_stochastic_training_kernel(K_train, shots, max_iter=20, eps=1e-10):
    K = mu.stochastic_kernel_matrix(K_train, shots=shots)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 1.0)

    # A finite-shot training kernel need not be PSD. Project it once into the
    # PSD, symmetric, unit-diagonal, [0, 1] kernel set before giving the exact
    # same matrix to both nominal and robust classifiers.
    for _ in range(max_iter):
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals = np.maximum(eigvals, eps)
        K = (eigvecs * eigvals) @ eigvecs.T
        diag = np.sqrt(np.maximum(np.diag(K), eps))
        K = K / diag[:, None] / diag[None, :]
        K = 0.5 * (K + K.T)
        K = np.clip(K, 0.0, 1.0)
        K = 0.5 * (K + K.T)
        np.fill_diagonal(K, 1.0)

    return K


def load_experiment_dataset(name, random_state=0):
    if name == "digits":
        X, y = load_digits(return_X_y=True)
        return X, y, {"dataset": name}

    if name == "digits_noisy":
        X, y = load_digits(return_X_y=True)
        rng = np.random.default_rng(random_state)
        X = X + rng.normal(loc=0.0, scale=4.0, size=X.shape)
        return X, y, {"dataset": name, "feature_noise_std": 4.0}

    if name == "synthetic_overlap":
        X, y = make_classification(
            n_samples=520,
            n_features=12,
            n_informative=5,
            n_redundant=3,
            n_repeated=0,
            n_classes=4,
            n_clusters_per_class=2,
            class_sep=0.75,
            flip_y=0.08,
            random_state=random_state,
        )
        return X, y, {
            "dataset": name,
            "n_samples": 520,
            "n_classes": 4,
            "class_sep": 0.75,
            "flip_y": 0.08,
        }

    if name == "vehicle":
        data_home = os.path.join(os.getcwd(), "data", "openml")
        data = fetch_openml(
            name="vehicle",
            version=1,
            as_frame=False,
            data_home=data_home,
            parser="liac-arff",
        )
        X = data.data
        y = data.target
        _, y = np.unique(y, return_inverse=True)
        return X, y, {
            "dataset": name,
            "source": "OpenML vehicle version 1",
            "description": "UCI Statlog Vehicle Silhouettes",
        }

    raise ValueError(f"Unknown dataset: {name}")


def make_run_dir(config, root="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_tag = "exact" if config["train_kernel_shots"] is None else f"T{config['train_kernel_shots']}"
    shots_tag = "-".join(str(s) for s in config["shots_array"])
    run_name = (
        f"{timestamp}_{config['dataset']}_{config['kernel']}_"
        f"train-{train_tag}_eval-{shots_tag}"
    )
    run_dir = os.path.join(root, run_name)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=False)
    return run_dir


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_summary_csv(path, results):
    rows = [
        "shots,accuracy_nominal_mean,accuracy_robust_mean,"
        "scalar_reliability_nominal,scalar_reliability_robust,"
        "mean_diag_reliability_confusion_nominal,"
        "mean_diag_reliability_confusion_robust\n"
    ]
    for idx, shots in enumerate(results["shots_array"]):
        mean_diag_nominal = np.diag(results["reliability_confusion_nominal"][idx]).mean()
        mean_diag_robust = np.diag(results["reliability_confusion_robust"][idx]).mean()
        rows.append(
            f"{shots},"
            f"{np.mean(results['accuracy_nominal'][idx]):.12g},"
            f"{np.mean(results['accuracy_robust'][idx]):.12g},"
            f"{results['reliability_nominal'][idx]:.12g},"
            f"{results['reliability_robust'][idx]:.12g},"
            f"{mean_diag_nominal:.12g},"
            f"{mean_diag_robust:.12g}\n"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(rows)


def train_test_rob(
    K_train,
    K_test,
    y_train,
    y_test,
    C=1.0,
    delta_1=0.01,
    delta_2=0.01,
    conf_int=0,
    return_predictions=False,
    compute_reliability_confusion=False,
    reliability_threshold=0.99,
    shots_array=None,
    N_trials=50,
    train_kernel_shots=np.inf,
    robust_uncertainty_mode="eval",
):
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if shots_array is None:
        shots_array = [2**n for n in range(1, 12)]
    else:
        shots_array = list(shots_array)
    s = len(shots_array)
    n_test = K_test.shape[0]

    y_pred_nominal = np.zeros((s, N_trials, n_test), dtype=y_test.dtype)
    y_pred_robust = np.zeros((s, N_trials, n_test), dtype=y_test.dtype)
    accuracy_nominal = np.zeros((s, N_trials))
    accuracy_robust = np.zeros((s, N_trials))
    reliability_nominal = np.zeros(s)
    reliability_robust = np.zeros(s)
    labels = np.unique(np.concatenate([y_train, y_test]))
    reliability_confusion_nominal = []
    reliability_confusion_robust = []
    failure_confusion_nominal = []
    failure_confusion_robust = []
    per_class_reliability_nominal = []
    per_class_reliability_robust = []

    if train_kernel_shots == np.inf:
        print("Shared training kernel: exact K_train", flush=True)
        print(
            "Note: T=inf gives a zero training-uncertainty radius; this is a "
            "sanity/equivalence run, not a robustness-enhanced setting.",
            flush=True,
        )
        K_train_fit = K_train
    else:
        print(f"Shared training kernel: stochastic K_train with T={train_kernel_shots}", flush=True)
        K_train_fit = make_shared_stochastic_training_kernel(K_train, shots=train_kernel_shots)

    nominal_SVC = SVC(kernel="precomputed", C=C).fit(K_train_fit, y_train)
    y_pred_nominal_exact = nominal_SVC.predict(K_test)
    robust_SVC_exact = fit_robust_ovo(
        K_train_fit,
        y_train,
        C=C,
        delta_1=delta_1,
        delta_2=delta_2,
        shots=np.inf,
        conf_int=conf_int,
    )
    y_pred_robust_exact = robust_SVC_exact.predict(K_test)

    for idx, shots in enumerate(shots_array):
        print(f"\n--- Shot snapshot {idx + 1}/{s}: shots={shots} ---", flush=True)
        if robust_uncertainty_mode == "eval":
            robust_uncertainty_shots = shots
        elif robust_uncertainty_mode == "train":
            robust_uncertainty_shots = train_kernel_shots
        elif robust_uncertainty_mode == "inf":
            robust_uncertainty_shots = np.inf
        else:
            raise ValueError(f"Unknown robust_uncertainty_mode: {robust_uncertainty_mode}")
        print(
            "Robust classifier uncertainty shots: "
            f"{'inf' if robust_uncertainty_shots == np.inf else robust_uncertainty_shots}",
            flush=True,
        )
        robust_SVC = fit_robust_ovo(
            K_train_fit,
            y_train,
            C=C,
            delta_1=delta_1,
            delta_2=delta_2,
            shots=robust_uncertainty_shots,
            conf_int=conf_int,
        )
        K_test_set = mu.skm(K_test, shots=shots, N_trials=N_trials)

        for t in range(N_trials):
            K_test_trial = K_test_set[t]
            y_pred_nominal[idx, t] = nominal_SVC.predict(K_test_trial)
            y_pred_robust[idx, t] = robust_SVC.predict(K_test_trial)
            accuracy_nominal[idx, t] = mu.accuracy_score(y_test, y_pred_nominal[idx, t])
            accuracy_robust[idx, t] = mu.accuracy_score(y_test, y_pred_robust[idx, t])

        reliability_nominal[idx] = mu.robustness(
            y_pred_nominal[idx, :], y_pred_nominal_exact, fraction=1
        )
        reliability_robust[idx] = mu.robustness(
            y_pred_robust[idx, :], y_pred_robust_exact, fraction=1
        )

        if compute_reliability_confusion:
            nominal_matrix, labels = reliability_confusion_matrix(
                y_pred_nominal[idx], y_pred_nominal_exact, labels=labels, normalize=True
            )
            robust_matrix, _ = reliability_confusion_matrix(
                y_pred_robust[idx], y_pred_robust_exact, labels=labels, normalize=True
            )
            nominal_failure, _ = conditional_failure_matrix(
                y_pred_nominal[idx], y_pred_nominal_exact, labels=labels
            )
            robust_failure, _ = conditional_failure_matrix(
                y_pred_robust[idx], y_pred_robust_exact, labels=labels
            )
            nominal_class_reliability, _ = per_class_reliability(
                y_pred_nominal[idx],
                y_pred_nominal_exact,
                labels=labels,
                threshold=reliability_threshold,
            )
            robust_class_reliability, _ = per_class_reliability(
                y_pred_robust[idx],
                y_pred_robust_exact,
                labels=labels,
                threshold=reliability_threshold,
            )

            reliability_confusion_nominal.append(nominal_matrix)
            reliability_confusion_robust.append(robust_matrix)
            failure_confusion_nominal.append(nominal_failure)
            failure_confusion_robust.append(robust_failure)
            per_class_reliability_nominal.append(nominal_class_reliability)
            per_class_reliability_robust.append(robust_class_reliability)

    if return_predictions or compute_reliability_confusion:
        results = {
            "shots_array": np.asarray(shots_array),
            "accuracy_nominal": accuracy_nominal,
            "accuracy_robust": accuracy_robust,
            "reliability_nominal": reliability_nominal,
            "reliability_robust": reliability_robust,
            "y_ref": y_pred_nominal_exact,
            "y_ref_nominal": y_pred_nominal_exact,
            "y_ref_robust": y_pred_robust_exact,
        }
        if return_predictions:
            results["y_pred_nominal"] = y_pred_nominal
            results["y_pred_robust"] = y_pred_robust
        if compute_reliability_confusion:
            results.update(
                {
                    "reliability_confusion_nominal": np.asarray(reliability_confusion_nominal),
                    "reliability_confusion_robust": np.asarray(reliability_confusion_robust),
                    "failure_confusion_nominal": np.asarray(failure_confusion_nominal),
                    "failure_confusion_robust": np.asarray(failure_confusion_robust),
                    "per_class_reliability_nominal": np.asarray(per_class_reliability_nominal),
                    "per_class_reliability_robust": np.asarray(per_class_reliability_robust),
                    "labels": labels,
                }
            )
        return results

    return accuracy_nominal, accuracy_robust, reliability_nominal, reliability_robust


# Backward-compatible alias for older scripts.
train_val_rob = train_test_rob


def run_experiment(args):
    X, y, dataset_meta = load_experiment_dataset(args.dataset, random_state=args.random_state)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    K_train, K_test, gamma = rbf_precomputed(X_train, X_test)
    train_kernel_shots = np.inf if args.train_kernel_shots == 0 else args.train_kernel_shots
    shots_array = [int(s) for s in args.shots.split(",") if s]

    config = {
        "dataset": args.dataset,
        "dataset_meta": dataset_meta,
        "kernel": "rbf",
        "gamma": gamma,
        "C": args.C,
        "delta_1": args.delta_1,
        "delta_2": args.delta_2,
        "conf_int": args.conf_int,
        "N_trials": args.N_trials,
        "shots_array": shots_array,
        "train_kernel_shots": None if train_kernel_shots == np.inf else train_kernel_shots,
        "robust_uncertainty_mode": args.robust_uncertainty_mode,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "classes": [int(c) if np.issubdtype(type(c), np.integer) else str(c) for c in np.unique(y)],
    }
    run_dir = make_run_dir(config, root=args.results_root)
    print(f"Run directory: {run_dir}", flush=True)
    save_json(os.path.join(run_dir, "config.json"), config)

    svc_nom = SVC(kernel="precomputed", C=args.C).fit(K_train, y_train)
    y_pred_nom_exact = svc_nom.predict(K_test)
    print("\n=== Nominal SVC baseline, exact training/eval kernel ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_nom_exact))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_nom_exact))

    robust_model = fit_robust_ovo(
        K_train,
        y_train,
        C=args.C,
        delta_1=args.delta_1,
        delta_2=args.delta_2,
        shots=np.inf,
        conf_int=args.conf_int,
    )
    y_pred_rob_exact = robust_model.predict(K_test)
    print("\n=== Robust SVC baseline, exact training/eval kernel ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_rob_exact))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rob_exact))

    results = train_test_rob(
        K_train,
        K_test,
        y_train,
        y_test,
        C=args.C,
        delta_1=args.delta_1,
        delta_2=args.delta_2,
        conf_int=args.conf_int,
        return_predictions=args.return_predictions,
        compute_reliability_confusion=True,
        shots_array=shots_array,
        N_trials=args.N_trials,
        train_kernel_shots=train_kernel_shots,
        robust_uncertainty_mode=args.robust_uncertainty_mode,
    )

    save_summary_csv(os.path.join(run_dir, "summary.csv"), results)
    np.savez_compressed(
        os.path.join(run_dir, "results_arrays.npz"),
        shots_array=results["shots_array"],
        accuracy_nominal=results["accuracy_nominal"],
        accuracy_robust=results["accuracy_robust"],
        reliability_nominal=results["reliability_nominal"],
        reliability_robust=results["reliability_robust"],
        reliability_confusion_nominal=results["reliability_confusion_nominal"],
        reliability_confusion_robust=results["reliability_confusion_robust"],
        failure_confusion_nominal=results["failure_confusion_nominal"],
        failure_confusion_robust=results["failure_confusion_robust"],
        per_class_reliability_nominal=results["per_class_reliability_nominal"],
        per_class_reliability_robust=results["per_class_reliability_robust"],
        y_ref_nominal=results["y_ref_nominal"],
        y_ref_robust=results["y_ref_robust"],
        labels=results["labels"],
    )

    print("\nAs a function of increasing measurement shots:")
    print("Average accuracy for nominal SVC\n", np.mean(results["accuracy_nominal"], axis=1))
    print("Average accuracy for ROBUST SVC\n", np.mean(results["accuracy_robust"], axis=1))
    print("Reliability for nominal SVC\n", results["reliability_nominal"])
    print("Reliability for ROBUST SVC\n", results["reliability_robust"])
    print(
        "Mean reliability-confusion diagonal for nominal SVC\n",
        np.asarray([np.diag(m).mean() for m in results["reliability_confusion_nominal"]]),
    )
    print(
        "Mean reliability-confusion diagonal for ROBUST SVC\n",
        np.asarray([np.diag(m).mean() for m in results["reliability_confusion_robust"]]),
    )
    fig, _ = plot_accuracy_reliability_curves(
        results["shots_array"],
        results["accuracy_nominal"],
        results["accuracy_robust"],
        results["reliability_nominal"],
        results["reliability_robust"],
        output_path=os.path.join(run_dir, "plots", "accuracy_reliability_vs_shots.png"),
    )
    close_figure(fig)

    print("\nReliability confusion diagnostic")
    print("Rows: ideal/noiseless prediction; columns: finite-shot prediction")
    print("Labels:", results["labels"])
    for idx, shots in enumerate(results["shots_array"]):
        print(f"\nShots: {shots}")
        print("Nominal reliability confusion matrix:")
        print(results["reliability_confusion_nominal"][idx])
        print("Robust reliability confusion matrix:")
        print(results["reliability_confusion_robust"][idx])

        fig, _ = plot_reliability_confusion_matrix(
            results["reliability_confusion_nominal"][idx],
            results["labels"],
            title=f"Nominal reliability confusion, shots={shots}",
            output_path=os.path.join(run_dir, "plots", f"nominal_reliability_confusion_shots_{shots}.png"),
        )
        close_figure(fig)
        fig, _ = plot_reliability_confusion_matrix(
            results["reliability_confusion_robust"][idx],
            results["labels"],
            title=f"Robust reliability confusion, shots={shots}",
            output_path=os.path.join(run_dir, "plots", f"robust_reliability_confusion_shots_{shots}.png"),
        )
        close_figure(fig)

    print(f"\nSaved config, arrays, summary, and plots under:\n{run_dir}")
    return run_dir, results


def parse_args():
    parser = argparse.ArgumentParser(description="Run nominal vs robust reliability experiments.")
    parser.add_argument(
        "--dataset",
        choices=["digits", "digits_noisy", "synthetic_overlap", "vehicle"],
        default="synthetic_overlap",
    )
    parser.add_argument("--shots", default="2,32,256")
    parser.add_argument("--train-kernel-shots", type=int, default=4000)
    parser.add_argument(
        "--robust-uncertainty-mode",
        choices=["eval", "train", "inf"],
        default="eval",
        help=(
            "How to set the shots parameter in fit_robust_ovo. "
            "'eval' matches the original shot-sweep script: robust is refit for each "
            "finite-shot evaluation budget. 'train' uses the shared training-kernel "
            "shot count. 'inf' forces zero uncertainty."
        ),
    )
    parser.add_argument("--N-trials", type=int, default=50)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--delta-1", type=float, default=0.01)
    parser.add_argument("--delta-2", type=float, default=0.01)
    parser.add_argument("--conf-int", type=float, default=0.0)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--return-predictions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
