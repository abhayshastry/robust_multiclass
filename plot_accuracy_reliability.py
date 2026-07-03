import argparse
import os

import numpy as np

from reliability_diagnostics import close_figure, plot_accuracy_reliability_curves


def plot_saved_results(results_path, output_path=None):
    data = np.load(results_path)
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(results_path),
            "plots",
            "accuracy_reliability_vs_shots.png",
        )

    fig, _ = plot_accuracy_reliability_curves(
        data["shots_array"],
        data["accuracy_nominal"],
        data["accuracy_robust"],
        data["reliability_nominal"],
        data["reliability_robust"],
        output_path=output_path,
    )
    close_figure(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot nominal/robust accuracy and scalar reliability vs shots."
    )
    parser.add_argument("results_path", help="Path to results_arrays.npz")
    parser.add_argument("--output", help="Output image path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = plot_saved_results(args.results_path, output_path=args.output)
    print(f"Saved plot to {path}")
