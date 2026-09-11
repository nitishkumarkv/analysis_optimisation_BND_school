#!/usr/bin/env python3
"""
Simple GATO optimisation example for the BND Graduate School.

The script loads nominal multi-class classifier scores, removes one redundant
softmax dimension, and optimises GATO categories by maximising the expected
Asimov significance for H->tautau.

By default, the input is:
    data_for_BND_school/data_5M/gato_like_scores_nominal.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import mplhep as hep
except ImportError:
    hep = None

from gatohep.losses import high_bkg_uncertainty_penalty, low_bkg_penalty
from gatohep.models import gato_gmm_model
from gatohep.plotting_utils import (
    assign_bins_and_order,
    fill_histogram_from_assignments,
    plot_history,
    plot_learned_gaussians,
    plot_stacked_histograms,
    plot_yield_vs_uncertainty,
)
from gatohep.utils import (
    LearningRateScheduler,
    TemperatureScheduler,
    compute_significance_from_hists,
    create_hist,
)

if hep is not None:
    plt.style.use(hep.style.ROOT)

SIGNAL_LABEL = "htautau"
DEFAULT_INPUT_FILE = "data_for_BND_school/data_5M/gato_like_scores_nominal.pkl"


def load_data(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load the nominal classifier-score pickle."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    print(f"Loading: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)

    if SIGNAL_LABEL not in data:
        raise KeyError(f"Signal '{SIGNAL_LABEL}' not found. Available: {list(data)}")

    return data


def reject_dimension(
    data: dict[str, pd.DataFrame],
    dim: int | None,
) -> dict[str, pd.DataFrame]:
    """Remove one redundant classifier-output dimension."""
    if dim is None:
        return data

    cleaned = {}
    for proc, df in data.items():
        df = df.copy()
        df["NN_output"] = df["NN_output"].apply(
            lambda x: np.delete(np.asarray(x), dim)
        )
        cleaned[proc] = df

    return cleaned


def convert_to_tensors(
    data: dict[str, pd.DataFrame],
) -> dict[str, dict[str, tf.Tensor]]:
    """Convert NN outputs and event weights to TensorFlow tensors."""
    tensor_data = {}

    for proc, df in data.items():
        tensor_data[proc] = {
            "NN_output": tf.constant(
                np.stack(df["NN_output"].values)[:, :3],
                dtype=tf.float32,
            ),
            "weight": tf.constant(
                df["weight"].values,
                dtype=tf.float32,
            ),
        }

    return tensor_data


class Gato3D(gato_gmm_model):
    """Three-dimensional GATO model optimising H->tautau significance."""

    def __init__(self, n_cats: int, temperature: float = 1.0):
        super().__init__(
            n_cats=n_cats,
            dim=3,
            temperature=temperature,
            mean_norm="softmax",
            name="gato_3D",
        )

    def call(self, data):
        significances, bkg_yield, bkg_sum_w2 = (
            self.get_differentiable_significance(
                data,
                signal_labels=[SIGNAL_LABEL],
                return_details=True,
            )
        )

        # TensorFlow minimises the loss, therefore use -Z_A.
        loss = -significances[SIGNAL_LABEL]
        return loss, bkg_yield, bkg_sum_w2


def plot_input_scores(
    data: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Plot the three classifier-score dimensions used by GATO."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for dim in range(3):
        hists = {}

        for proc, df in data.items():
            values = np.stack(df["NN_output"].values)[:, dim]
            hists[proc] = create_hist(
                values,
                df["weight"].values,
                bins=25,
                low=0.0,
                high=1.0,
            )

        backgrounds = [p for p in data if p != SIGNAL_LABEL]

        plot_stacked_histograms(
            stacked_hists=[hists[p] for p in backgrounds],
            process_labels=backgrounds,
            signal_hists=[100 * hists[SIGNAL_LABEL]],
            signal_labels=[f"{SIGNAL_LABEL} x100"],
            output_filename=str(output_dir / f"score_dim{dim}.pdf"),
            axis_labels=(f"Softmax dimension {dim}", "Events"),
            log=True,
        )


def evaluate_model(
    model: Gato3D,
    data: dict[str, pd.DataFrame],
    n_cats: int,
):
    """Evaluate the trained model using hard category assignments."""
    assignments, order, _, inv = assign_bins_and_order(
        model,
        data,
        reduce=True,
    )

    filled = {
        proc: fill_histogram_from_assignments(
            assignments[proc],
            data[proc]["weight"],
            n_cats,
        )
        for proc in data
    }

    backgrounds = [p for p in data if p != SIGNAL_LABEL]

    significance = compute_significance_from_hists(
        filled[SIGNAL_LABEL],
        [filled[p] for p in backgrounds],
    )

    return filled, backgrounds, order, inv, float(significance)


def run_gato(
    data: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    output_dir = Path(args.save_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_data = convert_to_tensors(data)
    plot_input_scores(data, output_dir / "input_scores")

    for n_cats in args.gato_bins:
        print(f"\nTraining GATO with {n_cats} categories")

        model = Gato3D(
            n_cats=n_cats,
            temperature=args.temp_initial,
        )

        optimizer = tf.keras.optimizers.RMSprop(args.lr_initial)

        lr_scheduler = LearningRateScheduler(
            optimizer,
            lr_initial=args.lr_initial,
            lr_final=args.lr_final,
            total_epochs=args.epochs,
            mode="cosine",
        )

        temp_scheduler = TemperatureScheduler(
            model,
            t_initial=args.temp_initial,
            t_final=args.temp_final,
            total_epochs=args.epochs,
            mode="cosine",
        )

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                loss, bkg_yield, bkg_sum_w2 = model.call(tensor_data)

                # Avoid categories with too little background or
                # excessively large statistical uncertainty.
                penalty_yield = low_bkg_penalty(
                    bkg_yield,
                    threshold=args.thr_yield,
                )

                penalty_unc = high_bkg_uncertainty_penalty(
                    bkg_sum_w2,
                    bkg_yield,
                    rel_threshold=args.thr_unc,
                )

                total_loss = (
                    loss
                    + args.lam_yield * penalty_yield
                    + args.lam_unc * penalty_unc
                )

            gradients = tape.gradient(
                total_loss,
                model.trainable_variables,
            )
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )

            return loss, total_loss

        loss_history = []

        for epoch in range(args.epochs):
            lr_scheduler.update(epoch)
            temp_scheduler.update(epoch)

            loss, total_loss = train_step()
            loss_history.append(float(loss.numpy()))

            if epoch % args.print_interval == 0:
                _, _, _, _, significance = evaluate_model(
                    model,
                    data,
                    n_cats,
                )

                print(
                    f"Epoch {epoch:4d} | "
                    f"loss = {loss.numpy():.4f} | "
                    f"Z = {significance:.4f} | "
                    f"T = {float(model.temperature):.3f}"
                )

        # Final hard-category evaluation
        filled, backgrounds, order, inv, significance = evaluate_model(
            model,
            data,
            n_cats,
        )

        print(
            f"Final significance with {n_cats} categories: "
            f"Z = {significance:.4f}"
        )

        model_dir = output_dir / "checkpoints" / f"{n_cats}_bins"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir))

        plot_history(
            np.asarray(loss_history),
            str(output_dir / f"loss_{n_cats}bins.pdf"),
            y_label=r"GATO objective $-Z_A$",
            x_label="Epoch",
        )

        plot_stacked_histograms(
            stacked_hists=[filled[p] for p in backgrounds],
            process_labels=backgrounds,
            signal_hists=[100 * filled[SIGNAL_LABEL]],
            signal_labels=[f"{SIGNAL_LABEL} x100"],
            output_filename=str(
                output_dir / f"optimized_{n_cats}bins.pdf"
            ),
            axis_labels=("GATO category", "Events"),
            normalize=False,
            log=True,
        )

        plot_learned_gaussians(
            data=data,
            model=model,
            dim_x=0,
            dim_y=1,
            output_filename=str(
                output_dir / f"gaussians_{n_cats}bins_dim01.pdf"
            ),
            inv_mapping=inv,
        )

        background_yield, rel_unc, _ = model.compute_hard_bkg_stats(
            tensor_data
        )

        plot_yield_vs_uncertainty(
            background_yield[order],
            rel_unc[order],
            log=False,
            output_filename=str(
                output_dir / f"yield_vs_unc_{n_cats}bins.pdf"
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nominal GATO optimisation example for the BND school."
    )

    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help="Nominal classifier-score pickle.",
    )

    parser.add_argument(
        "--save-folder",
        default="BND_gato_nominal",
    )

    parser.add_argument(
        "--gato-bins",
        nargs="+",
        type=int,
        default=[3, 10],
    )

    parser.add_argument("--epochs", type=int, default=500)

    # Penalties used to avoid poorly populated categories.
    parser.add_argument("--lam-yield", type=float, default=0.02)
    parser.add_argument("--lam-unc", type=float, default=0.02)
    parser.add_argument("--thr-yield", type=float, default=1.0)
    parser.add_argument("--thr-unc", type=float, default=0.5)

    # Remove one softmax dimension because the outputs sum to one.
    parser.add_argument("--reject-dim", type=int, default=1)

    parser.add_argument("--lr-initial", type=float, default=0.05)
    parser.add_argument("--lr-final", type=float, default=0.001)
    parser.add_argument("--temp-initial", type=float, default=1.0)
    parser.add_argument("--temp-final", type=float, default=0.05)

    parser.add_argument("--print-interval", type=int, default=25)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_data(args.input_file)

    reject_dim = None if args.reject_dim < 0 else args.reject_dim
    data = reject_dimension(data, reject_dim)

    run_gato(data, args)


if __name__ == "__main__":
    main()