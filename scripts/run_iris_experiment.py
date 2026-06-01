#!/usr/bin/env python3
"""Run the main Iris experiment for FYS5419 Project 2.

Example
-------
python scripts/run_iris_experiment.py --features 4 --layers 2 --epochs 20 --ansatz simple
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fys5419_project_2.baselines import logistic_regression_baseline
from fys5419_project_2.data import load_binary_iris_data
from fys5419_project_2.io_utils import save_history_csv, save_json, save_predictions_csv
from fys5419_project_2.model import QuantumCircuitClassifier
from fys5419_project_2.optimizers import train_adam, train_gradient_descent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=4, help="number of Iris features to use")
    parser.add_argument("--layers", type=int, default=2, help="number of ansatz layers")
    parser.add_argument(
        "--encoding",
        choices=["h_rz", "ry", "ry_rz"],
        default="h_rz",
        help="feature encoding circuit",
    )
    parser.add_argument(
        "--ansatz",
        choices=["simple", "strong"],
        default="simple",
        help="parameterized ansatz",
    )
    parser.add_argument("--epochs", type=int, default=20, help="training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="optimizer step size")
    parser.add_argument(
        "--optimizer",
        choices=["adam", "gd"],
        default="adam",
        help="optimizer: Adam or plain gradient descent",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--verbose", action="store_true", help="print progress every 10 epochs")
    parser.add_argument("--results-dir", default="results", help="where to save result files")
    parser.add_argument("--figures-dir", default="figures", help="where to save figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = load_binary_iris_data(n_features=args.features, random_state=args.seed)
    model = QuantumCircuitClassifier(
        n_qubits=args.features,
        n_layers=args.layers,
        encoding=args.encoding,
        ansatz=args.ansatz,
        measured_qubit=args.features - 1,
    )
    theta0 = model.initial_parameters(seed=args.seed, distribution="small")

    if args.optimizer == "adam":
        training = train_adam(
            model,
            split.X_train,
            split.y_train,
            theta0,
            X_val=split.X_test,
            y_val=split.y_test,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            verbose=args.verbose,
        )
    else:
        training = train_gradient_descent(
            model,
            split.X_train,
            split.y_train,
            theta0,
            X_val=split.X_test,
            y_val=split.y_test,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            verbose=args.verbose,
        )

    qml_probability_train = model.predict_proba(split.X_train, training.theta)
    qml_probability_test = model.predict_proba(split.X_test, training.theta)
    logistic = logistic_regression_baseline(
        split.X_train, split.y_train, split.X_test, split.y_test, random_state=args.seed
    )

    final = training.history[-1]
    metrics = {
        "experiment": "binary_iris",
        "config": {
            "features": args.features,
            "feature_names": split.feature_names,
            "target_names": split.target_names,
            "layers": args.layers,
            "encoding": args.encoding,
            "ansatz": args.ansatz,
            "measured_qubit": model.resolved_measured_qubit,
            "n_parameters": model.n_parameters,
            "epochs_requested": args.epochs,
            "epochs_completed": len(training.history),
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "seed": args.seed,
        },
        "quantum_model": {
            "train_loss": float(final["train_loss"]),
            "test_loss": float(final["test_loss"]),
            "train_accuracy": float(final["train_accuracy"]),
            "test_accuracy": float(final["test_accuracy"]),
        },
        "logistic_regression": logistic,
    }

    result_stem = f"iris_{args.encoding}_{args.ansatz}_{args.features}features_{args.layers}layers"
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    save_json(metrics, results_dir / f"{result_stem}_metrics.json")
    save_history_csv(training.history, results_dir / f"{result_stem}_history.csv")
    save_predictions_csv(split.y_train, qml_probability_train, results_dir / f"{result_stem}_train_predictions.csv")
    save_predictions_csv(split.y_test, qml_probability_test, results_dir / f"{result_stem}_test_predictions.csv")
    np.save(results_dir / f"{result_stem}_theta.npy", training.theta)

    plot_history(training.history, figures_dir / f"{result_stem}_loss.png")

    print("Finished Iris experiment")
    print(f"Quantum model test accuracy:      {metrics['quantum_model']['test_accuracy']:.3f}")
    print(f"Quantum model test loss:          {metrics['quantum_model']['test_loss']:.6f}")
    print(f"Logistic regression test accuracy:{logistic['test_accuracy']:.3f}")
    print(f"Saved metrics to: {results_dir / f'{result_stem}_metrics.json'}")
    print(f"Saved loss figure to: {figures_dir / f'{result_stem}_loss.png'}")


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    test_loss = [row.get("test_loss", np.nan) for row in history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, test_loss, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("Quantum classifier training loss on Iris")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
