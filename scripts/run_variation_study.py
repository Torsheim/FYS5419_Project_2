#!/usr/bin/env python3
"""Compare several encoders and ansaetze for Project 2f.

The script writes one JSON file and one CSV-style summary to results/.  It is
small enough to run on a laptop because it uses only a few qubits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fys5419_project_2.baselines import logistic_regression_baseline
from fys5419_project_2.data import load_binary_iris_data
from fys5419_project_2.io_utils import save_json
from fys5419_project_2.model import QuantumCircuitClassifier
from fys5419_project_2.optimizers import train_adam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = load_binary_iris_data(n_features=args.features, random_state=args.seed)
    combinations = [
        ("h_rz", "simple"),
        ("ry", "simple"),
        ("ry_rz", "simple"),
        ("h_rz", "strong"),
        ("ry_rz", "strong"),
    ]
    rows = []
    for encoding, ansatz in combinations:
        model = QuantumCircuitClassifier(
            n_qubits=args.features,
            n_layers=args.layers,
            encoding=encoding,
            ansatz=ansatz,
            measured_qubit=args.features - 1,
        )
        theta0 = model.initial_parameters(seed=args.seed, distribution="small")
        result = train_adam(
            model,
            split.X_train,
            split.y_train,
            theta0,
            X_val=split.X_test,
            y_val=split.y_test,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        final = result.history[-1]
        rows.append(
            {
                "encoding": encoding,
                "ansatz": ansatz,
                "n_parameters": model.n_parameters,
                "train_loss": final["train_loss"],
                "test_loss": final["test_loss"],
                "train_accuracy": final["train_accuracy"],
                "test_accuracy": final["test_accuracy"],
            }
        )
        print(
            f"{encoding:6s} {ansatz:6s} "
            f"test_acc={final['test_accuracy']:.3f} test_loss={final['test_loss']:.6f}"
        )

    logistic = logistic_regression_baseline(
        split.X_train, split.y_train, split.X_test, split.y_test, random_state=args.seed
    )
    output = {
        "experiment": "iris_variation_study",
        "config": {
            "features": args.features,
            "layers": args.layers,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "quantum_models": rows,
        "logistic_regression": logistic,
    }
    out_path = Path(args.results_dir) / "iris_variation_study.json"
    save_json(output, out_path)
    print(f"Logistic baseline test_acc={logistic['test_accuracy']:.3f}")
    print(f"Saved variation-study results to {out_path}")


if __name__ == "__main__":
    main()
