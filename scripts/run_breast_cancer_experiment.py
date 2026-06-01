#!/usr/bin/env python3
"""Optional breast-cancer experiment for the more challenging Project 2f task."""

from __future__ import annotations

import argparse
from pathlib import Path

from fys5419_project_2.baselines import logistic_regression_baseline
from fys5419_project_2.data import load_breast_cancer_data
from fys5419_project_2.io_utils import save_history_csv, save_json
from fys5419_project_2.model import QuantumCircuitClassifier
from fys5419_project_2.optimizers import train_adam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--encoding", choices=["h_rz", "ry", "ry_rz"], default="ry_rz")
    parser.add_argument("--ansatz", choices=["simple", "strong"], default="strong")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = load_breast_cancer_data(n_features=args.features, random_state=args.seed)
    model = QuantumCircuitClassifier(
        n_qubits=args.features,
        n_layers=args.layers,
        encoding=args.encoding,
        ansatz=args.ansatz,
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
        verbose=args.verbose,
    )
    logistic = logistic_regression_baseline(
        split.X_train, split.y_train, split.X_test, split.y_test, random_state=args.seed
    )
    final = result.history[-1]
    output = {
        "experiment": "breast_cancer_optional",
        "config": vars(args) | {"n_parameters": model.n_parameters},
        "quantum_model": {
            "train_loss": final["train_loss"],
            "test_loss": final["test_loss"],
            "train_accuracy": final["train_accuracy"],
            "test_accuracy": final["test_accuracy"],
        },
        "logistic_regression": logistic,
    }
    stem = f"breast_cancer_{args.encoding}_{args.ansatz}_{args.features}features_{args.layers}layers"
    results_dir = Path(args.results_dir)
    save_json(output, results_dir / f"{stem}_metrics.json")
    save_history_csv(result.history, results_dir / f"{stem}_history.csv")
    print("Finished optional breast-cancer experiment")
    print(f"Quantum model test accuracy:      {final['test_accuracy']:.3f}")
    print(f"Logistic regression test accuracy:{logistic['test_accuracy']:.3f}")


if __name__ == "__main__":
    main()
