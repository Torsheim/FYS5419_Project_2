#!/usr/bin/env python3
"""Repeated-seed robustness study for the binary Iris QML experiment.

This script addresses the main statistical weakness of the single-split report:
it repeats training for several random train/test splits and initializations,
then reports mean +/- standard deviation for the quantum circuits and logistic
regression.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fys5419_project_2.baselines import logistic_regression_baseline
from fys5419_project_2.data import load_binary_iris_data
from fys5419_project_2.io_utils import save_json
from fys5419_project_2.model import QuantumCircuitClassifier
from fys5419_project_2.optimizers import train_adam


QUANTUM_CONFIGS = [
    ("h_rz", "simple"),
    ("ry", "simple"),
    ("ry_rz", "strong"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-seeds", type=int, default=5, help="Use seeds 0, ..., n_seeds-1")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Explicit seed list. Overrides --n-seeds when supplied.",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args()


def std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    summary = []
    for model_name, model_rows in grouped.items():
        first = model_rows[0]
        acc = [float(r["test_accuracy"]) for r in model_rows]
        loss = [float(r["test_loss"]) for r in model_rows]
        train_acc = [float(r["train_accuracy"]) for r in model_rows]
        train_loss = [float(r["train_loss"]) for r in model_rows]
        summary.append(
            {
                "model": model_name,
                "encoding": first.get("encoding", "logistic"),
                "ansatz": first.get("ansatz", "regression"),
                "n_parameters": first.get("n_parameters", ""),
                "n_runs": len(model_rows),
                "train_accuracy_mean": float(mean(train_acc)),
                "train_accuracy_std": std(train_acc),
                "test_accuracy_mean": float(mean(acc)),
                "test_accuracy_std": std(acc),
                "train_loss_mean": float(mean(train_loss)),
                "train_loss_std": std(train_loss),
                "test_loss_mean": float(mean(loss)),
                "test_loss_std": std(loss),
            }
        )
    order = {"h_rz/simple": 0, "ry/simple": 1, "ry_rz/strong": 2, "logistic/regression": 3}
    return sorted(summary, key=lambda r: order.get(str(r["model"]), 99))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: list[dict[str, Any]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(r["model"]) for r in summary]
    x = list(range(len(labels)))

    plt.figure(figsize=(8.0, 4.8))
    plt.bar(x, [float(r["test_accuracy_mean"]) for r in summary], yerr=[float(r["test_accuracy_std"]) for r in summary], capsize=4)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Test accuracy")
    plt.title("Iris repeated-seed study: test accuracy")
    plt.tight_layout()
    plt.savefig(figures_dir / "iris_repeated_seed_accuracy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8.0, 4.8))
    plt.bar(x, [float(r["test_loss_mean"]) for r in summary], yerr=[float(r["test_loss_std"]) for r in summary], capsize=4)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Test binary cross-entropy")
    plt.title("Iris repeated-seed study: test loss")
    plt.tight_layout()
    plt.savefig(figures_dir / "iris_repeated_seed_loss.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds else list(range(args.n_seeds))
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        split = load_binary_iris_data(n_features=args.features, random_state=seed)

        for encoding, ansatz in QUANTUM_CONFIGS:
            model = QuantumCircuitClassifier(
                n_qubits=args.features,
                n_layers=args.layers,
                encoding=encoding,
                ansatz=ansatz,
                measured_qubit=args.features - 1,
            )
            theta0 = model.initial_parameters(seed=seed, distribution="small")
            result = train_adam(
                model,
                split.X_train,
                split.y_train,
                theta0,
                X_val=split.X_test,
                y_val=split.y_test,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                verbose=False,
            )
            final = result.history[-1]
            row = {
                "seed": seed,
                "model": f"{encoding}/{ansatz}",
                "encoding": encoding,
                "ansatz": ansatz,
                "n_parameters": model.n_parameters,
                "train_accuracy": final["train_accuracy"],
                "test_accuracy": final["test_accuracy"],
                "train_loss": final["train_loss"],
                "test_loss": final["test_loss"],
            }
            rows.append(row)
            print(
                f"seed={seed:2d} {row['model']:14s} "
                f"test_acc={row['test_accuracy']:.3f} test_loss={row['test_loss']:.3f}"
            )

        logistic = logistic_regression_baseline(
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            random_state=seed,
        )
        rows.append(
            {
                "seed": seed,
                "model": "logistic/regression",
                "encoding": "logistic",
                "ansatz": "regression",
                "n_parameters": "",
                "train_accuracy": logistic["train_accuracy"],
                "test_accuracy": logistic["test_accuracy"],
                "train_loss": logistic["train_loss"],
                "test_loss": logistic["test_loss"],
            }
        )
        print(
            f"seed={seed:2d} {'logistic/regression':14s} "
            f"test_acc={logistic['test_accuracy']:.3f} test_loss={logistic['test_loss']:.3f}"
        )

    summary = summarize(rows)
    write_csv(rows, results_dir / "iris_repeated_seed_study.csv")
    write_csv(summary, results_dir / "iris_repeated_seed_summary.csv")
    save_json(
        {
            "experiment": "iris_repeated_seed_study",
            "config": {
                "features": args.features,
                "layers": args.layers,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "seeds": seeds,
            },
            "summary": summary,
            "runs": rows,
        },
        results_dir / "iris_repeated_seed_study.json",
    )
    plot_summary(summary, figures_dir)

    print("\nSummary: mean +/- standard deviation")
    for row in summary:
        print(
            f"{row['model']:20s} "
            f"acc={row['test_accuracy_mean']:.3f} +/- {row['test_accuracy_std']:.3f}  "
            f"loss={row['test_loss_mean']:.3f} +/- {row['test_loss_std']:.3f}"
        )
    print(f"\nSaved repeated-seed results to {results_dir}")
    print(f"Saved repeated-seed figures to {figures_dir}")


if __name__ == "__main__":
    main()
