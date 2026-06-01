#!/usr/bin/env python3
"""Finite-shot measurement study for the Iris quantum classifier.

The main report uses exact state-vector probabilities.  This script keeps the
exact-probability training but replaces the final test-set probabilities by
finite-shot Bernoulli estimates, mimicking qasm-style measurement noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fys5419_project_2.data import load_binary_iris_data
from fys5419_project_2.io_utils import save_json
from fys5419_project_2.model import QuantumCircuitClassifier
from fys5419_project_2.optimizers import train_adam
from fys5419_project_2.quantum_state import sample_measurements


DEFAULT_CONFIGS = [
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args()


def binary_cross_entropy(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-10) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y_true = np.asarray(y_true, dtype=float)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def accuracy(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) >= 0.5).astype(int) == np.asarray(y_true).astype(int)))


def std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_model(args: argparse.Namespace, encoding: str, ansatz: str, split) -> tuple[QuantumCircuitClassifier, np.ndarray]:
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
        verbose=False,
    )
    return model, result.theta


def sample_test_probabilities(
    model: QuantumCircuitClassifier,
    X_test: np.ndarray,
    theta: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    probabilities = []
    for x in X_test:
        state = model.statevector(x, theta)
        seed = int(rng.integers(0, 2**32 - 1))
        probabilities.append(
            sample_measurements(
                state,
                qubit=model.resolved_measured_qubit,
                n_qubits=model.n_qubits,
                shots=shots,
                seed=seed,
            )
        )
    return np.asarray(probabilities, dtype=float)


def plot_rows(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for row in rows:
        if row["model"] not in models:
            models.append(row["model"])
    shot_labels = ["exact"] + sorted({int(row["shots"]) for row in rows if row["shots"] != "exact"})

    for metric, ylabel, filename in [
        ("test_accuracy_mean", "Test accuracy", "iris_shot_noise_accuracy.png"),
        ("test_loss_mean", "Test binary cross-entropy", "iris_shot_noise_loss.png"),
    ]:
        plt.figure(figsize=(8.2, 4.8))
        for model_name in models:
            model_rows = [r for r in rows if r["model"] == model_name]
            values = []
            errors = []
            xs = []
            for idx, label in enumerate(shot_labels):
                if label == "exact":
                    match = [r for r in model_rows if r["shots"] == "exact"]
                else:
                    match = [r for r in model_rows if r["shots"] == label]
                if match:
                    xs.append(idx)
                    values.append(float(match[0][metric]))
                    errors.append(float(match[0][metric.replace("_mean", "_std")]))
            plt.errorbar(xs, values, yerr=errors, marker="o", capsize=4, label=model_name)
        plt.xticks(range(len(shot_labels)), [str(x) for x in shot_labels])
        plt.xlabel("Shots per test-sample measurement")
        plt.ylabel(ylabel)
        if "accuracy" in metric:
            plt.ylim(0.0, 1.05)
        plt.title("Iris finite-shot measurement study")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    split = load_binary_iris_data(n_features=args.features, random_state=args.seed)
    rng = np.random.default_rng(args.seed + 12345)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []

    for encoding, ansatz in DEFAULT_CONFIGS:
        model, theta = train_model(args, encoding, ansatz, split)
        model_name = f"{encoding}/{ansatz}"
        exact_probability = model.predict_proba(split.X_test, theta)
        exact_acc = accuracy(split.y_test, exact_probability)
        exact_loss = binary_cross_entropy(split.y_test, exact_probability)
        rows.append(
            {
                "model": model_name,
                "encoding": encoding,
                "ansatz": ansatz,
                "shots": "exact",
                "repeats": 1,
                "test_accuracy_mean": exact_acc,
                "test_accuracy_std": 0.0,
                "test_loss_mean": exact_loss,
                "test_loss_std": 0.0,
            }
        )
        print(f"{model_name:14s} exact      acc={exact_acc:.3f} loss={exact_loss:.3f}")

        for shots in args.shots:
            acc_values: list[float] = []
            loss_values: list[float] = []
            for repeat in range(args.repeats):
                p_hat = sample_test_probabilities(model, split.X_test, theta, shots, rng)
                acc_val = accuracy(split.y_test, p_hat)
                loss_val = binary_cross_entropy(split.y_test, p_hat)
                acc_values.append(acc_val)
                loss_values.append(loss_val)
                raw.append(
                    {
                        "model": model_name,
                        "encoding": encoding,
                        "ansatz": ansatz,
                        "shots": shots,
                        "repeat": repeat,
                        "test_accuracy": acc_val,
                        "test_loss": loss_val,
                    }
                )
            row = {
                "model": model_name,
                "encoding": encoding,
                "ansatz": ansatz,
                "shots": shots,
                "repeats": args.repeats,
                "test_accuracy_mean": float(mean(acc_values)),
                "test_accuracy_std": std(acc_values),
                "test_loss_mean": float(mean(loss_values)),
                "test_loss_std": std(loss_values),
            }
            rows.append(row)
            print(
                f"{model_name:14s} shots={shots:<5d} "
                f"acc={row['test_accuracy_mean']:.3f} +/- {row['test_accuracy_std']:.3f} "
                f"loss={row['test_loss_mean']:.3f} +/- {row['test_loss_std']:.3f}"
            )

    write_csv(rows, results_dir / "iris_shot_noise_study.csv")
    write_csv(raw, results_dir / "iris_shot_noise_raw.csv")
    save_json(
        {
            "experiment": "iris_shot_noise_study",
            "config": {
                "features": args.features,
                "layers": args.layers,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "shots": args.shots,
                "repeats": args.repeats,
            },
            "summary": rows,
            "raw": raw,
        },
        results_dir / "iris_shot_noise_study.json",
    )
    plot_rows(rows, figures_dir)
    print(f"\nSaved finite-shot results to {results_dir}")
    print(f"Saved finite-shot figures to {figures_dir}")


if __name__ == "__main__":
    main()
