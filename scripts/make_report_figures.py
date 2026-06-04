#!/usr/bin/env python3
"""Generate report-ready figures from saved Project 2 result files.

This script does not retrain the models. It reads the CSV/JSON files created by
scripts/run_iris_experiment.py, scripts/run_variation_study.py, and optionally
scripts/run_breast_cancer_experiment.py, then writes additional PNG figures to
figures/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"saved {path}")


def plot_history(history_path: Path, figures_dir: Path) -> None:
    rows = read_csv_dicts(history_path)
    if not rows:
        return

    stem = history_path.name.replace("_history.csv", "")
    epochs = np.array([float(r["epoch"]) for r in rows])
    train_loss = np.array([float(r["train_loss"]) for r in rows])
    train_acc = np.array([float(r["train_accuracy"]) for r in rows])

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_loss, label="train")
    if "test_loss" in rows[0]:
        test_loss = np.array([float(r["test_loss"]) for r in rows])
        plt.plot(epochs, test_loss, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title(stem.replace("_", " ") + ": loss")
    plt.legend()
    save(figures_dir / f"{stem}_loss_from_csv.png")

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_acc, label="train")
    if "test_accuracy" in rows[0]:
        test_acc = np.array([float(r["test_accuracy"]) for r in rows])
        plt.plot(epochs, test_acc, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.title(stem.replace("_", " ") + ": accuracy")
    plt.legend()
    save(figures_dir / f"{stem}_accuracy.png")


def plot_predictions(predictions_path: Path, figures_dir: Path) -> None:
    rows = read_csv_dicts(predictions_path)
    if not rows:
        return

    stem = predictions_path.name.replace("_test_predictions.csv", "").replace("_train_predictions.csv", "")
    split = "test" if "_test_predictions" in predictions_path.name else "train"
    y_true = np.array([int(r["y_true"]) for r in rows])
    p1 = np.array([float(r["probability_class_1"]) for r in rows])
    y_pred = np.array([int(r["prediction"]) for r in rows])

    order = np.argsort(y_true + 0.01 * np.arange(len(y_true)))
    plt.figure(figsize=(7, 4.5))
    plt.scatter(np.arange(len(p1)), p1[order], label="model probability")
    plt.scatter(np.arange(len(p1)), y_true[order], marker="x", label="true class")
    plt.axhline(0.5, linestyle="--", linewidth=1, label="decision boundary")
    plt.xlabel(f"{split.capitalize()} sample index, sorted by true class")
    plt.ylabel("Probability of class 1")
    plt.ylim(-0.05, 1.05)
    plt.title(stem.replace("_", " ") + f": {split} predictions")
    plt.legend()
    save(figures_dir / f"{stem}_{split}_predictions.png")

    if split == "test":
        cm = np.zeros((2, 2), dtype=int)
        for yt, yp in zip(y_true, y_pred):
            cm[yt, yp] += 1
        plt.figure(figsize=(4.8, 4.2))
        plt.imshow(cm)
        plt.xticks([0, 1], ["pred 0", "pred 1"])
        plt.yticks([0, 1], ["true 0", "true 1"])
        plt.title(stem.replace("_", " ") + ": confusion matrix")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.colorbar(label="Count")
        save(figures_dir / f"{stem}_confusion_matrix.png")


def plot_metrics(metrics_path: Path, figures_dir: Path) -> None:
    data = read_json(metrics_path)
    stem = metrics_path.name.replace("_metrics.json", "")
    if "quantum_model" not in data or "logistic_regression" not in data:
        return

    q = data["quantum_model"]
    l = data["logistic_regression"]
    labels = ["Quantum train", "Quantum test", "Logistic train", "Logistic test"]
    acc = [q.get("train_accuracy", np.nan), q.get("test_accuracy", np.nan), l.get("train_accuracy", np.nan), l.get("test_accuracy", np.nan)]
    loss = [q.get("train_loss", np.nan), q.get("test_loss", np.nan), l.get("train_loss", np.nan), l.get("test_loss", np.nan)]

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(labels, acc)
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.title(stem.replace("_", " ") + ": accuracy comparison")
    save(figures_dir / f"{stem}_model_comparison_accuracy.png")

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(labels, loss)
    plt.ylabel("Binary cross-entropy")
    plt.xticks(rotation=20, ha="right")
    plt.title(stem.replace("_", " ") + ": loss comparison")
    save(figures_dir / f"{stem}_model_comparison_loss.png")


def plot_variation_study(path: Path, figures_dir: Path) -> None:
    if not path.exists():
        return
    data = read_json(path)
    rows = data.get("quantum_models", [])
    if not rows:
        return
    labels = [f"{r['encoding']}\n{r['ansatz']}" for r in rows]
    test_acc = [float(r["test_accuracy"]) for r in rows]
    test_loss = [float(r["test_loss"]) for r in rows]

    if "logistic_regression" in data:
        labels.append("logistic\nregression")
        test_acc.append(float(data["logistic_regression"]["test_accuracy"]))
        test_loss.append(float(data["logistic_regression"]["test_loss"]))

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, test_acc)
    plt.ylabel("Test accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("Iris variation study: encoder and ansatz comparison")
    save(figures_dir / "iris_variation_study_accuracy.png")

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, test_loss)
    plt.ylabel("Test binary cross-entropy")
    plt.title("Iris variation study: test loss comparison")
    save(figures_dir / "iris_variation_study_loss.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="report/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    for path in sorted(results_dir.glob("*_history.csv")):
        plot_history(path, figures_dir)

    for path in sorted(results_dir.glob("*_predictions.csv")):
        plot_predictions(path, figures_dir)

    for path in sorted(results_dir.glob("*_metrics.json")):
        plot_metrics(path, figures_dir)

    plot_variation_study(results_dir / "iris_variation_study.json", figures_dir)


if __name__ == "__main__":
    main()
