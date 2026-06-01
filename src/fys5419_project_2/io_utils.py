"""Small input/output helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

Array = np.ndarray


def ensure_parent(path: str | Path) -> Path:
    """Create the parent folder for a path and return the path as ``Path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    """Save JSON with indentation."""
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_history_csv(history: Iterable[dict[str, float]], path: str | Path) -> None:
    """Save training history to CSV."""
    rows = list(history)
    if not rows:
        raise ValueError("history is empty")
    path = ensure_parent(path)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions_csv(y_true: Array, probability: Array, path: str | Path) -> None:
    """Save targets, probabilities and hard predictions to CSV."""
    path = ensure_parent(path)
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "probability_class_1", "prediction"])
        for yi, pi in zip(y_true, probability):
            writer.writerow([int(yi), float(pi), int(pi >= 0.5)])
