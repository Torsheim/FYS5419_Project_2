"""Optimizers for the quantum classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import QuantumCircuitClassifier

Array = np.ndarray


@dataclass
class TrainingResult:
    """Result returned by the training functions."""

    theta: Array
    history: list[dict[str, float]]


def train_adam(
    model: QuantumCircuitClassifier,
    X_train: Array,
    y_train: Array,
    theta0: Array,
    X_val: Array | None = None,
    y_val: Array | None = None,
    epochs: int = 100,
    learning_rate: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    tolerance: float | None = None,
    verbose: bool = False,
) -> TrainingResult:
    """Train with full-batch Adam and parameter-shift gradients."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    theta = np.asarray(theta0, dtype=float).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        grad = model.gradient(X_train, y_train, theta)
        grad_norm = float(np.linalg.norm(grad))
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**epoch)
        v_hat = v / (1.0 - beta2**epoch)
        theta -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        row = {
            "epoch": float(epoch),
            "train_loss": model.loss(X_train, y_train, theta),
            "train_accuracy": model.accuracy(X_train, y_train, theta),
            "grad_norm": grad_norm,
        }
        if X_val is not None and y_val is not None:
            row["test_loss"] = model.loss(X_val, y_val, theta)
            row["test_accuracy"] = model.accuracy(X_val, y_val, theta)
        history.append(row)

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            message = (
                f"epoch={epoch:4d} train_loss={row['train_loss']:.6f} "
                f"train_acc={row['train_accuracy']:.3f} grad_norm={grad_norm:.3e}"
            )
            if "test_accuracy" in row:
                message += f" test_acc={row['test_accuracy']:.3f}"
            print(message)

        if tolerance is not None and grad_norm < tolerance:
            break

    return TrainingResult(theta=theta, history=history)


def train_gradient_descent(
    model: QuantumCircuitClassifier,
    X_train: Array,
    y_train: Array,
    theta0: Array,
    X_val: Array | None = None,
    y_val: Array | None = None,
    epochs: int = 100,
    learning_rate: float = 0.05,
    verbose: bool = False,
) -> TrainingResult:
    """Plain full-batch gradient descent."""
    theta = np.asarray(theta0, dtype=float).copy()
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        grad = model.gradient(X_train, y_train, theta)
        theta -= learning_rate * grad
        row = {
            "epoch": float(epoch),
            "train_loss": model.loss(X_train, y_train, theta),
            "train_accuracy": model.accuracy(X_train, y_train, theta),
            "grad_norm": float(np.linalg.norm(grad)),
        }
        if X_val is not None and y_val is not None:
            row["test_loss"] = model.loss(X_val, y_val, theta)
            row["test_accuracy"] = model.accuracy(X_val, y_val, theta)
        history.append(row)
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(
                f"epoch={epoch:4d} train_loss={row['train_loss']:.6f} "
                f"train_acc={row['train_accuracy']:.3f}"
            )
    return TrainingResult(theta=theta, history=history)
