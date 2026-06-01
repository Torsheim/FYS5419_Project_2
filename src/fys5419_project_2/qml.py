"""Quantum machine learning placeholders for the Iris project path."""

from __future__ import annotations

import numpy as np


def load_binary_iris():
    """Load the first two Iris classes as described in the project brief."""
    from sklearn import datasets

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    idx = np.where(y < 2)
    return x[idx], y[idx]


def parameter_shift_gradient(model, x: np.ndarray, theta: np.ndarray, shift: float = np.pi / 2) -> np.ndarray:
    """Compute parameter-shift derivatives of a scalar model output.

    ``model`` should have signature ``model(x, theta) -> float``.
    """
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta)
    for j in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += shift
        minus[j] -= shift
        grad[j] = 0.5 * (model(x, plus) - model(x, minus))
    return grad


def encode_features(*args, **kwargs):
    """TODO: Build your Qiskit feature map or custom circuit encoder."""
    raise NotImplementedError("Add feature encoding circuit")


def ansatz(*args, **kwargs):
    """TODO: Build your parameterized circuit ansatz."""
    raise NotImplementedError("Add ansatz circuit")


def predict(*args, **kwargs):
    """TODO: Return model predictions for a design matrix."""
    raise NotImplementedError("Add prediction function")
