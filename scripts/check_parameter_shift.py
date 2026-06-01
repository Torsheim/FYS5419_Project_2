#!/usr/bin/env python3
"""Compare parameter-shift gradients with finite differences."""

from __future__ import annotations

import numpy as np

from fys5419_project_2.model import QuantumCircuitClassifier


def main() -> None:
    rng = np.random.default_rng(123)
    X = rng.random((5, 2))
    y = np.array([0, 1, 0, 1, 1])
    model = QuantumCircuitClassifier(n_qubits=2, n_layers=1, encoding="h_rz", ansatz="simple")
    theta = model.initial_parameters(seed=123, distribution="small")
    grad_shift = model.gradient(X, y, theta)
    grad_fd = model.finite_difference_gradient(X, y, theta)
    max_abs_error = np.max(np.abs(grad_shift - grad_fd))
    print("parameter-shift gradient:", grad_shift)
    print("finite-difference gradient:", grad_fd)
    print("max absolute error:", max_abs_error)


if __name__ == "__main__":
    main()
