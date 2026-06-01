"""Hamiltonian helpers for optional QPE/VQE comparisons."""

from __future__ import annotations

import numpy as np


def one_qubit_hamiltonian(e1: float, e2: float, v11: float = 0.0, v12: float = 0.0, v21: float = 0.0, v22: float = 0.0) -> np.ndarray:
    """Return the 2x2 Hamiltonian H = H0 + HI from the project statement."""
    h0 = np.array([[e1, 0.0], [0.0, e2]], dtype=float)
    hi = np.array([[v11, v12], [v21, v22]], dtype=float)
    return h0 + hi


def two_qubit_hamiltonian(e00: float, e10: float, e01: float, e11: float, hx: float, hz: float) -> np.ndarray:
    """Return the 4x4 two-qubit Hamiltonian from the project statement."""
    return np.array(
        [
            [e00 + hz, 0.0, 0.0, hx],
            [0.0, e10 - hz, hx, 0.0],
            [0.0, hx, e01 - hz, 0.0],
            [hx, 0.0, 0.0, e11 + hz],
        ],
        dtype=float,
    )


def exact_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Return sorted exact eigenvalues for comparison."""
    return np.sort(np.linalg.eigvalsh(np.asarray(matrix, dtype=float)))
