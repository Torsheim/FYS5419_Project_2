"""Utilities for Quantum Fourier Transform work.

This file contains a few small helpers plus TODO-marked functions where the
project implementation can be added.
"""

from __future__ import annotations

import numpy as np


def qft_matrix(n_qubits: int) -> np.ndarray:
    """Return the dense QFT matrix for ``n_qubits``.

    This is useful as a reference solution when testing circuit-based or
    gate-by-gate implementations.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    dim = 2**n_qubits
    j = np.arange(dim).reshape((dim, 1))
    k = np.arange(dim).reshape((1, dim))
    omega = np.exp(2j * np.pi / dim)
    return omega ** (j * k) / np.sqrt(dim)


def inverse_qft_matrix(n_qubits: int) -> np.ndarray:
    """Return the inverse dense QFT matrix for ``n_qubits``."""
    return qft_matrix(n_qubits).conj().T


def is_unitary(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """Check whether a square matrix is unitary within a tolerance."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    eye = np.eye(matrix.shape[0], dtype=complex)
    return np.allclose(matrix.conj().T @ matrix, eye, atol=atol)


def apply_qft_statevector(state: np.ndarray) -> np.ndarray:
    """Apply dense QFT to a statevector.

    TODO: Replace or complement this with your own gate-by-gate implementation
    if you select the QFT/QPE project path.
    """
    state = np.asarray(state, dtype=complex)
    dim = state.size
    n_qubits_float = np.log2(dim)
    n_qubits = int(round(n_qubits_float))
    if 2**n_qubits != dim:
        raise ValueError("statevector length must be a power of two")
    return qft_matrix(n_qubits) @ state


def qft_gate_sequence(n_qubits: int):
    """Return a symbolic gate sequence for a QFT circuit.

    TODO: Implement using your chosen gate convention. A useful return format is
    a list of tuples, for example ``("H", qubit)`` and ``("CR", control, target, k)``.
    """
    raise NotImplementedError("Implement symbolic or executable QFT gate sequence")
