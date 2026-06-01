"""Quantum Phase Estimation placeholders."""

from __future__ import annotations

import numpy as np


def phase_from_bitstring(bitstring: str) -> float:
    """Convert a binary phase-estimation bitstring to a number in [0, 1)."""
    if not bitstring or any(bit not in "01" for bit in bitstring):
        raise ValueError("bitstring must contain only 0 and 1")
    return int(bitstring, 2) / (2 ** len(bitstring))


def estimate_phase(*args, **kwargs):
    """Implement QPE for the selected unitary/operator.

    TODO: Add your QPE implementation here. Decide whether the function should
    work with dense matrices, with your own circuit representation, or with
    Qiskit circuits for comparison.
    """
    raise NotImplementedError("Add QPE implementation")


def eigenvalue_from_phase(phase: float, time: float = 1.0) -> float:
    """Convert phase to an energy-like eigenvalue for U = exp(-i H t).

    The sign and scaling depend on your definition of the time-evolution
    operator, so verify this formula in the report before using it for results.
    """
    return 2 * np.pi * phase / time
