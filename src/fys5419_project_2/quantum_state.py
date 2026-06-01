"""Small NumPy state-vector simulator used for Project 2.

The goal is transparency rather than speed.  The code uses a little-endian
basis convention: qubit 0 is the least significant bit of the computational
basis index.  For example, for two qubits the state-vector indices are

    index 0 -> |00>, index 1 -> |01>, index 2 -> |10>, index 3 -> |11>

where the rightmost ket bit corresponds to qubit 0.  This convention is common
in low-level simulators and is documented here so that figures/tables in the
report can be interpreted correctly.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def zero_state(n_qubits: int) -> Array:
    """Return |00...0> for ``n_qubits``.

    Parameters
    ----------
    n_qubits:
        Number of qubits in the simulated register.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0
    return state


def hadamard() -> Array:
    """Hadamard gate."""
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)


def rx(theta: float) -> Array:
    """Single-qubit rotation around the x-axis."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry(theta: float) -> Array:
    """Single-qubit rotation around the y-axis."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> Array:
    """Single-qubit rotation around the z-axis."""
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )


def apply_single_qubit_gate(state: Array, gate: Array, qubit: int, n_qubits: int) -> Array:
    """Apply a 2x2 gate to one qubit of a state vector.

    The implementation uses NumPy tensor operations instead of Python loops.
    This matters because parameter-shift training evaluates the circuit many
    times. The little-endian convention is preserved: qubit 0 is the least
    significant bit, corresponding to the last tensor axis.
    """
    _validate_state(state, n_qubits)
    _validate_qubit(qubit, n_qubits)
    gate = np.asarray(gate, dtype=np.complex128)
    if gate.shape != (2, 2):
        raise ValueError("gate must have shape (2, 2)")

    tensor = np.asarray(state, dtype=np.complex128).reshape((2,) * n_qubits)
    axis = n_qubits - 1 - qubit
    moved = np.moveaxis(tensor, axis, 0)
    updated = np.tensordot(gate, moved, axes=([1], [0]))
    out = np.moveaxis(updated, 0, axis).reshape(-1)
    return out.astype(np.complex128, copy=False)


def apply_cnot(state: Array, control: int, target: int, n_qubits: int) -> Array:
    """Apply a controlled-X gate."""
    _validate_state(state, n_qubits)
    _validate_qubit(control, n_qubits)
    _validate_qubit(target, n_qubits)
    if control == target:
        raise ValueError("control and target must be different qubits")

    indices = np.arange(2**n_qubits)
    control_mask = 1 << control
    target_mask = 1 << target
    output_indices = np.where((indices & control_mask) != 0, indices ^ target_mask, indices)
    out = np.empty_like(state, dtype=np.complex128)
    out[output_indices] = state
    return out


def probability_one(state: Array, qubit: int, n_qubits: int) -> float:
    """Return the exact probability of measuring ``qubit`` in state |1>."""
    _validate_state(state, n_qubits)
    _validate_qubit(qubit, n_qubits)
    indices = np.arange(2**n_qubits)
    mask = (indices & (1 << qubit)) != 0
    probability = float(np.sum(np.abs(state[mask]) ** 2))
    return float(np.clip(probability, 0.0, 1.0))


def sample_measurements(
    state: Array,
    qubit: int,
    n_qubits: int,
    shots: int = 1000,
    seed: int | None = None,
) -> float:
    """Estimate P(qubit=1) by Bernoulli sampling.

    This is useful when you want to mimic shot-based measurement.  The training
    code uses exact probabilities because they are deterministic and make the
    parameter-shift gradients less noisy.
    """
    if shots < 1:
        raise ValueError("shots must be at least 1")
    p_one = probability_one(state, qubit=qubit, n_qubits=n_qubits)
    rng = np.random.default_rng(seed)
    return float(rng.binomial(shots, p_one) / shots)


def state_norm(state: Array) -> float:
    """Return the Euclidean norm of a state vector."""
    return float(np.linalg.norm(state))


def _validate_state(state: Array, n_qubits: int) -> None:
    state = np.asarray(state)
    expected_shape = (2**n_qubits,)
    if state.shape != expected_shape:
        raise ValueError(f"state must have shape {expected_shape}, got {state.shape}")


def _validate_qubit(qubit: int, n_qubits: int) -> None:
    if not 0 <= qubit < n_qubits:
        raise ValueError(f"qubit must be in [0, {n_qubits - 1}], got {qubit}")
