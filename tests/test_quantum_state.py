import numpy as np

from fys5419_project_2.quantum_state import (
    apply_cnot,
    apply_single_qubit_gate,
    hadamard,
    probability_one,
    ry,
    state_norm,
    zero_state,
)


def test_hadamard_preserves_norm():
    state = zero_state(1)
    state = apply_single_qubit_gate(state, hadamard(), qubit=0, n_qubits=1)
    assert np.isclose(state_norm(state), 1.0)
    assert np.isclose(probability_one(state, qubit=0, n_qubits=1), 0.5)


def test_ry_pi_maps_zero_to_one():
    state = zero_state(1)
    state = apply_single_qubit_gate(state, ry(np.pi), qubit=0, n_qubits=1)
    assert np.isclose(probability_one(state, qubit=0, n_qubits=1), 1.0)


def test_cnot_flips_target_when_control_is_one():
    # two-qubit little-endian state |01>: qubit 0 is 1, qubit 1 is 0.
    state = np.zeros(4, dtype=np.complex128)
    state[1] = 1.0
    out = apply_cnot(state, control=0, target=1, n_qubits=2)
    expected = np.zeros(4, dtype=np.complex128)
    expected[3] = 1.0
    assert np.allclose(out, expected)
