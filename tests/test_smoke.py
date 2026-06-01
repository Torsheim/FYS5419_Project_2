import numpy as np

from fys5419_project_2.hamiltonians import exact_eigenvalues, one_qubit_hamiltonian
from fys5419_project_2.qft import is_unitary, qft_matrix
from fys5419_project_2.qpe import phase_from_bitstring


def test_qft_matrix_unitary():
    assert is_unitary(qft_matrix(3))


def test_phase_from_bitstring():
    assert phase_from_bitstring("01") == 0.25
    assert phase_from_bitstring("10") == 0.5


def test_one_qubit_hamiltonian_eigenvalues():
    h = one_qubit_hamiltonian(1.0, 2.0)
    assert np.allclose(exact_eigenvalues(h), [1.0, 2.0])
