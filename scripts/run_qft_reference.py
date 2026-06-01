"""Small reference run for dense QFT utilities.

Usage:
    python scripts/run_qft_reference.py
"""

from __future__ import annotations

import numpy as np

from fys5419_project_2.qft import apply_qft_statevector, is_unitary, qft_matrix


def main() -> None:
    n_qubits = 3
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    transformed = apply_qft_statevector(state)
    print(f"QFT matrix is unitary: {is_unitary(qft_matrix(n_qubits))}")
    print("QFT |000> amplitudes:")
    print(transformed)


if __name__ == "__main__":
    main()
