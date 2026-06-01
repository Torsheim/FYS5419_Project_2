"""Quantum machine-learning model for FYS5419 Project 2.

This module implements Alternative 2 of Project 2 with a small NumPy
state-vector simulator.  It contains

* feature encoders,
* parameterized ansaetze,
* exact measurement probabilities,
* binary cross-entropy,
* parameter-shift gradients.

The implementation deliberately avoids Qiskit/PennyLane dependencies so that
all details are visible in the submitted source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .quantum_state import (
    apply_cnot,
    apply_single_qubit_gate,
    hadamard,
    probability_one,
    rx,
    ry,
    rz,
    zero_state,
)

Array = np.ndarray
EncodingName = Literal["h_rz", "ry", "ry_rz"]
AnsatzName = Literal["simple", "strong"]


@dataclass(frozen=True)
class QuantumCircuitClassifier:
    """Parameterized quantum circuit used as a binary classifier.

    Parameters
    ----------
    n_qubits:
        Number of data qubits.  In the experiments this is equal to the number
        of selected features.
    n_layers:
        Number of repeated ansatz blocks.
    encoding:
        ``"h_rz"`` implements the encoding shown in the project text:
        H followed by Rz(2*pi*x_j). ``"ry"`` and ``"ry_rz"`` are variations
        used in Project 2f.
    ansatz:
        ``"simple"`` follows the project example: two Ry layers with CNOT
        entanglement. ``"strong"`` adds more rotations and a ring entangler.
    measured_qubit:
        Which qubit to measure.  ``-1`` means the last qubit.
    """

    n_qubits: int
    n_layers: int = 1
    encoding: EncodingName = "h_rz"
    ansatz: AnsatzName = "simple"
    measured_qubit: int = -1

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if self.encoding not in {"h_rz", "ry", "ry_rz"}:
            raise ValueError(f"unknown encoding: {self.encoding}")
        if self.ansatz not in {"simple", "strong"}:
            raise ValueError(f"unknown ansatz: {self.ansatz}")
        measured = self.resolved_measured_qubit
        if measured < 0 or measured >= self.n_qubits:
            raise ValueError("measured_qubit is outside the qubit register")

    @property
    def resolved_measured_qubit(self) -> int:
        """Return the non-negative measured qubit index."""
        return self.n_qubits - 1 if self.measured_qubit == -1 else self.measured_qubit

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters used by the selected ansatz."""
        if self.ansatz == "simple":
            return 2 * self.n_layers * self.n_qubits
        if self.ansatz == "strong":
            return 3 * self.n_layers * self.n_qubits
        raise RuntimeError("unreachable")

    def initial_parameters(
        self,
        seed: int = 42,
        distribution: Literal["small", "uniform"] = "small",
    ) -> Array:
        """Create reproducible initial parameters.

        ``distribution="small"`` starts close to zero, which is often stable for
        this small example.  ``distribution="uniform"`` samples angles in
        [0, 2*pi), matching the project-code style.
        """
        rng = np.random.default_rng(seed)
        if distribution == "small":
            return 0.1 * rng.standard_normal(self.n_parameters)
        if distribution == "uniform":
            return 2.0 * np.pi * rng.random(self.n_parameters)
        raise ValueError("distribution must be 'small' or 'uniform'")

    def statevector(self, x: Array, theta: Array) -> Array:
        """Return the final state vector for one input sample."""
        x = self._validate_sample(x)
        theta = self._validate_theta(theta)
        state = zero_state(self.n_qubits)
        state = self._encode(state, x)
        state = self._apply_ansatz(state, theta)
        return state

    def predict_one(self, x: Array, theta: Array) -> float:
        """Return P(y=1|x, theta) for one sample."""
        state = self.statevector(x, theta)
        return probability_one(state, self.resolved_measured_qubit, self.n_qubits)

    def predict_proba(self, X: Array, theta: Array) -> Array:
        """Return probabilities P(y=1|x_i, theta) for all samples."""
        X = self._validate_design_matrix(X)
        theta = self._validate_theta(theta)
        return np.array([self.predict_one(x, theta) for x in X], dtype=float)

    def predict(self, X: Array, theta: Array, threshold: float = 0.5) -> Array:
        """Return class predictions using a probability threshold."""
        return (self.predict_proba(X, theta) >= threshold).astype(int)

    def loss(self, X: Array, y: Array, theta: Array, eps: float = 1e-10) -> float:
        """Mean binary cross-entropy.

        The project text gives the derivative corresponding to binary
        cross-entropy.  We therefore use the standard binary classification loss

            - mean(y log(f) + (1-y) log(1-f)).
        """
        y = self._validate_targets(y)
        f = np.clip(self.predict_proba(X, theta), eps, 1.0 - eps)
        return float(-np.mean(y * np.log(f) + (1.0 - y) * np.log(1.0 - f)))

    def accuracy(self, X: Array, y: Array, theta: Array, threshold: float = 0.5) -> float:
        """Classification accuracy."""
        y = self._validate_targets(y).astype(int)
        return float(np.mean(self.predict(X, theta, threshold=threshold) == y))

    def parameter_shift_jacobian(self, X: Array, theta: Array) -> Array:
        """Jacobian df_i/dtheta_j using the parameter-shift rule.

        Returns an array with shape ``(n_samples, n_parameters)``.
        """
        X = self._validate_design_matrix(X)
        theta = self._validate_theta(theta)
        jacobian = np.zeros((X.shape[0], theta.size), dtype=float)
        shift = np.pi / 2.0
        for parameter_index in range(theta.size):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[parameter_index] += shift
            theta_minus[parameter_index] -= shift
            f_plus = self.predict_proba(X, theta_plus)
            f_minus = self.predict_proba(X, theta_minus)
            jacobian[:, parameter_index] = 0.5 * (f_plus - f_minus)
        return jacobian

    def gradient(self, X: Array, y: Array, theta: Array, eps: float = 1e-10) -> Array:
        """Gradient of mean binary cross-entropy with respect to theta."""
        X = self._validate_design_matrix(X)
        y = self._validate_targets(y)
        f = np.clip(self.predict_proba(X, theta), eps, 1.0 - eps)
        jac = self.parameter_shift_jacobian(X, theta)
        coefficient = (f - y) / (f * (1.0 - f))
        return (coefficient @ jac) / X.shape[0]

    def finite_difference_gradient(
        self,
        X: Array,
        y: Array,
        theta: Array,
        h: float = 1e-6,
    ) -> Array:
        """Numerical gradient used only for tests and debugging."""
        theta = self._validate_theta(theta)
        grad = np.zeros_like(theta)
        for parameter_index in range(theta.size):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[parameter_index] += h
            theta_minus[parameter_index] -= h
            grad[parameter_index] = (self.loss(X, y, theta_plus) - self.loss(X, y, theta_minus)) / (2.0 * h)
        return grad

    def _encode(self, state: Array, x: Array) -> Array:
        for qubit, feature in enumerate(x):
            angle = 2.0 * np.pi * float(feature)
            if self.encoding == "h_rz":
                state = apply_single_qubit_gate(state, hadamard(), qubit, self.n_qubits)
                state = apply_single_qubit_gate(state, rz(angle), qubit, self.n_qubits)
            elif self.encoding == "ry":
                state = apply_single_qubit_gate(state, ry(angle), qubit, self.n_qubits)
            elif self.encoding == "ry_rz":
                state = apply_single_qubit_gate(state, hadamard(), qubit, self.n_qubits)
                state = apply_single_qubit_gate(state, ry(np.pi * float(feature)), qubit, self.n_qubits)
                state = apply_single_qubit_gate(state, rz(angle), qubit, self.n_qubits)
            else:
                raise RuntimeError("unreachable")
        return state

    def _apply_ansatz(self, state: Array, theta: Array) -> Array:
        if self.ansatz == "simple":
            return self._apply_simple_ansatz(state, theta)
        if self.ansatz == "strong":
            return self._apply_strong_ansatz(state, theta)
        raise RuntimeError("unreachable")

    def _apply_simple_ansatz(self, state: Array, theta: Array) -> Array:
        parameter_index = 0
        for _ in range(self.n_layers):
            for qubit in range(self.n_qubits):
                state = apply_single_qubit_gate(
                    state, ry(theta[parameter_index]), qubit, self.n_qubits
                )
                parameter_index += 1
            state = self._entangle_chain(state)
            for qubit in range(self.n_qubits):
                state = apply_single_qubit_gate(
                    state, ry(theta[parameter_index]), qubit, self.n_qubits
                )
                parameter_index += 1
            state = self._entangle_chain(state)
        return state

    def _apply_strong_ansatz(self, state: Array, theta: Array) -> Array:
        parameter_index = 0
        for _ in range(self.n_layers):
            for qubit in range(self.n_qubits):
                state = apply_single_qubit_gate(
                    state, ry(theta[parameter_index]), qubit, self.n_qubits
                )
                parameter_index += 1
                state = apply_single_qubit_gate(
                    state, rz(theta[parameter_index]), qubit, self.n_qubits
                )
                parameter_index += 1
            state = self._entangle_ring(state)
            for qubit in range(self.n_qubits):
                state = apply_single_qubit_gate(
                    state, rx(theta[parameter_index]), qubit, self.n_qubits
                )
                parameter_index += 1
        return state

    def _entangle_chain(self, state: Array) -> Array:
        for control in range(self.n_qubits - 1):
            state = apply_cnot(state, control=control, target=control + 1, n_qubits=self.n_qubits)
        return state

    def _entangle_ring(self, state: Array) -> Array:
        state = self._entangle_chain(state)
        if self.n_qubits > 2:
            state = apply_cnot(state, control=self.n_qubits - 1, target=0, n_qubits=self.n_qubits)
        return state

    def _validate_sample(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_qubits,):
            raise ValueError(f"sample must have shape ({self.n_qubits},), got {x.shape}")
        return x

    def _validate_design_matrix(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_qubits:
            raise ValueError(
                f"X must have shape (n_samples, {self.n_qubits}), got {X.shape}"
            )
        return X

    def _validate_targets(self, y: Array) -> Array:
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got {y.shape}")
        unique = np.unique(y)
        if not np.all(np.isin(unique, [0.0, 1.0])):
            raise ValueError("targets must be binary labels 0/1")
        return y

    def _validate_theta(self, theta: Array) -> Array:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_parameters,):
            raise ValueError(
                f"theta must have shape ({self.n_parameters},), got {theta.shape}"
            )
        return theta
