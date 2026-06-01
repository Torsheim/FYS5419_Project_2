import numpy as np

from fys5419_project_2.model import QuantumCircuitClassifier


def test_prediction_shape_and_range():
    model = QuantumCircuitClassifier(n_qubits=2, n_layers=1, encoding="h_rz", ansatz="simple")
    theta = model.initial_parameters(seed=1)
    X = np.array([[0.1, 0.2], [0.8, 0.4], [0.5, 0.5]])
    proba = model.predict_proba(X, theta)
    assert proba.shape == (3,)
    assert np.all((0.0 <= proba) & (proba <= 1.0))


def test_parameter_count_simple_matches_project_example():
    model = QuantumCircuitClassifier(n_qubits=2, n_layers=1, ansatz="simple")
    assert model.n_parameters == 4


def test_parameter_shift_gradient_matches_finite_difference():
    rng = np.random.default_rng(2024)
    X = rng.random((4, 2))
    y = np.array([0, 1, 0, 1])
    model = QuantumCircuitClassifier(n_qubits=2, n_layers=1, encoding="h_rz", ansatz="simple")
    theta = model.initial_parameters(seed=2)
    grad_shift = model.gradient(X, y, theta)
    grad_fd = model.finite_difference_gradient(X, y, theta, h=1e-6)
    assert np.allclose(grad_shift, grad_fd, atol=1e-5, rtol=1e-4)
