import numpy as np

from fys5419_project_2.model import QuantumCircuitClassifier
from fys5419_project_2.optimizers import train_adam


def test_training_smoke_test_runs_two_epochs():
    X = np.array([[0.0, 0.0], [0.1, 0.2], [0.8, 0.9], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    model = QuantumCircuitClassifier(n_qubits=2, n_layers=1, encoding="ry", ansatz="simple")
    theta0 = model.initial_parameters(seed=3)
    result = train_adam(model, X, y, theta0, epochs=2, learning_rate=0.05)
    assert result.theta.shape == theta0.shape
    assert len(result.history) == 2
    assert np.isfinite(result.history[-1]["train_loss"])
