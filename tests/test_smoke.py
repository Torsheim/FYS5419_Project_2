import numpy as np

from fys5419_project_2.data import load_binary_iris_data
from fys5419_project_2.model import QuantumCircuitClassifier
from fys5419_project_2.optimizers import train_adam


def test_qml_smoke_pipeline_runs():
    split = load_binary_iris_data(
        n_features=2,
        test_size=0.25,
        random_state=42,
    )

    model = QuantumCircuitClassifier(
        n_qubits=2,
        n_layers=1,
        encoding="ry",
        ansatz="simple",
    )

    theta0 = model.initial_parameters(seed=7)

    probabilities = model.predict_proba(split.X_test[:5], theta0)
    assert probabilities.shape == (5,)
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))

    result = train_adam(
        model,
        split.X_train[:8],
        split.y_train[:8],
        theta0,
        X_val=split.X_test[:4],
        y_val=split.y_test[:4],
        epochs=2,
        learning_rate=0.05,
    )

    assert result.theta.shape == theta0.shape
    assert len(result.history) == 2
    assert np.isfinite(result.history[-1]["train_loss"])
