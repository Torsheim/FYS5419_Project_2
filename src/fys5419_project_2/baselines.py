"""Classical baselines for comparison."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

Array = np.ndarray


def logistic_regression_baseline(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array,
    random_state: int = 42,
) -> dict[str, float]:
    """Train logistic regression and return comparable metrics."""
    classifier = LogisticRegression(random_state=random_state, max_iter=1000)
    classifier.fit(X_train, y_train)
    train_probability = classifier.predict_proba(X_train)[:, 1]
    test_probability = classifier.predict_proba(X_test)[:, 1]
    train_prediction = classifier.predict(X_train)
    test_prediction = classifier.predict(X_test)
    return {
        "train_loss": float(log_loss(y_train, train_probability, labels=[0, 1])),
        "test_loss": float(log_loss(y_test, test_probability, labels=[0, 1])),
        "train_accuracy": float(accuracy_score(y_train, train_prediction)),
        "test_accuracy": float(accuracy_score(y_test, test_prediction)),
        "n_train": float(len(y_train)),
        "n_test": float(len(y_test)),
    }
