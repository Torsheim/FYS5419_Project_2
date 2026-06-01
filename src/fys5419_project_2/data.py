"""Dataset loading and preprocessing for the QML experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

Array = np.ndarray


@dataclass
class DatasetSplit:
    """Container for train/test data and preprocessing metadata."""

    X_train: Array
    X_test: Array
    y_train: Array
    y_test: Array
    feature_names: list[str]
    target_names: list[str]
    scaler: MinMaxScaler


def load_binary_iris_data(
    n_features: int = 4,
    test_size: float = 0.25,
    random_state: int = 42,
) -> DatasetSplit:
    """Load the first two classes of the Iris data set.

    The project asks for the first two Iris targets.  Features are scaled to
    [0, 1] because the quantum encoders map feature values to rotation angles.
    """
    if not 1 <= n_features <= 4:
        raise ValueError("n_features for Iris must be between 1 and 4")

    iris = load_iris()
    X = iris.data
    y = iris.target
    keep = y < 2
    X = X[keep, :n_features]
    y = y[keep].astype(int)
    feature_names = list(iris.feature_names[:n_features])
    target_names = list(iris.target_names[:2])
    return _split_and_scale(X, y, feature_names, target_names, test_size, random_state)


def load_breast_cancer_data(
    n_features: int = 4,
    test_size: float = 0.25,
    random_state: int = 42,
) -> DatasetSplit:
    """Load a low-dimensional breast-cancer subset for the optional task."""
    data = load_breast_cancer()
    if not 1 <= n_features <= data.data.shape[1]:
        raise ValueError(f"n_features must be between 1 and {data.data.shape[1]}")
    X = data.data[:, :n_features]
    y = data.target.astype(int)
    feature_names = list(data.feature_names[:n_features])
    target_names = list(data.target_names)
    return _split_and_scale(X, y, feature_names, target_names, test_size, random_state)


def _split_and_scale(
    X: Array,
    y: Array,
    feature_names: list[str],
    target_names: list[str],
    test_size: float,
    random_state: int,
) -> DatasetSplit:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Keep possible out-of-range test values from producing unexpectedly large
    # angles when the test set contains values outside the train min/max range.
    X_train = np.clip(X_train, 0.0, 1.0)
    X_test = np.clip(X_test, 0.0, 1.0)
    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_names=target_names,
        scaler=scaler,
    )
