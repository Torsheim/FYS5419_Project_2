"""Placeholder script for the QML path."""

from __future__ import annotations

from fys5419_project_2.qml import load_binary_iris


def main() -> None:
    x, y = load_binary_iris()
    print(f"Binary Iris data shape: X={x.shape}, y={y.shape}")
    print("Add circuit training code here if you choose the QML path.")


if __name__ == "__main__":
    main()
