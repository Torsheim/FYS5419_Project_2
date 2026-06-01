# FYS5419 Project 2 — Quantum Machine Learning implementation

This is a complete starter implementation for **Alternative 2: Quantum Machine Learning**.
It solves the required Iris-data workflow with a transparent NumPy state-vector simulator instead of relying on Qiskit internals.

The code covers:

1. Encoding classical features into quantum states.
2. Applying trainable parameterized ansaetze.
3. Measuring a qubit and using the probability of `|1>` as the model output.
4. Combining the pieces into a model that maps a design matrix `X` to predictions.
5. Training with the parameter-shift rule and Adam/gradient descent.
6. Comparing with logistic regression.
7. Testing variations of the encoder and ansatz.
8. Optionally running a small breast-cancer experiment.

## Main commands

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
pytest

python scripts/run_iris_experiment.py --features 4 --layers 2 --epochs 20 --verbose
python scripts/run_variation_study.py --features 4 --layers 2 --epochs 20
python scripts/run_breast_cancer_experiment.py --features 4 --layers 2 --epochs 20 --verbose
```

The scripts write selected results to `results/` and figures to `figures/`.

## Important files

```text
src/fys5419_project_2/quantum_state.py   Basic state-vector simulator and gates
src/fys5419_project_2/model.py           Quantum classifier and parameter-shift gradients
src/fys5419_project_2/data.py            Iris and breast-cancer data loading
src/fys5419_project_2/optimizers.py      Adam and gradient descent
src/fys5419_project_2/baselines.py       Logistic-regression comparison
scripts/run_iris_experiment.py           Main experiment
scripts/run_variation_study.py           Encoder/ansatz comparison
scripts/run_breast_cancer_experiment.py  Optional harder data set
tests/                                  Unit tests
IMPLEMENTATION_GUIDE.md                 Step-by-step guide for WSL
```

## Design choice

The project examples use Qiskit, but this implementation uses a pure NumPy simulator.
That makes the code easier to explain in the report because the gates, state-vector evolution, measurement probability, and parameter-shift gradient are all visible in the repository.
The measurement is computed as the exact probability of obtaining `1`, corresponding to the infinite-shot limit of a shot-based simulator.
