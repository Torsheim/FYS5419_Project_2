# FYS5419 Project 2 implementation guide: Quantum Machine Learning

This code implements **Alternative 2: Quantum Machine Learning**.

It solves the main Iris task with a small NumPy statevector simulator.  The implementation does not depend on Qiskit, PennyLane, or Qiskit Aer.  The code still implements the same quantum-computing ingredients explicitly: feature encoding, parameterized gates, CNOT entanglement, measurement probability, parameter-shift gradients, and training.

The guide assumes your WSL repository is:

```bash
~/projects/FYS5419_Project_2
```

You said you do **not** want to commit before the whole project is finished.  None of the commands below commit or push anything.

---

## 1. Copy the solution zip into your repository

After downloading `FYS5419_Project_2_qml_solution.zip`, it will usually be in your Windows Downloads folder.

Find your Windows username:

```bash
ls /mnt/c/Users
```

Then copy the zip into your repository.  Replace `<YOUR_WINDOWS_USERNAME>` with the username you found.

```bash
cd ~/projects/FYS5419_Project_2
cp /mnt/c/Users/<YOUR_WINDOWS_USERNAME>/Downloads/FYS5419_Project_2_qml_solution.zip .
```

Example:

```bash
cd ~/projects/FYS5419_Project_2
cp /mnt/c/Users/torsheim/Downloads/FYS5419_Project_2_qml_solution.zip .
```

---

## 2. Unpack the files

From the repository root:

```bash
cd ~/projects/FYS5419_Project_2
unzip -o FYS5419_Project_2_qml_solution.zip
rm FYS5419_Project_2_qml_solution.zip
```

Check that the files are present:

```bash
find src/fys5419_project_2 -maxdepth 1 -type f | sort
find scripts tests -maxdepth 2 -type f | sort
```

You should see files such as:

```text
src/fys5419_project_2/quantum_state.py
src/fys5419_project_2/model.py
src/fys5419_project_2/optimizers.py
src/fys5419_project_2/data.py
scripts/run_iris_experiment.py
scripts/run_variation_study.py
tests/test_model.py
```

---

## 3. Activate the virtual environment

```bash
cd ~/projects/FYS5419_Project_2
source .venv/bin/activate
```

Your prompt should begin with `(.venv)`.

---

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The `-e .` command installs the repository in editable mode, so changes under `src/` are used immediately.

---

## 5. Run the tests

```bash
pytest
```

or:

```bash
make test
```

Expected result: all tests pass.  One of the tests checks that the parameter-shift gradient agrees with a finite-difference gradient.

You can also run:

```bash
python scripts/check_parameter_shift.py
```

---

## 6. Run the main Iris experiment

Quick run:

```bash
python scripts/run_iris_experiment.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05 --verbose
```

or:

```bash
make iris
```

This creates files like:

```text
results/iris_h_rz_simple_4features_2layers_metrics.json
results/iris_h_rz_simple_4features_2layers_history.csv
results/iris_h_rz_simple_4features_2layers_train_predictions.csv
results/iris_h_rz_simple_4features_2layers_test_predictions.csv
results/iris_h_rz_simple_4features_2layers_theta.npy
figures/iris_h_rz_simple_4features_2layers_loss.png
```

Inspect the metrics:

```bash
cat results/iris_h_rz_simple_4features_2layers_metrics.json
```

For a longer final run, use more epochs:

```bash
python scripts/run_iris_experiment.py --features 4 --layers 2 --epochs 60 --learning-rate 0.05 --verbose
```

The longer run can take several minutes because the parameter-shift rule evaluates the circuit twice for every parameter at every epoch.

---

## 7. Run the encoder/ansatz variation study

This addresses the project part where you change or add gates to the encoder and parameterized ansatz.

```bash
python scripts/run_variation_study.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05
```

or:

```bash
make variation
```

It writes:

```text
results/iris_variation_study.json
```

View it with:

```bash
cat results/iris_variation_study.json
```

---

## 8. Optional Breast Cancer experiment

The project text mentions Breast Cancer as an optional harder problem.  Run it only if you have time to discuss it in the report.

```bash
python scripts/run_breast_cancer_experiment.py --features 4 --layers 2 --epochs 20 --learning-rate 0.03 --verbose
```

or:

```bash
make breast
```

---

## 9. What each file does

```text
src/fys5419_project_2/quantum_state.py
```

Small statevector simulator: zero state, Hadamard, Rx, Ry, Rz, CNOT, exact measurement probability, and optional shot sampling.

```text
src/fys5419_project_2/model.py
```

Quantum classifier: feature encoders, ansatz circuits, predictions, loss, accuracy, parameter-shift Jacobian, and gradient.

```text
src/fys5419_project_2/optimizers.py
```

Training loops: Adam and plain gradient descent.

```text
src/fys5419_project_2/data.py
```

Data loading and scaling for binary Iris and optional Breast Cancer.

```text
src/fys5419_project_2/baselines.py
```

Classical logistic-regression baseline.

```text
scripts/run_iris_experiment.py
```

Main experiment for the report.

```text
scripts/run_variation_study.py
```

Compares different encoders and ansaetze.

---

## 10. How the implementation maps to the project tasks

### Project 2a: Encoding data into a quantum state

Implemented in:

```python
QuantumCircuitClassifier._encode
```

The default encoding is:

```text
H -> Rz(2π x_j)
```

for each feature/qubit.

### Project 2b: Parameterized gates / ansatz

Implemented in:

```python
QuantumCircuitClassifier._apply_simple_ansatz
QuantumCircuitClassifier._apply_strong_ansatz
```

The simple ansatz uses Ry rotations and CNOT entanglement.  The strong ansatz adds more rotations and ring entanglement.

### Project 2c: Measurement and inference

Implemented in:

```python
QuantumCircuitClassifier.predict_one
QuantumCircuitClassifier.predict_proba
```

The prediction is the probability that the measured qubit is in state `|1>`.

### Project 2d: Model for a full design matrix

Implemented in:

```python
QuantumCircuitClassifier.predict_proba(X, theta)
```

It returns one prediction for each row of `X`.

### Project 2e: Parameter-shift training

Implemented in:

```python
QuantumCircuitClassifier.parameter_shift_jacobian
QuantumCircuitClassifier.gradient
train_adam
train_gradient_descent
```

### Project 2f: Variations

Implemented by:

```bash
python scripts/run_variation_study.py
```

and optionally:

```bash
python scripts/run_breast_cancer_experiment.py
```

---

## 11. Suggested report structure

1. **Introduction**: Binary classification with a variational quantum classifier.
2. **Data**: First two Iris targets, feature scaling to `[0,1]`, train/test split.
3. **Quantum model**: Feature map, ansatz, CNOT entanglement, measurement probability.
4. **Training**: Binary cross-entropy, parameter-shift rule, Adam.
5. **Baseline**: Logistic regression.
6. **Results**: Include the metrics JSON/table, loss figure, and variation-study results.
7. **Discussion**: Compare QML and logistic regression, discuss expressivity, computational cost, and limitations of classical statevector simulation.
8. **Conclusion**: Summarize whether the QML model learned the task and what changed with a more expressive ansatz.

---

## 12. Safe Git usage before final delivery

Safe while working:

```bash
git status
git diff
```

Do **not** run these until the project is completely finished:

```bash
git add .
git commit -m "Complete FYS5419 Project 2"
git push
```
