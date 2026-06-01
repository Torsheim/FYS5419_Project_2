# FYS5419 Project 2 - Quantum Machine Learning

Repository for **FYS5419/FYS9419 Project 2, Alternative 2: Quantum Machine Learning**.

This project implements a binary quantum classifier for the first two Iris classes using a transparent NumPy state-vector simulator. Classical features are encoded into a quantum state, a trainable parameterized circuit is applied, and the prediction is obtained from the probability of measuring one qubit in the state `|1>`. The parameters are trained with binary cross-entropy, the parameter-shift rule, and Adam. Logistic regression is used as the main classical baseline.

The implementation avoids hiding the circuit behind high-level Qiskit or PennyLane abstractions. The gates, CNOT operations, state-vector evolution, exact measurement probabilities, finite-shot sampling, loss function, parameter-shift gradients, and optimizer are visible in the source code.

## Main results

The report studies:

- the basic Iris classifier with an `h_rz` encoder and simple ansatz,
- comparison with logistic regression,
- encoder/ansatz variations,
- repeated-seed robustness over several random splits and initializations,
- finite-shot measurement noise with 100, 1000, and 10000 shots,
- an optional Breast Cancer extension using the first four features.

The basic `h_rz/simple` Iris model reaches about 0.84 test accuracy on the selected split, while logistic regression reaches 1.00. The `ry/simple` and `ry_rz/strong` variants reach 1.00 on the selected split. Repeated-seed and finite-shot studies show that circuit design is important and that 1000-10000 shots preserve the qualitative exact-probability conclusions. These results should not be interpreted as quantum advantage.

## Repository layout

```text
src/fys5419_project_2/
  quantum_state.py      NumPy state-vector simulator, gates, CNOT, measurement
  model.py              Encoders, ansaetze, probabilities, loss, gradients
  data.py               Iris and Breast Cancer data loading and scaling
  optimizers.py         Adam and gradient-descent training
  baselines.py          Logistic-regression baseline

scripts/
  run_iris_experiment.py             Main Iris experiment
  run_variation_study.py             Encoder/ansatz comparison
  run_breast_cancer_experiment.py    Optional Breast Cancer extension
  check_parameter_shift.py           Explicit parameter-shift check
  run_repeated_seed_study.py         Robustness over random seeds/splits
  run_shot_noise_study.py            Finite-shot measurement study
  render_robustness_latex_tables.py  Creates LaTeX robustness tables

tests/                    Unit tests and smoke tests
figures/                  Generated figures used in the report
results/                  Selected metrics and result summaries
report/                   Scientific report PDF and LaTeX source
```

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Tests

```bash
pytest -q
```

or:

```bash
make test
```

## Reproduce the main experiments

```bash
python scripts/run_iris_experiment.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05 --verbose
python scripts/run_variation_study.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05
python scripts/run_breast_cancer_experiment.py --features 4 --layers 2 --epochs 20 --learning-rate 0.03 --verbose
python scripts/make_report_figures.py
```

Equivalent Makefile commands:

```bash
make iris
make variation
make breast
```

## Robustness and finite-shot studies

```bash
python scripts/run_repeated_seed_study.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05 --n-seeds 10
python scripts/run_shot_noise_study.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05 --shots 100 1000 10000 --repeats 30
python scripts/render_robustness_latex_tables.py
```

or:

```bash
make robustness
```

These commands write additional result files to `results/`, figures to `figures/`, and a compact LaTeX table file to `report/robustness_tables.tex`.

## Report

The final report is in:

```text
report/project2_report.pdf
report/project2_report.tex
```

To compile locally:

```bash
cd report
pdflatex project2_report.tex
pdflatex project2_report.tex
cd ..
```

## Interpretation

The correct conclusion is conservative: the quantum classifiers can be implemented transparently and can learn useful binary classifiers, but logistic regression remains a very strong baseline on these small classical data sets. The results demonstrate sensitivity to feature maps and ansaetze, not quantum advantage.
