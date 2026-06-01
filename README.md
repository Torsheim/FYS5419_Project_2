# FYS5419 Project 2

Repository skeleton for **FYS5419/9419 - Quantum Computing and Quantum Machine Learning**, Project 2, Spring 2026.

The repository is set up so you can add code later while keeping the report, figures, results, and reproducibility files organized from the start.

## What is included

```text
FYS5419_Project_2/
├── docs/                  # Project description PDF and source files
├── report/                # Scientific report LaTeX template
├── src/                   # Python package for project code
├── scripts/               # Command-line entry points / experiments
├── notebooks/             # Exploration notebooks
├── data/                  # Optional local data files
├── results/               # Figures, tables, and numerical output
├── tests/                 # Smoke tests and later unit tests
├── external/              # Optional local clone of course sources (ignored by git)
├── SETUP_GUIDE.md         # Terminal commands for GitHub setup
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment alternative
├── pyproject.toml         # Package metadata and tool config
└── Makefile               # Build/test helper commands
```

## Project paths

The official brief asks you to select one project path. The skeleton supports all four paths, with placeholders for:

1. Quantum Fourier Transform and Quantum Phase Estimation
2. Quantum Machine Learning on Iris / Breast Cancer data
3. Variational Quantum Boltzmann Machines
4. Adaptive VQE / eigenvalue problems

Once you decide which path to submit, delete or ignore the unused sections in `report/main.tex` and the unused code skeletons in `src/fys5419_project_2/`.

## Recommended workflow

1. Read `docs/FYS5419_Project_2_Project_Description.pdf`.
2. Choose one project path.
3. Put mathematical derivations and written work in `report/main.tex`.
4. Put reusable code in `src/fys5419_project_2/`.
5. Put small experiment scripts in `scripts/`.
6. Save final plots in `results/figures/` and tables in `results/tables/`.
7. Build the final report PDF with `make pdf`.

## Useful commands

```bash
make install      # install package in editable mode
make test         # run tests
make pdf          # build report/main.pdf
make clean        # remove LaTeX build files
```

## Course source links

- Course repository: https://github.com/CompPhysics/QuantumComputingMachineLearning
- Project 2 folder: https://github.com/CompPhysics/QuantumComputingMachineLearning/tree/gh-pages/doc/Projects/2026/Project2
- Project 2 LaTeX source: https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/Projects/2026/Project2/pdf/Project2.tex

## Notes on Qiskit

The project text contains older examples such as `qk.execute` and `qk.Aer`. With current Qiskit/Aer installations, it is usually safer to import the simulator from `qiskit_aer`, for example:

```python
from qiskit_aer import AerSimulator
```

Update the example code accordingly when you implement the project.
