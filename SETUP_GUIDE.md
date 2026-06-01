# Terminal guide for `FYS5419_Project_2`

This guide assumes that you first create an empty GitHub repository named **FYS5419_Project_2** in your GitHub account. Do not initialize it with a README on GitHub if you want the commands below to work without merging.

Replace `<YOUR_GITHUB_USERNAME>` with your GitHub username.

## 1. Unzip and enter the project folder

```bash
unzip FYS5419_Project_2.zip
cd FYS5419_Project_2
```

## 2. Create a Python environment

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 3. Optional: clone course sources locally

The folder `external/` is ignored by git, so it can contain a local copy of the course repository without making your own repository huge.

```bash
git clone --depth 1 --branch gh-pages \
  https://github.com/CompPhysics/QuantumComputingMachineLearning.git \
  external/QuantumComputingMachineLearning
```

Useful Project 2 files will then be under:

```text
external/QuantumComputingMachineLearning/doc/Projects/2026/Project2/
```

## 4. Check that the skeleton works

```bash
make test
make pdf
```

The report PDF will be created as:

```text
report/main.pdf
```

## 5. Initialize git locally

```bash
git init
git add .
git commit -m "Initial FYS5419 Project 2 structure"
git branch -M main
```

## 6. Connect to your GitHub repository

SSH remote:

```bash
git remote add origin \
  git@github.com:<YOUR_GITHUB_USERNAME>/FYS5419_Project_2.git
git push -u origin main
```

HTTPS remote:

```bash
git remote add origin \
  https://github.com/<YOUR_GITHUB_USERNAME>/FYS5419_Project_2.git
git push -u origin main
```

Use either SSH or HTTPS, not both.

## 7. Regular workflow after adding code

```bash
git status
git add src scripts notebooks report results tests
git commit -m "Describe what you changed"
git push
```

## 8. Before hand-in

```bash
make test
make pdf
git status
```

Commit the final report PDF if your course group wants the PDF stored in the repository, otherwise commit only the LaTeX source and figures. The project brief asks for a GitHub/GitLab repository link and a folder containing selected results, so keep your final plots/tables in `results/`.
