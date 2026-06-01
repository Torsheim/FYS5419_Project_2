# FYS5419 Project 2 feedback fixes

The project is already a strong B once the repository content is visible.  These files address the main remaining feedback items:

1. Replace the root README so GitHub no longer presents the project as a skeleton.
2. Add a repeated-seed robustness study with mean +/- standard deviation.
3. Add a finite-shot measurement study with 100, 1000 and 10000 shots.
4. Generate LaTeX tables that can be inserted into the report.

## Apply patch

From WSL, after downloading `FYS5419_Project_2_A_level_patch.zip`:

```bash
cd ~/projects/FYS5419_Project_2
source .venv/bin/activate
cp /mnt/c/Users/torsheim/Downloads/FYS5419_Project_2_A_level_patch.zip .
unzip -o FYS5419_Project_2_A_level_patch.zip
rm FYS5419_Project_2_A_level_patch.zip
```

## Run checks

```bash
python -m pip install -e .
pytest
```

## Run the new robustness studies

Fast version, suitable for checking that everything works:

```bash
make robustness
```

More thorough repeated-seed run, closer to what the feedback asked for:

```bash
python scripts/run_repeated_seed_study.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05 --n-seeds 10
python scripts/run_shot_noise_study.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05 --shots 100 1000 10000 --repeats 30
python scripts/render_robustness_latex_tables.py
```

This creates:

```text
results/iris_repeated_seed_study.csv
results/iris_repeated_seed_summary.csv
results/iris_repeated_seed_study.json
results/iris_shot_noise_study.csv
results/iris_shot_noise_raw.csv
results/iris_shot_noise_study.json
figures/iris_repeated_seed_accuracy.png
figures/iris_repeated_seed_loss.png
figures/iris_shot_noise_accuracy.png
figures/iris_shot_noise_loss.png
report/robustness_tables.tex
```

## Add the new tables to the report

Open:

```bash
nano report/project2_report.tex
```

Find the section called `Reliability and limitations`. Add this line near the beginning of that subsection, or near the end of the Results and discussion section:

```latex
\input{robustness_tables}
```

Then compile:

```bash
cd report
pdflatex project2_report.tex
pdflatex project2_report.tex
cd ..
```

The report should still be well below 10 pages. If it gets too long, keep the tables and remove one less important figure, for example the optional Breast Cancer figure.

## Optional repository cleanup

If the repository still contains unused QFT/QPE skeleton files from the initial scaffold, either remove them or move them into a clearly named folder. Check first:

```bash
find . -iname '*qft*' -o -iname '*qpe*' -o -iname '*placeholder*'
```

For files that are definitely unused, move them like this:

```bash
mkdir -p archive_unused_skeleton
# Example only; use the actual paths printed by find:
# git mv scripts/run_qml_placeholder.py archive_unused_skeleton/
```

## Commit and push

```bash
git status
git add README.md Makefile scripts/run_repeated_seed_study.py scripts/run_shot_noise_study.py scripts/render_robustness_latex_tables.py results figures report A_LEVEL_FIX_GUIDE.md
git commit -m "Add robustness and finite-shot studies"
git push
```

Then open GitHub in a private/incognito browser and check that the root page shows the files, not an empty repository message.
