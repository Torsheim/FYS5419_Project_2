# Final repository checklist

Before submitting, verify that GitHub shows the files in the repository root, not an empty repository page.

```bash
cd ~/projects/FYS5419_Project_2
source .venv/bin/activate
pytest -q
ls scripts/run_repeated_seed_study.py scripts/run_shot_noise_study.py
ls results/iris_repeated_seed_summary.csv results/iris_shot_noise_study.csv
ls figures/iris_repeated_seed_accuracy.png figures/iris_shot_noise_accuracy.png
ls report/project2_report.pdf report/project2_report.tex
pdfinfo report/project2_report.pdf | grep Pages
```

Then commit and push:

```bash
git status
git add README.md README_QML.md Makefile A_LEVEL_FIX_GUIDE.md FINAL_REPOSITORY_CHECKLIST.md \
        scripts/run_repeated_seed_study.py scripts/run_shot_noise_study.py scripts/render_robustness_latex_tables.py \
        results figures report
git commit -m "Sync final report, robustness scripts, and selected results"
git push
```
