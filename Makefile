.PHONY: install test iris iris-long variation breast robustness repeated-seed shot-noise robustness-tables clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install -e .

test:
	pytest -q

iris:
	python scripts/run_iris_experiment.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05 --verbose

iris-long:
	python scripts/run_iris_experiment.py --features 4 --layers 2 --epochs 60 --learning-rate 0.05 --verbose

variation:
	python scripts/run_variation_study.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05

breast:
	python scripts/run_breast_cancer_experiment.py --features 4 --layers 2 --epochs 20 --learning-rate 0.03 --verbose

repeated-seed:
	python scripts/run_repeated_seed_study.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05 --n-seeds 5

shot-noise:
	python scripts/run_shot_noise_study.py --features 4 --layers 2 --epochs 20 --learning-rate 0.05 --shots 100 1000 10000 --repeats 30

robustness-tables:
	python scripts/render_robustness_latex_tables.py

robustness: repeated-seed shot-noise robustness-tables

clean:
	rm -rf .pytest_cache **/__pycache__ src/*.egg-info
