.PHONY: install test iris iris-long variation breast clean

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

clean:
	rm -rf .pytest_cache **/__pycache__ src/*.egg-info
