# Set up the environment with the required dependencies
setup:
	python -m venv venv
	source venv/bin/activate && \
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.DS_Store' -type f -delete
	rm -rf .tox .coverage htmlcov coverage-reports pylint.txt .pytest_cache

# Run all tests
test-all:
	make test-functions
	make test-load
	make test-models
	make test-preprocess
	make test-coverage

test-clean:
	make test-all
	make test-pipeline
	make test-integration
	make clean

test-coverage:
	coverage report -m --fail-under 80 --omit "*/__init__.py"
	coverage html -d coverage-reports

# Run the tests in the tests directory
test-functions:
	coverage run --source=ml -m pytest -q tests/ml/functions -W ignore::UserWarning
	coverage report -m --fail-under 80 --omit="*/__init__.py,ml/load/*.py,ml/models/*.py,ml/preprocess/*.py"
	coverage html -d coverage-reports

test-load:
	coverage run --source=ml/load -m pytest -q tests/ml/load -W ignore::UserWarning
	make test-coverage

test-models:
	coverage run --source=ml/models -m pytest -q tests/ml/models -W ignore::UserWarning
	make test-coverage

test-preprocess:
	coverage run --source=ml/preprocess -m pytest -q tests/ml/preprocess -W ignore::UserWarning
	make test-coverage

# Run pipeline test
test-pipeline:
	coverage run --source=pipeline -m pytest -q tests/pipeline -W ignore::UserWarning
	make test-coverage

# Run integration test
test-integration:
	coverage run --source=main -m pytest -q tests/integration -W ignore::UserWarning
	make test-coverage

# Run the linter on the code in the ml directory
lint:
	pylint ml/
	pylint pipeline/

# Run the main script in the ml directory
run:
	chmod +x run-model.sh
	sh run-model.sh

# Run Docker build
docker-build:
	sh scripts/docker-build.sh $(tag)
