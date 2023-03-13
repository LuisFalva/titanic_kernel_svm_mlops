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
	make test-pipeline
	make test-integration

test-clean:
	make test-all
	make clean

# Run the tests in the tests directory
test-functions:
	coverage run -m unittest discover -v tests/ml/functions
	coverage report --fail-under 80
	coverage html -d coverage-reports

test-load:
	coverage run -m unittest discover -v tests/ml/load
	coverage report --fail-under 80
	coverage html -d coverage-reports

test-models:
	coverage run -m unittest discover -v tests/ml/models
	coverage report --fail-under 80
	coverage html -d coverage-reports

test-preprocess:
	coverage run -m unittest discover -v tests/ml/preprocess
	coverage report --fail-under 80
	coverage html -d coverage-reports

# Run pipeline test
test-pipeline:
	coverage run -m unittest discover -v tests/pipeline
	coverage report --fail-under 80
	coverage html -d coverage-reports

# Run integration test
test-integration:
	coverage run -m unittest discover -v tests/integration
	coverage report --fail-under 80
	coverage html -d coverage-reports

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
