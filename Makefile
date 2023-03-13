# Set up the environment with the required dependencies
setup:
	python -m venv venv
	source venv/bin/activate && \
	venv/bin/pip install -r requirements.txt

# Run all tests
test-all:
	make test-functions
	make test-load
	make test-models
	make test-preprocess
	make test-pipeline
	make test-integration

# Run the tests in the tests directory
test-functions:
	python -m unittest discover -v tests/ml/functions

test-load:
	python -m unittest discover -v tests/ml/load

test-models:
	python -m unittest discover -v tests/ml/models

test-preprocess:
	python -m unittest discover -v tests/ml/preprocess

# Run pipeline test
test-pipeline:
	python -m unittest discover -v tests/pipeline

# Run integration test
test-integration:
	python -m unittest discover -v tests/integration

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
