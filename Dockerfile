FROM python:3.9

WORKDIR /app

COPY . ./

RUN pip install -r requirements.txt

# Run unit tests
RUN python -m unittest discover -v tests/ml/functions
RUN python -m unittest discover -v tests/ml/load
RUN python -m unittest discover -v tests/ml/models
RUN python -m unittest discover -v tests/ml/preprocess

# Run pipeline test
RUN python -m unittest discover -v tests/pipeline

# Run integration test
RUN python -m unittest discover -v tests/integration

RUN chmod +x run-model.sh
CMD ["/bin/bash", "./run-model.sh"]