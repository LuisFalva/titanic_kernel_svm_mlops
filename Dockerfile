FROM python:3.9

WORKDIR /app

COPY . ./

RUN make setup

# Run unit tests
RUN make test-functions
RUN make test-load
RUN make test-models
RUN make test-preprocess

# Run pipeline test
RUN make test-pipeline

# Run integration test
RUN make test-integration

RUN chmod +x run-model.sh
CMD ["/bin/bash", "./run-model.sh"]