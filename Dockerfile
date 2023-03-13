FROM python:3.9

WORKDIR /app

COPY . ./

RUN pip install -r requirements.txt

# Run unit tests
RUN make test-all

RUN chmod +x run-model.sh
CMD ["/bin/bash", "./run-model.sh"]