# Base image
FROM python:3.11-slim

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY config/ config/
COPY LICENSE LICENSE
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY animals10/ animals10/

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "animals10/train_model.py"]