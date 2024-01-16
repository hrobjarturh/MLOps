# Use a base image with Python installed
FROM python:3.11-slim

# 
WORKDIR /code

# 
COPY ./requirements_serve.txt /code/requirements_serve.txt

RUN pip install -r /code/requirements_serve.txt --no-cache-dir


COPY ./animals10 /code/animals10
COPY ./models /code/models
COPY ./models/googlenet_model_0.pth /code/models/googlenet_model_0.pth
COPY ./data/processed/test /code/models
COPY ./config /code/config

# 
CMD ["uvicorn", "animals10.serve:app", "--host", "0.0.0.0", "--port", "80"]