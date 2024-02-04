FROM python:3.8

WORKDIR /model_api
COPY . .
RUN apt-get update && \
    pip install fastapi && \
    pip install uvicorn && \
    pip install torch && \
    pip install transformers && \
    pip install accelerate

EXPOSE 8000
CMD uvicorn --host 0.0.0.0 --port 5000 model_api.api.main:app