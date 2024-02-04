FROM python:3.8

WORKDIR /app
COPY qa_squad .
COPY pyproject.toml .
COPY poetry.lock .

RUN apt-get update && \
    apt-get install -y htop && \
    pip install --upgrade pip &&\
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry config installer.max-workers 10

RUN poetry install -vvv && \
    python -m spacy download en_core_web_sm

CMD sleep infinity