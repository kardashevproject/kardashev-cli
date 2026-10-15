FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCOMMONS_STORE_PATH=/data/opencommons/store.json

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY commons ./commons

RUN pip install --no-cache-dir -e . && \
    mkdir -p /data/opencommons && \
    cp /app/commons/store.json /data/opencommons/store.json && \
    chmod -R 777 /data/opencommons

EXPOSE 7860

CMD ["kardashev", "commons", "serve-hub", "--host", "0.0.0.0", "--port", "7860", "--store", "/data/opencommons/store.json"]
