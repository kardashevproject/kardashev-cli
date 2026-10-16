FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCOMMONS_DB_PATH=/data/opencommons/opencommons.db \
    OPENCOMMONS_HOST=0.0.0.0 \
    OPENCOMMONS_PORT=7860

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY commons ./commons

RUN pip install --no-cache-dir -e . && \
    mkdir -p /data/opencommons && \
    chmod -R 777 /data/opencommons

EXPOSE 7860
VOLUME ["/data/opencommons"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz')" || exit 1

CMD ["kardashev", "commons", "serve-hub", "--host", "0.0.0.0", "--port", "7860", "--store", "/data/opencommons/opencommons.db"]
