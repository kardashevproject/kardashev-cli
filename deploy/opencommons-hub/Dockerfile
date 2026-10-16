FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCOMMONS_HUB_DB=/data/opencommons_hub.db \
    PORT=7860

WORKDIR /app

# Create non-root runtime user and persistent volume directory
RUN useradd -m -u 1000 opencommons && \
    mkdir -p /data && \
    chown -R opencommons:opencommons /data /app

COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY commons/ /app/commons/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

USER opencommons

EXPOSE 7860

CMD ["python3", "-c", "import os; from kardashev_cli.opencommons_hub import run_hub_server; run_hub_server(host='0.0.0.0', port=int(os.environ.get('PORT', '7860')), db_path=os.environ.get('OPENCOMMONS_HUB_DB', '/data/opencommons_hub.db'))"]
