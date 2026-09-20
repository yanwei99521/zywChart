FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    CHART_OUTPUT_DIR=/app/output

RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml build_chart.py ./
COPY chart_service ./chart_service
RUN pip install --no-cache-dir .

COPY bitcoin-power-law-*.png bitcoin-power-law-*.svg model-summary.json ./output/
COPY data ./output/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl --fail http://127.0.0.1:8000/api/v1/health || exit 1

# Keep one process: the in-process scheduler must have a single owner.
CMD ["uvicorn", "chart_service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

