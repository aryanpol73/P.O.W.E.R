# P.O.W.E.R. API: FastAPI + embedded simulator and agents. Build context: repository root.
#   docker build -f services/api/Dockerfile -t power-api .            (CPU PyTorch for the LSTM)
#   docker build -f services/api/Dockerfile --build-arg TORCH=none .  (smaller; serves without M5)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=8 UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH MPLBACKEND=Agg
# libgomp: OpenMP runtime for LightGBM / XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --default-timeout=120 --retries 10 uv==0.12.5

WORKDIR /app
# 1) third-party dependencies (cached until the lock file changes)
COPY pyproject.toml uv.lock ./
COPY packages/power_core/pyproject.toml packages/power_core/
COPY ml/pyproject.toml ml/
COPY services/api/pyproject.toml services/api/
COPY services/mcp/pyproject.toml services/mcp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package power-api --extra postgres
ARG TORCH=cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TORCH" = "cpu" ]; then uv pip install --python /opt/venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu; fi
# 2) our packages
COPY packages packages
COPY ml ml
COPY services/api services/api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package power-api --extra postgres --inexact

RUN useradd --create-home --uid 1000 power && mkdir -p /app/data /app/artifacts /app/reports /app/configs \
    && chown -R power /app
USER power
ENV POWER_ROOT=/app
EXPOSE 8000
# Seeds the database on first start (data/ must be mounted), then serves with one worker: the embedded
# simulator is the single writer.
CMD ["sh", "-c", "power-api seed --if-empty --wait 120 && exec power-api serve --host 0.0.0.0 --port 8000"]
