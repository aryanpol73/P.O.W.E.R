# P.O.W.E.R. MCP server (streamable HTTP on :8765/mcp). Build context: repository root.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=8 UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 uv==0.12.5
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/power_core/pyproject.toml packages/power_core/
COPY ml/pyproject.toml ml/
COPY services/api/pyproject.toml services/api/
COPY services/mcp services/mcp
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --package power-mcp
RUN useradd --create-home --uid 1000 power
USER power
EXPOSE 8765
CMD ["power-mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8765"]
