FROM python:3.11.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models/huggingface

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project

COPY . .

RUN uv sync \
    --frozen \
    --no-dev

RUN adduser --system --group --home /app app \
    && chown -R app:app /app

USER app

EXPOSE 8005

CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005", "--workers", "1"]
