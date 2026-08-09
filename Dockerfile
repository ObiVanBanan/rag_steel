FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models/huggingface

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project

COPY . .

RUN uv sync \
    --frozen \
    --no-dev

EXPOSE 8005

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005", "--workers", "1"]
