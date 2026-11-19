FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project manifest and install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Pre-download spaCy NER model and Hugging Face embedding model into image
ENV HF_HOME=/root/.cache/huggingface
RUN uv run python -m spacy download en_core_web_sm && \
    uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application source code
COPY scache ./scache
COPY README.md ./

# Environment defaults
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONPATH=/app
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "scache.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
