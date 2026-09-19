FROM python:3.11-slim

WORKDIR /app
ENV HF_HOME=/opt/hf PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# CPU-only torch: the default Linux wheel bundles CUDA and adds gigabytes the service never uses.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Editable install on purpose: config.py resolves data/ and reports/ relative to the source tree.
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .

# Bake the embedding model into the image so the first request does not download it.
# The judge is never run at build time: it needs GEMINI_API_KEY at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY data/golden ./data/golden

RUN useradd --create-home app && mkdir -p reports && chown -R app /app /opt/hf
USER app

# Render injects $PORT; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn evalforge.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
