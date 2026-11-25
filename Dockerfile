# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1. Dependencies. They are copied before the code, so this slow layer is rebuilt only when a
#    requirements file changes. PyTorch comes from the CPU-only index: the default Linux wheel
#    bundles CUDA and is several GB larger, and a small server has no GPU to use it on.
COPY requirements.txt requirements-ml.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt -r requirements-ml.txt

# 2. An unprivileged user for everything that follows. /data holds the saved index.
RUN useradd --create-home --uid 1000 appuser \
 && mkdir /data \
 && chown appuser:appuser /data
USER appuser

# 3. Bake the embedding model into the image: the container then starts fast and needs no internet.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

# 4. The application itself (last, because it changes most often).
COPY --chown=appuser:appuser app ./app

# Set after the download above: from here on the container never tries to reach the Hugging Face Hub.
# The backend is explicit so a missing model is an error, not a silent switch to the weaker embedder.
ENV DOCQA_DATA_DIR=/data \
    DOCQA_EMBEDDING_BACKEND=sentence-transformers \
    DOCQA_EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    HF_HUB_OFFLINE=1

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# ONE worker on purpose: the index lives in this process's memory (and is saved to /data), so
# several workers would each hold their own copy and overwrite each other's saves.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
