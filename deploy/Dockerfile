# CardioLens on the server (D153): web UI, /api and the scanner's /scans in one container.
# Unlike the repo-root Dockerfile (local demo with Ollama) it carries the scanner extras and
# its ONNX segmenter, and the Claude client for the narrative (ANTHROPIC_API_KEY; without a
# key the deterministic offline text is used).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY models ./models
COPY data/samples ./data/samples
RUN pip install --upgrade pip && pip install -e ".[digitize,scanner,llm,dicom]"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c \
  "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "cardiolens.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
