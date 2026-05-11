FROM python:3.11-slim

# System deps (Tesseract OCR + Arabic language pack)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast, backtrack-free dependency resolution
RUN pip install --no-cache-dir uv

# Install torch CPU-only BEFORE copying requirements.txt so this layer
# stays cached even when requirements.txt changes (saves 180 MB re-download)
RUN uv pip install --system --no-cache-dir \
    "torch==2.3.1" \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (torch already satisfied, no CUDA packages pulled)
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Pre-download the embedding model so the container is self-contained
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Data directory for raw documents
RUN mkdir -p /data/raw

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
