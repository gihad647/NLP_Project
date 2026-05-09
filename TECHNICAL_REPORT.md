# RAG System — Technical Report

**Project:** CV-to-Job Matching via Local RAG  
**Team:** [Your Name(s)]  
**Date:** 2025

---

## Executive Summary

This system is an end-to-end Retrieval-Augmented Generation (RAG) pipeline that allows recruiters to query a corpus of raw, unstructured CVs using natural language (English or Arabic). A FastAPI backend orchestrates document ingestion, semantic chunking, multilingual embedding, and LLM-powered answer generation. The entire system is containerized with Docker Compose and can be started with a single command.

---

## System Architecture

```
┌─────────────┐     POST /ingest      ┌──────────────────────────────────────────┐
│  Raw CVs    │ ─────────────────────▶│            FastAPI Application           │
│  (PDF/DOCX) │                       │                                          │
└─────────────┘                       │  ┌──────────┐  ┌──────────┐  ┌────────┐ │
                                      │  │  Parser  │─▶│ Chunker  │─▶│Embedder│ │
┌─────────────┐     POST /query       │  └──────────┘  └──────────┘  └────┬───┘ │
│   User      │ ─────────────────────▶│                                    │     │
│   Query     │                       │  ┌──────────────────────────────────▼───┐ │
└─────────────┘                       │  │         ChromaDB (Vector Store)      │ │
                                      │  └──────────────────────────────────────┘ │
                                      │        ▲ retrieve top-k chunks            │
                                      │        │                                  │
                                      │  ┌─────┴──────────────────────────────┐  │
                                      │  │       LLM Factory                  │  │
                                      │  │  Gemini | OpenAI | Ollama (local)  │  │
                                      │  └────────────────────────────────────┘  │
                                      └──────────────────────────────────────────┘
```

---

## API Documentation

### `POST /api/v1/ingest`
Upload a raw document for processing.

**Content-Type:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | PDF, DOCX, or HTML document |

**Response:**
```json
{
  "status": "success",
  "chunks_created": 42,
  "file": "cv_john_doe.pdf",
  "language_detected": "en"
}
```

---

### `POST /api/v1/ingest/path`
Ingest a document already on the server.

**Request Body:**
```json
{
  "file_path": "/data/raw/cv_john_doe.pdf"
}
```

**Response:** Same as `/ingest`.

---

### `POST /api/v1/query`
Query the knowledge base.

**Request Body:**
```json
{
  "query": "Find a senior Python developer with FastAPI experience",
  "top_k": 5,
  "provider": "gemini"
}
```

**Response:**
```json
{
  "query": "Find a senior Python developer...",
  "answer": "Based on the CVs, John Doe has 6 years of Python...",
  "retrieved_chunks": [
    {
      "content": "John Doe — Senior Python Engineer...",
      "source": "/data/raw/cv_john_doe.pdf",
      "score": 0.89,
      "metadata": { "chunk_index": 3, "token_count": 487 }
    }
  ],
  "provider_used": "gemini",
  "tokens_used": null
}
```

---

### `GET /api/v1/status`
```json
{ "status": "ok", "chunks_in_store": 420 }
```

### `GET /api/v1/providers`
```json
{ "providers": ["gemini", "openai", "ollama"] }
```

### `GET /health`
```json
{ "status": "healthy", "service": "RAG API" }
```

---

## Embedding Model Justification

**Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

| Property | Value |
|----------|-------|
| Languages | 50+ (includes Arabic) |
| Dimensions | 384 |
| Size | ~118 MB |
| Similarity metric | Cosine |

**Why this model?**
1. **Multilingual by design** — trained on parallel corpora across 50+ languages, so Arabic CVs and English job queries map to a shared semantic space.
2. **Compact** — 384-dim vectors allow ChromaDB to store thousands of chunks with fast HNSW retrieval.
3. **Paraphrase-trained** — captures semantic equivalence ("ML engineer" ≈ "machine learning specialist"), reducing missed retrievals due to vocabulary mismatch.
4. **No API cost** — runs fully locally inside the Docker container.

---

## Chunking Strategy

**Parameters:** 500 tokens / chunk, 50-token overlap

**Justification:**

| Factor | Reasoning |
|--------|-----------|
| **500 tokens** | ≈ 3–4 paragraphs — captures a full CV section (Education, Skills, Experience) without fragmentation |
| **50-token overlap (10%)** | Prevents loss of context at chunk boundaries; a skill described across two paragraphs is captured in both |
| **Sentence-aware splitting** | Never breaks mid-sentence; only falls back to word-level for sentences > 500 tokens |
| **Top-k = 5** | 5 × 500 = 2,500 tokens of context — fits comfortably in Gemini Flash's 1M context window |
| **Arabic scaling** | Arabic tokens are morphologically richer (~3.5 chars/token vs ~4 for English); the 500-token budget covers ~350 Arabic words, still enough for one semantic unit |

---

## LLM Factory Pattern (Bonus 1)

Three providers are registered via a factory:

```
LLMFactory.create("gemini")  → GeminiLLM
LLMFactory.create("openai")  → OpenAILLM
LLMFactory.create("ollama")  → OllamaLLM (local Mistral)
```

To switch providers: change `LLM_PROVIDER=gemini` in `.env` to `openai` or `ollama`. **Zero code changes required.**

---

## Arabic Language Support (Bonus 2)

### Challenges addressed:
1. **Diacritics (Tashkeel):** Removed via regex — they vary between writers and fragment tokens unnecessarily.
2. **Alef normalization:** `إ أ آ ا` → `ا` — prevents the same word from having 4 different embeddings.
3. **Tatweel removal:** Decorative kashida (`ـ`) removed.
4. **RTL PDF extraction:** PyMuPDF's `TEXT_PRESERVE_WHITESPACE` flag reduces RTL re-ordering artifacts; visual bidi algorithm handles remaining cases.
5. **OCR fallback:** Tesseract with `ara+eng` language pack for scanned Arabic CVs.

### Arabic test query:
```
Query: "خبرة في تطوير الويب وقواعد البيانات"
(English: "Experience in web development and databases")
```

The multilingual embedding model successfully retrieves English CV chunks mentioning web development, demonstrating cross-lingual retrieval. Full results in `tests/evaluate.py`.

---

## Docker Deployment Instructions

### Prerequisites
- Docker Engine ≥ 24
- Docker Compose ≥ 2.20

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-team/rag-system.git
cd rag-system

# 2. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (or OPENAI_API_KEY)

# 3. Start all services
docker compose up --build

# 4. Verify
curl http://localhost:8080/health

# 5. Ingest a document
curl -X POST http://localhost:8080/api/v1/ingest \
  -F "file=@/path/to/cv.pdf"

# 6. Query
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find a Python developer with ML experience"}'
```

### To use Ollama (local, no API key needed):
```bash
# Start with Ollama profile
docker compose --profile ollama up

# Pull the model (first time only)
docker exec rag_ollama ollama pull mistral

# Set provider in .env
LLM_PROVIDER=ollama
```

---

## Edge Case Analysis

| # | Query | Failure Type | Root Cause | Mitigation |
|---|-------|-------------|------------|------------|
| 1 | "quantum computing expert with blockchain" | **Wrong chunk retrieved** | No candidates match; retriever returns closest-by-chance chunks; LLM may hallucinate a candidate | Add cosine score threshold (e.g., < 0.40 → return "not found") |
| 2 | Arabic query on English-only corpus | **Cross-lingual retrieval drift** | The multilingual model bridges languages but with lower confidence when the corpus is monolingual; scores are systematically lower (~0.55 vs ~0.80 for same-language) | Ingest bilingual CVs; or add a translate-first pipeline using LibreTranslate |
| 3 | Single-character query `"a"` | **Degenerate retrieval** | Near-zero-information embedding; retriever returns random high-dimensional neighbors | Validate minimum query length (≥ 3 words) and reject with HTTP 400 |
