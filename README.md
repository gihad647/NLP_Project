
ْ# 🔍 Local RAG System — 

An end-to-end, containerized Retrieval-Augmented Generation (RAG) system built with FastAPI, ChromaDB, and multilingual embeddings. Supports English and Arabic CVs.

## Quick Start

```bash
cp .env.example .env          # Add your API key
docker compose up --build     # Start everything
```

API will be available at `http://localhost:8080`  
Interactive docs: `http://localhost:8080/docs`

## Project Structure

```
rag_project/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints (Controller)
│   ├── core/
│   │   ├── config.py          # Settings & env vars
│   │   ├── parser.py          # PDF/DOCX/HTML parser + Arabic normalization
│   │   └── chunker.py         # Sentence-aware chunking (500t / 50t overlap)
│   ├── models/
│   │   └── schemas.py         # Pydantic request/response models
│   ├── services/
│   │   ├── vector_store.py    # ChromaDB wrapper
│   │   ├── llm_factory.py     # Factory: Gemini | OpenAI | Ollama
│   │   └── rag_service.py     # RAG orchestration
│   └── main.py
├── tests/
│   └── evaluate.py            # Phase 4 evaluation with edge cases
├── data/raw/                  # Put your raw PDFs/CVs here
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── TECHNICAL_REPORT.md
```

## Features

- **No toy data:** Handles raw PDFs, DOCX, HTML — including scanned docs (OCR)
- **Dirty hands:** Custom parsing, Arabic normalization, sentence-aware chunking
- **MVC architecture:** Routes / Services / Models cleanly separated
- **LLM Factory (Bonus 1):** Switch between Gemini, OpenAI, Ollama with one env var
- **Arabic support (Bonus 2):** Diacritic removal, alef normalization, RTL extraction, OCR

## Evaluation

```bash
python tests/evaluate.py
```

See `TECHNICAL_REPORT.md` for full architecture, API docs, and edge case analysis.
