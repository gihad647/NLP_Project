"""
API Routes (Controller layer in MVC)
─────────────────────────────────────
POST /ingest   → upload & process a document
POST /query    → ask a question against the knowledge base
GET  /status   → collection stats
GET  /providers → list available LLM providers
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile
import shutil
import os
from typing import Optional

from app.models.schemas import (
    QueryRequest, QueryResponse,
    IngestRequest, IngestResponse,
    ErrorResponse,
)
from app.services.rag_service import rag_service
from app.services.llm_factory import LLMFactory
from app.services.vector_store import vector_store

router = APIRouter()


# ─── POST /ingest ───────────────────────────────────────────
@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a document file",
    description="Upload a raw PDF, DOCX, or HTML document. The system parses, chunks, embeds, and stores it.",
)
async def ingest_file(
    file: UploadFile = File(...),
):
    allowed_extensions = {".pdf", ".docx", ".doc", ".html", ".htm"}
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}",
        )

    # Save to temp file and ingest
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = rag_service.ingest_document(tmp_path)
        result.file = file.filename  # show original name
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        os.unlink(tmp_path)


# ─── POST /ingest/path ──────────────────────────────────────
@router.post(
    "/ingest/path",
    response_model=IngestResponse,
    summary="Ingest a document by server-side path",
    description="Provide a path to a file already on the server (e.g., /data/raw/cv.pdf).",
)
async def ingest_by_path(request: IngestRequest):
    try:
        return rag_service.ingest_document(request.file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


# ─── POST /query ────────────────────────────────────────────
@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG system",
    description="Submit a natural language query (English or Arabic). Returns an LLM-generated answer with source chunks.",
)
async def query(request: QueryRequest):
    try:
        return rag_service.query(
            query_text=request.query,
            top_k=request.top_k or 5,
            provider=request.provider,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# ─── GET /status ────────────────────────────────────────────
@router.get(
    "/status",
    summary="Collection statistics",
    description="Returns the number of chunks currently stored in the vector database.",
)
async def status():
    try:
        count = vector_store.collection_count()
        return {"status": "ok", "chunks_in_store": count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector store unreachable: {e}")


# ─── GET /chunks ────────────────────────────────────────────
@router.get(
    "/chunks",
    summary="List all stored chunks",
    description="Returns every chunk currently stored in the vector database, grouped by source file.",
)
async def list_chunks():
    try:
        chunks = vector_store.get_all_chunks()
        return {"total": len(chunks), "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector store unreachable: {e}")


# ─── GET /providers ─────────────────────────────────────────
@router.get(
    "/providers",
    summary="List available LLM providers",
    description="Returns all LLM providers registered in the factory.",
)
async def providers():
    return {"providers": LLMFactory.available_providers()}
