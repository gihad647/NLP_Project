import glob
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System API",
    description="Local Retrieval-Augmented Generation system for CV-to-Job matching",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.on_event("startup")
async def auto_ingest_sample_data():
    """On first start, ingest all documents found in /data/raw/ if the collection is empty."""
    try:
        from app.services.vector_store import vector_store
        from app.services.rag_service import rag_service

        if vector_store.collection_count() > 0:
            logger.info("Collection already populated — skipping auto-ingest.")
            return

        data_dir = "/data/raw"
        supported = {".pdf", ".docx", ".doc", ".html", ".htm"}
        files = [
            f for f in glob.glob(os.path.join(data_dir, "*"))
            if os.path.splitext(f)[1].lower() in supported
        ]

        if not files:
            logger.info("No documents found in /data/raw — skipping auto-ingest.")
            return

        logger.info(f"Auto-ingesting {len(files)} documents from {data_dir} ...")
        for f in files:
            try:
                result = rag_service.ingest_document(f)
                logger.info(f"  ✓ {os.path.basename(f)}: {result.chunks_created} chunks ({result.language_detected})")
            except Exception as exc:
                logger.error(f"  ✗ {os.path.basename(f)}: {exc}")

    except Exception as exc:
        logger.error(f"Auto-ingest startup error: {exc}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RAG API"}
