"""
RAG Service — orchestrates the full pipeline:
  ingest → parse → chunk → embed → store
  query  → embed → retrieve → inject → generate
"""
import hashlib
import logging
from pathlib import Path
from typing import Optional

from app.core.parser import parse_document
from app.core.chunker import chunk_text
from app.core.config import settings
from app.models.schemas import (
    QueryResponse,
    IngestResponse,
    ChunkResult,
)
from app.services.vector_store import vector_store
from app.services.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class RAGService:
    # ─── Ingestion ──────────────────────────────────────────

    def ingest_document(self, file_path: str) -> IngestResponse:
        """Parse → chunk → embed → store a raw document."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Ingesting: {file_path}")

        # 1. Parse (handles PDF, DOCX, HTML — messy or clean)
        parsed = parse_document(file_path)
        full_text = parsed["full_text"]
        metadata = parsed["metadata"]

        # 2. Detect language
        from app.core.parser import detect_language
        lang = detect_language(full_text)

        # 3. Chunk (sentence-aware, 500 tokens / 50 overlap)
        chunks = chunk_text(
            text=full_text,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            metadata=metadata,
        )

        if not chunks:
            raise ValueError("No text could be extracted from the document.")

        # 4. Store in ChromaDB
        source_id = _file_hash(file_path)
        vector_store.add_chunks(chunks, source_id=source_id)

        return IngestResponse(
            status="success",
            chunks_created=len(chunks),
            file=file_path,
            language_detected=lang,
        )

    # ─── Query ──────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        provider: Optional[str] = None,
    ) -> QueryResponse:
        """Embed query → retrieve chunks → generate answer via LLM."""

        if not query_text.strip():
            raise ValueError("Query cannot be empty.")
        if len(query_text.strip()) < 3:
            raise ValueError("Query too short — please provide at least 3 characters.")

        logger.info(f"Query: '{query_text[:80]}' | top_k={top_k} | provider={provider}")

        # 1. Retrieve relevant chunks
        retrieved = vector_store.query(query_text, top_k=top_k)

        if not retrieved:
            return QueryResponse(
                query=query_text,
                answer="I could not find any relevant documents in the knowledge base.",
                retrieved_chunks=[],
                provider_used=provider or settings.LLM_PROVIDER,
            )

        # 2. Build LLM (factory pattern — swap provider with config)
        llm = LLMFactory.create(provider)

        # 3. Inject chunks into LLM context → generate
        context_chunks = [c["content"] for c in retrieved]
        result = llm.generate(query_text, context_chunks)

        return QueryResponse(
            query=query_text,
            answer=result["answer"],
            retrieved_chunks=[
                ChunkResult(
                    content=c["content"],
                    source=c["source"],
                    score=c["score"],
                    metadata=c["metadata"],
                )
                for c in retrieved
            ],
            provider_used=provider or settings.LLM_PROVIDER,
            tokens_used=result.get("tokens_used"),
        )


def _file_hash(file_path: str) -> str:
    """Stable ID for a file based on its path."""
    return hashlib.md5(file_path.encode()).hexdigest()[:12]


# Singleton
rag_service = RAGService()
