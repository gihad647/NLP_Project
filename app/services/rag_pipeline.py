from typing import List, Dict, Any
from app.services.parser import PDFParser
from app.services.chunker import ChunkingService
from app.services.embedder import EmbedderFactory
from app.services.vector_store import VectorStore
from app.services.llm import LLMFactory
from app.core.config import settings


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline:
    1. Parse PDFs  →  2. Chunk  →  3. Embed + Store  →  4. Query + Generate
    """

    def __init__(self):
        self.parser = PDFParser()
        self.chunker = ChunkingService()
        self.embedder = EmbedderFactory.create(settings.EMBEDDING_PROVIDER)
        self.vector_store = VectorStore(embedder=self.embedder)
        self.llm = LLMFactory.create(settings.LLM_PROVIDER)

        print(
            f"[RAGPipeline] 🚀 Initialized | "
            f"Embedder: {self.embedder.model_name} | "
            f"LLM: {self.llm.provider_name}/{self.llm.model_name}"
        )

    # ──────────────────────────────────────────
    #  Ingestion
    # ──────────────────────────────────────────

    def ingest_directory(self, dir_path: str) -> Dict[str, Any]:
        """
        Full ingestion pipeline: parse all PDFs → chunk → embed → store.

        Args:
            dir_path: Directory containing PDF files.

        Returns:
            Summary dict with stats.
        """
        print(f"\n[RAGPipeline] 📂 Ingesting directory: {dir_path}")

        # Phase 1: Parse
        documents = self.parser.parse_directory(dir_path)
        if not documents:
            return {"files_processed": 0, "chunks_stored": 0,
                    "message": "No PDF files found in the directory."}

        # Phase 2: Chunk
        chunks = self.chunker.chunk_documents(documents)

        # Phase 3: Embed + Store
        stored = self.vector_store.add_chunks(chunks)

        summary = {
            "files_processed": len(documents),
            "chunks_stored": stored,
            "message": f"Successfully ingested {len(documents)} resumes into {stored} chunks.",
        }
        print(f"[RAGPipeline] ✅ Ingestion complete: {summary}")
        return summary

    # ──────────────────────────────────────────
    #  Querying
    # ──────────────────────────────────────────

    def query(self, user_query: str, top_k: int = settings.TOP_K) -> Dict[str, Any]:
        """
        Full RAG query pipeline: embed query → retrieve → generate answer.

        Args:
            user_query: Natural language question (English or Arabic).
            top_k: Number of chunks to retrieve.

        Returns:
            Dict with answer, retrieved chunks, and metadata.
        """
        print(f"\n[RAGPipeline] 🔍 Query: {user_query!r}")

        # Step 1: Retrieve top-k chunks
        retrieved_chunks = self.vector_store.query(user_query, top_k=top_k)

        if not retrieved_chunks:
            return {
                "query": user_query,
                "answer": "No relevant documents found. Please ingest resumes first.",
                "retrieved_chunks": [],
                "model_used": f"{self.llm.provider_name}/{self.llm.model_name}",
            }

        # Step 2: Extract text for context injection
        context_texts = [chunk["content"] for chunk in retrieved_chunks]

        # Step 3: Generate answer via LLM
        answer = self.llm.generate(prompt=user_query, context_chunks=context_texts)

        print(f"[RAGPipeline] 💬 Answer generated ({len(answer)} chars)")

        return {
            "query": user_query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "model_used": f"{self.llm.provider_name}/{self.llm.model_name}",
        }

    # ──────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "collection_name": self.vector_store.collection_name,
            "total_chunks": self.vector_store.count(),
            "embedding_model": self.embedder.model_name,
            "llm_provider": self.llm.provider_name,
            "llm_model": self.llm.model_name,
        }
