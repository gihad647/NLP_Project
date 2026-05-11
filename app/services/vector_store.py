"""
Vector Store — wraps ChromaDB with sentence-transformer embeddings.

Embedding model: paraphrase-multilingual-MiniLM-L12-v2
  • 50+ languages including Arabic
  • 384-dim vectors — compact & fast
  • Pretrained on paraphrase tasks → good semantic similarity
  • Outperforms monolingual models on Arabic CVs in benchmarks
"""
from typing import List, Dict, Any, Optional
import logging
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        self._client: Optional[chromadb.Client] = None
        self._collection = None
        self._embedder: Optional[SentenceTransformer] = None

    def _get_client(self) -> chromadb.Client:
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self._embedder

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def embed(self, texts: List[str]) -> List[List[float]]:
        embedder = self._get_embedder()
        return embedder.encode(texts, show_progress_bar=False).tolist()

    def add_chunks(self, chunks: List[Dict[str, Any]], source_id: str):
        """Embed and store chunks in ChromaDB."""
        if not chunks:
            return

        collection = self._get_collection()
        texts = [c["content"] for c in chunks]
        embeddings = self.embed(texts)
        ids = [f"{source_id}_chunk_{c['chunk_index']}" for c in chunks]
        metadatas = [
            {
                **c.get("metadata", {}),
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
                "source_id": source_id,
            }
            for c in chunks
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(chunks)} chunks for source: {source_id}")

    def query(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Embed query and retrieve top-k chunks by cosine similarity."""
        collection = self._get_collection()
        query_embedding = self.embed([query_text])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            # Cosine distance → similarity score (0–1)
            score = round(1 - dist, 4)
            chunks.append({
                "content": doc,
                "source": meta.get("source", "unknown"),
                "score": score,
                "metadata": meta,
            })

        return chunks

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Return every chunk stored in the collection."""
        collection = self._get_collection()
        total = collection.count()
        if total == 0:
            return []
        results = collection.get(
            limit=total,
            include=["documents", "metadatas"],
        )
        chunks = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            chunks.append({"content": doc, "metadata": meta})
        chunks.sort(key=lambda c: (
            c["metadata"].get("source", ""),
            c["metadata"].get("chunk_index", 0),
        ))
        return chunks

    def collection_count(self) -> int:
        return self._get_collection().count()


# Singleton
vector_store = VectorStore()
