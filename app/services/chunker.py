from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from app.core.config import settings


class ChunkingService:
    """
    Chunking Strategy:
    - Uses RecursiveCharacterTextSplitter for semantic-aware splitting.
    - Chunk Size: 500 chars — large enough to preserve a full job experience
      or skill section, small enough to stay focused for embedding.
    - Overlap: 50 chars — prevents information loss at boundaries, ensuring
      a multi-line skill or date range is never split without context.
    - Separators: paragraph → newline → sentence → word (hierarchical fallback).
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "،", " ", ""],
            # "،" = Arabic comma; included for Arabic CV support
            length_function=len,
        )

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a parsed document into chunks with metadata.

        Args:
            document: Output from PDFParser.parse_pdf()

        Returns:
            List of chunk dicts with text and metadata.
        """
        raw_text = document["text"]
        chunks_text = self.splitter.split_text(raw_text)

        chunks = []
        for idx, chunk_text in enumerate(chunks_text):
            chunk_id = f"{document['filename']}_chunk_{idx}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "filename": document["filename"],
                        "source": document["source"],
                        "chunk_id": chunk_id,
                        "chunk_index": idx,
                        "total_chunks": len(chunks_text),
                        "language": document.get("language", "unknown"),
                        "pages": document.get("pages", 0),
                    },
                }
            )

        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk a list of documents."""
        all_chunks = []
        for doc in documents:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
            print(
                f"[Chunker] 📄 {doc['filename']} → {len(doc_chunks)} chunks"
            )
        return all_chunks
