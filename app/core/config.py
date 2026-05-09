from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # LLM Provider: "gemini" | "openai" | "ollama"
    LLM_PROVIDER: Literal["gemini", "openai", "ollama"] = "gemini"
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "mistral"

    # Embedding
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # ChromaDB
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "rag_documents"

    # Chunking Strategy
    CHUNK_SIZE: int = 500        # tokens per chunk
    CHUNK_OVERLAP: int = 50      # overlap between chunks
    TOP_K: int = 5               # number of chunks to retrieve

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
