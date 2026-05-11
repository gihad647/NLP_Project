from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # LLM Provider: "gemini" | "openai" | "openrouter" | "groq" | "ollama"
    LLM_PROVIDER: Literal["gemini", "openai", "openrouter", "groq", "ollama"] = "groq"
    LLM_MODEL: str = "mistral"            # used by Ollama
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "google/gemma-4-31b-it:free"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "mistral"

    # Embedding
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_PROVIDER: str = "huggingface"

    # ChromaDB
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "rag_documents"

    # Chunking Strategy
    CHUNK_SIZE: int = 400        # chars per chunk
    CHUNK_OVERLAP: int = 50      # overlap between chunks
    TOP_K: int = 5               # number of chunks to retrieve

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
