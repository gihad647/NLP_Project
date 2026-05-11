from pydantic import BaseModel, Field
from typing import Optional, List


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query in English or Arabic")
    top_k: Optional[int] = Field(5, description="Number of chunks to retrieve")
    provider: Optional[str] = Field(None, description="Override LLM provider: gemini | openai | openrouter | groq | ollama")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Find a senior Python developer with FastAPI experience",
                "top_k": 5,
                "provider": "gemini"
            }
        }


class ChunkResult(BaseModel):
    content: str
    source: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[ChunkResult]
    provider_used: str
    tokens_used: Optional[int] = None


class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Path to the document to ingest")

    class Config:
        json_schema_extra = {
            "example": {"file_path": "/data/raw/cv_john_doe.pdf"}
        }


class IngestResponse(BaseModel):
    status: str
    chunks_created: int
    file: str
    language_detected: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
