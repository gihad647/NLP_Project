from abc import ABC, abstractmethod
from typing import List
from app.core.config import settings


# ─────────────────────────────────────────────
#  Abstract Base
# ─────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Abstract interface for all embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts and return vectors."""

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""


# ─────────────────────────────────────────────
#  Concrete Implementations
# ─────────────────────────────────────────────

class HuggingFaceEmbedder(BaseEmbedder):
    """
    Uses sentence-transformers locally.
    Default: paraphrase-multilingual-MiniLM-L12-v2
    Supports Arabic + English in the same vector space.
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        print(f"[Embedder] ✅ Loaded HuggingFace model: {model_name}")

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self._model.encode([query])[0].tolist()


class OpenAIEmbedder(BaseEmbedder):
    """Uses OpenAI's text-embedding-ada-002 via API."""

    def __init__(self, model_name: str = "text-embedding-ada-002"):
        import openai
        self._model_name = model_name
        self._client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model_name)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]


class OllamaEmbedder(BaseEmbedder):
    """Uses a local Ollama model for embeddings."""

    def __init__(self, model_name: str = "nomic-embed-text"):
        import requests
        self._model_name = model_name
        self._base_url = settings.OLLAMA_BASE_URL
        self._requests = requests

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        resp = self._requests.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model_name, "prompt": query},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────

class EmbedderFactory:
    """
    Factory Design Pattern for embedding providers.
    Switch providers via EMBEDDING_PROVIDER env variable.
    """

    _registry = {
        "huggingface": HuggingFaceEmbedder,
        "openai": OpenAIEmbedder,
        "ollama": OllamaEmbedder,
    }

    @classmethod
    def create(cls, provider: str = settings.EMBEDDING_PROVIDER) -> BaseEmbedder:
        provider = provider.lower()
        if provider not in cls._registry:
            raise ValueError(
                f"Unknown embedding provider: '{provider}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[provider]()

    @classmethod
    def available_providers(cls) -> List[str]:
        return list(cls._registry.keys())
