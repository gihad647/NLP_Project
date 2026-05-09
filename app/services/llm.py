from abc import ABC, abstractmethod
from typing import List, Dict
from app.core.config import settings


# ─────────────────────────────────────────────
#  Abstract Base
# ─────────────────────────────────────────────

class BaseLLM(ABC):
    """Abstract interface for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, context_chunks: List[str]) -> str:
        """Generate an answer given a prompt and retrieved context."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model identifier."""

    def _build_rag_prompt(self, query: str, context_chunks: List[str]) -> str:
        """Build the RAG prompt by injecting context chunks."""
        context = "\n\n---\n\n".join(context_chunks)
        return f"""You are an expert HR assistant and resume screener.
Use ONLY the context provided below to answer the question.
If the answer is not in the context, say "I could not find relevant information."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""


# ─────────────────────────────────────────────
#  Concrete Implementations
# ─────────────────────────────────────────────

class OllamaLLM(BaseLLM):
    """Uses a locally running Ollama model (e.g., Mistral, LLaMA3)."""

    def __init__(self, model: str = settings.LLM_MODEL):
        import requests
        self._model = model
        self._base_url = settings.OLLAMA_BASE_URL
        self._requests = requests

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, context_chunks: List[str]) -> str:
        full_prompt = self._build_rag_prompt(prompt, context_chunks)
        resp = self._requests.post(
            f"{self._base_url}/api/generate",
            json={"model": self._model, "prompt": full_prompt, "stream": False},
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


class OpenAILLM(BaseLLM):
    """Uses OpenAI Chat Completions API (GPT-4o, GPT-3.5, etc.)."""

    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        self._model = model
        self._client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, context_chunks: List[str]) -> str:
        full_prompt = self._build_rag_prompt(prompt, context_chunks)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


class GeminiLLM(BaseLLM):
    """Uses Google Gemini API."""

    def __init__(self, model: str = "gemini-1.5-flash"):
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self._model_name = model
        self._model = genai.GenerativeModel(model)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, context_chunks: List[str]) -> str:
        full_prompt = self._build_rag_prompt(prompt, context_chunks)
        response = self._model.generate_content(full_prompt)
        return response.text.strip()


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────

class LLMFactory:
    """
    Factory Design Pattern for LLM providers.
    Switch providers via LLM_PROVIDER env variable.
    """

    _registry = {
        "ollama": OllamaLLM,
        "openai": OpenAILLM,
        "gemini": GeminiLLM,
    }

    @classmethod
    def create(cls, provider: str = settings.LLM_PROVIDER) -> BaseLLM:
        provider = provider.lower()
        if provider not in cls._registry:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[provider]()

    @classmethod
    def available_providers(cls) -> List[str]:
        return list(cls._registry.keys())