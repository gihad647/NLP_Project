"""
LLM Factory Pattern (Bonus 1)
──────────────────────────────
A unified interface that lets the system switch between:
  • Gemini (Google AI)
  • OpenAI (GPT-4o, GPT-3.5)
  • Ollama (local Mistral / LLaMA)

With a single config change (LLM_PROVIDER env var), no code changes needed.
"""
from abc import ABC, abstractmethod
from typing import List, Dict
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Abstract base
# ────────────────────────────────────────────────────────────
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, context_chunks: List[str]) -> Dict:
        """
        Returns dict with keys: answer (str), tokens_used (int | None)
        """


# ────────────────────────────────────────────────────────────
# Gemini implementation
# ────────────────────────────────────────────────────────────
class GeminiLLM(BaseLLM):
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate(self, prompt: str, context_chunks: List[str]) -> Dict:
        context = "\n\n---\n\n".join(context_chunks)
        full_prompt = _build_prompt(prompt, context)
        response = self.model.generate_content(full_prompt)
        return {
            "answer": response.text,
            "tokens_used": None,  # Gemini SDK doesn't expose token count easily
        }


# ────────────────────────────────────────────────────────────
# OpenAI implementation
# ────────────────────────────────────────────────────────────
class OpenAILLM(BaseLLM):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate(self, prompt: str, context_chunks: List[str]) -> Dict:
        context = "\n\n---\n\n".join(context_chunks)
        full_prompt = _build_prompt(prompt, context)
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based solely on the provided context."},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.2,
        )
        return {
            "answer": resp.choices[0].message.content,
            "tokens_used": resp.usage.total_tokens,
        }


# ────────────────────────────────────────────────────────────
# Ollama (local) implementation
# ────────────────────────────────────────────────────────────
class OllamaLLM(BaseLLM):
    def __init__(self):
        import ollama
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        self.model = settings.OLLAMA_MODEL

    def generate(self, prompt: str, context_chunks: List[str]) -> Dict:
        context = "\n\n---\n\n".join(context_chunks)
        full_prompt = _build_prompt(prompt, context)
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
        )
        return {
            "answer": resp["message"]["content"],
            "tokens_used": None,
        }


# ────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────
class LLMFactory:
    _registry: Dict[str, type] = {
        "gemini": GeminiLLM,
        "openai": OpenAILLM,
        "ollama": OllamaLLM,
    }

    @classmethod
    def create(cls, provider: str | None = None) -> BaseLLM:
        provider = (provider or settings.LLM_PROVIDER).lower()
        if provider not in cls._registry:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                f"Choose from: {list(cls._registry.keys())}"
            )
        logger.info(f"LLMFactory: creating provider '{provider}'")
        return cls._registry[provider]()

    @classmethod
    def available_providers(cls) -> List[str]:
        return list(cls._registry.keys())


# ────────────────────────────────────────────────────────────
# Shared prompt builder
# ────────────────────────────────────────────────────────────
def _build_prompt(query: str, context: str) -> str:
    return f"""You are an expert assistant. Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so clearly. Do NOT hallucinate.

=== CONTEXT ===
{context}

=== QUESTION ===
{query}

=== ANSWER ==="""
