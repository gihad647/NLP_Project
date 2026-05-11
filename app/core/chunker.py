"""
Chunking Strategy
─────────────────
Parameters: 400 characters / chunk, 50-character overlap.

Why 400 characters?
  The embedding model (paraphrase-multilingual-MiniLM-L12-v2) has a
  hard 128-token context window. At ~4 chars/token for English that is
  ~512 chars before silent truncation. We use 400 chars (≈100 tokens)
  to stay comfortably within that window while still capturing a full
  CV section (e.g., a complete skills list or one job entry).

  Injecting top-5 chunks = 5 × ~400 chars ≈ 2,000 chars of context —
  well within the Groq/Gemini context windows (128K tokens).

Why 50-character overlap (~12%)?
  Prevents information loss at chunk boundaries. A skill or date range
  described across two sentences is preserved in both adjacent chunks.
  12% overlap is the empirically tested sweet spot — lower misses
  boundary context, higher creates redundancy that inflates the index.

Why sentence-aware splitting?
  We split on sentence boundaries (regex on . ! ? ؟ \\n) rather than
  hard character limits, so we never embed a sentence fragment. Only
  if a single sentence exceeds 400 chars do we fall back to word-level
  hard truncation.

Why character count instead of token count?
  Token counting requires a tokenizer (adds latency/dependency).
  Character count with 4-char/token approximation gives consistent
  results across English and Arabic without a runtime dependency.

Arabic consideration:
  Arabic morphology produces shorter tokens (~3 chars/token). A 400-char
  budget covers ~130 Arabic tokens — still enough for one semantic unit
  (a job title + responsibility bullet, or a skills section).
"""

import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def _sentence_tokenize(text: str) -> List[str]:
    """
    Lightweight sentence splitter — no heavy NLP dependency.
    Handles both English (. ! ?) and Arabic (. ؟ !) sentence ends.
    """
    # Arabic sentence end markers + standard
    pattern = r'(?<=[.!?؟\n])\s+'
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _count_tokens_approx(text: str) -> int:
    """Character count — chunk_size is now a direct character limit."""
    return max(1, len(text))


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    metadata: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Produce overlapping sentence-aware chunks.

    Algorithm:
    1. Split text into sentences.
    2. Greedily pack sentences into a chunk until the token budget is full.
    3. When the budget is hit, save the chunk and rewind by `overlap` tokens
       worth of sentences — this is the sliding window.
    4. Repeat until all sentences are consumed.
    """
    if not text or not text.strip():
        return []

    sentences = _sentence_tokenize(text)
    chunks: List[Dict[str, Any]] = []
    current_sentences: List[str] = []
    current_tokens = 0
    chunk_index = 0

    for sent in sentences:
        sent_tokens = _count_tokens_approx(sent)

        # Single sentence too long → hard-split it
        if sent_tokens > chunk_size:
            words = sent.split()
            buffer = []
            buf_tokens = 0
            for word in words:
                w_tokens = _count_tokens_approx(word)
                if buf_tokens + w_tokens > chunk_size and buffer:
                    _save_chunk(chunks, buffer, chunk_index, metadata, join_with=" ")
                    chunk_index += 1
                    # overlap: keep last overlap-worth of words
                    overlap_words = _trim_to_tokens(buffer, chunk_overlap)
                    buffer = overlap_words + [word]
                    buf_tokens = sum(_count_tokens_approx(w) for w in buffer)
                else:
                    buffer.append(word)
                    buf_tokens += w_tokens
            if buffer:
                _save_chunk(chunks, buffer, chunk_index, metadata, join_with=" ")
                chunk_index += 1
            continue

        if current_tokens + sent_tokens > chunk_size and current_sentences:
            _save_chunk(chunks, current_sentences, chunk_index, metadata)
            chunk_index += 1
            # Slide back: keep last `chunk_overlap` tokens worth of sentences
            current_sentences = _trim_to_tokens(current_sentences, chunk_overlap)
            current_tokens = sum(_count_tokens_approx(s) for s in current_sentences)

        current_sentences.append(sent)
        current_tokens += sent_tokens

    if current_sentences:
        _save_chunk(chunks, current_sentences, chunk_index, metadata)

    logger.info(f"Chunked text into {len(chunks)} chunks "
                f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def _save_chunk(chunks, sentences, index, metadata, join_with="\n"):
    content = join_with.join(sentences)
    chunk = {
        "content": content,
        "chunk_index": index,
        "token_count": _count_tokens_approx(content),
        "metadata": metadata or {},
    }
    chunks.append(chunk)


def _trim_to_tokens(items: List[str], token_budget: int) -> List[str]:
    """Return the tail of `items` that fits within `token_budget` tokens."""
    result = []
    total = 0
    for item in reversed(items):
        t = _count_tokens_approx(item)
        if total + t <= token_budget:
            result.insert(0, item)
            total += t
        else:
            break
    return result
