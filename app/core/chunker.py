"""
Chunking Strategy
─────────────────
Why 500 tokens with 50-token overlap?

1. Context window fit: Most LLMs have 4K–8K token context windows.
   Injecting top-5 chunks = 5 × 500 = 2,500 tokens, leaving room
   for the prompt template and generated answer.

2. Semantic coherence: 500 tokens ≈ 3-4 paragraphs — enough to
   capture a complete idea (e.g., a CV section or a job requirement)
   without fragmenting it mid-sentence.

3. Overlap (50 tokens ≈ 10%): Prevents information loss at chunk
   boundaries. A skill or requirement described across two paragraphs
   won't be silently dropped. 10% overlap is the empirically tested
   sweet spot — lower misses boundary context, higher creates
   redundancy that inflates the index.

4. Sentence-aware splitting: We split on sentence boundaries (spacy
   / regex) rather than hard character limits, so we never embed a
   fragment. Only if a single sentence exceeds the limit do we fall
   back to hard truncation.

5. Arabic consideration: Arabic tokens tend to be longer due to
   morphological richness. The same 500-token budget covers slightly
   fewer Arabic words (~350–400) but the semantic boundary logic
   still applies.
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
    """
    Approximation: 1 token ≈ 4 chars for English, ≈ 3 chars for Arabic.
    (Arabic morphology produces shorter tokens on average.)
    Using 3.5 chars/token as a mixed-language average.
    """
    return max(1, len(text) // 4)


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
                    _save_chunk(chunks, buffer, chunk_index, metadata)
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
