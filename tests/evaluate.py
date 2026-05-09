"""
Phase 4: Evaluation & Error Analysis
──────────────────────────────────────
Run this script after docker-compose up to test the RAG system.
It sends 10 queries (7 normal + 3 edge cases that expose failures)
and prints a structured report.
"""
import httpx
import json
from dataclasses import dataclass, field
from typing import List, Optional
import time

BASE_URL = "http://localhost:8080/api/v1"


@dataclass
class TestCase:
    query: str
    expected_keywords: List[str]
    is_edge_case: bool = False
    edge_case_type: Optional[str] = None
    language: str = "en"


TEST_CASES = [
    # ── Normal queries ────────────────────────────────────
    TestCase(
        query="Find a senior Python developer with FastAPI experience",
        expected_keywords=["python", "fastapi", "senior"],
    ),
    TestCase(
        query="Who has experience in machine learning and data science?",
        expected_keywords=["machine learning", "data science"],
    ),
    TestCase(
        query="List candidates with Docker and Kubernetes skills",
        expected_keywords=["docker", "kubernetes"],
    ),
    TestCase(
        query="Find a frontend developer experienced in React",
        expected_keywords=["react", "frontend"],
    ),
    TestCase(
        query="Who has a Computer Science degree?",
        expected_keywords=["computer science", "bachelor", "degree"],
    ),
    TestCase(
        query="Find candidates with more than 5 years of experience",
        expected_keywords=["experience", "years"],
    ),
    TestCase(
        query="Who speaks Arabic and English?",
        expected_keywords=["arabic", "english"],
    ),

    # ── Edge cases (expected failures) ───────────────────
    TestCase(
        query="Find a quantum computing expert with 10 years in blockchain",
        expected_keywords=["quantum", "blockchain"],
        is_edge_case=True,
        edge_case_type="Out-of-distribution query",
        # FAILURE REASON: The CV corpus likely has no quantum computing candidates.
        # The retriever will return the highest-scoring chunks by cosine similarity
        # even if they are irrelevant (no hard threshold by default).
        # The LLM may then hallucinate a candidate or give a fabricated answer
        # rather than clearly saying "not found".
    ),
    TestCase(
        query="خبرة في تطوير الويب وقواعد البيانات",   # Arabic: "Experience in web dev and databases"
        expected_keywords=["web", "database", "تطوير"],
        is_edge_case=True,
        edge_case_type="Arabic query on potentially English-only corpus",
        language="ar",
        # FAILURE REASON: If all CVs were ingested in English, the multilingual
        # embedding model (paraphrase-multilingual-MiniLM-L12-v2) creates a
        # cross-lingual embedding but the semantic space may drift due to the
        # domain mismatch (CV language ≠ query language). Retrieval scores will
        # be lower and the LLM may fail to map retrieved English context to the
        # Arabic question intent.
    ),
    TestCase(
        query="a",  # Nonsense / too-short query
        expected_keywords=[],
        is_edge_case=True,
        edge_case_type="Degenerate/ambiguous query",
        # FAILURE REASON: Single-character query produces a near-zero-information
        # embedding. The retriever returns random chunks (the ones whose embedding
        # happens to be closest to the letter 'a' in embedding space).
        # This exposes the lack of a minimum query length / intent detection guard.
    ),
]


def run_query(query: str) -> dict:
    resp = httpx.post(
        f"{BASE_URL}/query",
        json={"query": query, "top_k": 5},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def evaluate():
    print("=" * 70)
    print("RAG SYSTEM EVALUATION REPORT")
    print("=" * 70)

    results = []
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {'[EDGE CASE] ' if tc.is_edge_case else ''}Query: {tc.query[:60]}")
        if tc.is_edge_case:
            print(f"  Type: {tc.edge_case_type}")

        start = time.time()
        try:
            result = run_query(tc.query)
            elapsed = time.time() - start

            answer = result["answer"]
            chunks = result["retrieved_chunks"]
            top_score = max((c["score"] for c in chunks), default=0)

            # Check if any expected keyword appears in the answer (case-insensitive)
            hits = [kw for kw in tc.expected_keywords if kw.lower() in answer.lower()]
            keyword_hit_rate = len(hits) / len(tc.expected_keywords) if tc.expected_keywords else 1.0

            print(f"  ✓ Answer ({elapsed:.1f}s): {answer[:120]}...")
            print(f"  Top chunk score: {top_score:.3f} | Keyword hits: {len(hits)}/{len(tc.expected_keywords)}")

            results.append({
                "query": tc.query,
                "is_edge_case": tc.is_edge_case,
                "edge_case_type": tc.edge_case_type,
                "answer_snippet": answer[:200],
                "top_score": top_score,
                "keyword_hit_rate": keyword_hit_rate,
                "elapsed_s": round(elapsed, 2),
                "status": "ok",
            })

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append({"query": tc.query, "status": "error", "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ok = [r for r in results if r.get("status") == "ok"]
    avg_score = sum(r["top_score"] for r in ok) / max(len(ok), 1)
    avg_kw = sum(r["keyword_hit_rate"] for r in ok) / max(len(ok), 1)
    print(f"Queries run:           {len(results)}")
    print(f"Successful:            {len(ok)}")
    print(f"Avg top chunk score:   {avg_score:.3f}")
    print(f"Avg keyword hit rate:  {avg_kw:.2%}")

    print("\nEDGE CASE ANALYSIS")
    print("-" * 70)
    for r in results:
        if r.get("is_edge_case"):
            tc = next(t for t in TEST_CASES if t.query == r["query"])
            print(f"\n• Query: {r['query'][:60]}")
            print(f"  Failure type: {r.get('edge_case_type')}")
            print(f"  Root cause: See comments in evaluate.py for this test case.")
            print(f"  Mitigation: "
                  + {
                      "Out-of-distribution query":      "Add a retrieval confidence threshold; return 'not found' if top score < 0.4.",
                      "Arabic query on potentially English-only corpus": "Ingest Arabic CVs alongside English ones; or use a translate-then-retrieve pipeline.",
                      "Degenerate/ambiguous query":     "Add minimum query length validation (>= 3 words) in the API layer.",
                  }.get(r.get("edge_case_type", ""), "Review retrieval pipeline."))


if __name__ == "__main__":
    evaluate()
