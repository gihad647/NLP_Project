"""
Phase 4: Evaluation & Edge-Case Analysis
─────────────────────────────────────────
Run against a live system:
    python tests/evaluate.py

Uses only stdlib — no extra dependencies needed.
Prints a structured report showing retrieval accuracy and failure diagnosis.
"""
import json
import sys
import urllib.request
import urllib.error
import time

BASE_URL = "http://localhost:8080"


# ── HTTP helper ───────────────────────────────────────────────────────────────
def _query(text: str, top_k: int = 5) -> dict:
    payload = json.dumps({"query": text, "top_k": top_k}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _top_source(result: dict) -> str:
    if not result["retrieved_chunks"]:
        return "NONE"
    src = result["retrieved_chunks"][0]["source"]
    return src.split("/")[-1].replace(".pdf", "").replace(".html", "")


def _top_score(result: dict) -> float:
    if not result["retrieved_chunks"]:
        return 0.0
    return result["retrieved_chunks"][0]["score"]


def _rank_of(result: dict, keyword: str) -> int:
    """Return 1-based rank of first chunk whose source contains keyword, or 99."""
    for i, c in enumerate(result["retrieved_chunks"], 1):
        if keyword in c["source"]:
            return i
    return 99


# ── Test definitions ──────────────────────────────────────────────────────────
# (description, query_text, expected_source_keyword, is_edge_case, edge_type, mitigation)
TEST_CASES = [
    # ── Normal / should pass ──────────────────────────────────────────────────
    (
        "Senior Python / FastAPI developer",
        "Find a senior Python developer with FastAPI experience",
        "john_smith", False, None, None,
    ),
    (
        "ML engineer with NLP and RAG experience",
        "Who has experience with NLP, RAG pipelines and sentence transformers?",
        "sarah_chen", False, None, None,
    ),
    (
        "DevOps engineer — Kubernetes & Terraform",
        "Find a DevOps engineer experienced with Kubernetes and Terraform",
        "ahmed_hassan", False, None, None,
    ),
    (
        "React / TypeScript frontend developer",
        "Find a frontend developer with React and TypeScript skills",
        "maria_garcia", False, None, None,
    ),
    (
        "Arabic query — web dev + databases",
        "ابحث عن مطور ويب بخبرة في قواعد البيانات",
        "omar_arabic", False, None, None,
    ),

    # ── Edge cases — expected failures / degraded results ─────────────────────
    (
        "EDGE: out-of-domain — quantum / blockchain",
        "quantum computing expert with 10 years of blockchain experience",
        None,        # no correct answer exists in corpus
        True,
        "Out-of-distribution query",
        "Add a retrieval confidence threshold: if top score < 0.35, return "
        "'No suitable candidate found' instead of hallucinating one.",
    ),
    (
        "EDGE: synonym gap — 'AI Engineer' vs 'ML Engineer'",
        "Find a senior AI Engineer",
        "sarah_chen",   # correct but embedding gap may rank her low
        True,
        "Vocabulary / synonym mismatch",
        "Use query expansion: map 'AI Engineer' -> 'machine learning engineer, "
        "ML researcher, data scientist' before embedding.",
    ),
    (
        "EDGE: degenerate single-character query",
        "a",
        None,
        True,
        "Degenerate query — near-zero information embedding",
        "Enforce minimum query length (>= 3 words) at the API layer; "
        "return HTTP 400 for trivial queries.",
    ),
    (
        "EDGE: Arabic query on English-dominant corpus",
        "من لديه خبرة Kubernetes والبنية التحتية السحابية",
        "ahmed_hassan",
        True,
        "Cross-lingual retrieval — Arabic query vs English CV",
        "Ingest bilingual CVs; or pre-translate the query to English with "
        "a lightweight model (e.g. Helsinki-NLP/opus-mt-ar-en) before embedding.",
    ),
]


# ── Runner ────────────────────────────────────────────────────────────────────
def run() -> int:
    sep = "=" * 78
    print(f"\n{sep}")
    print("RAG SYSTEM — RETRIEVAL EVALUATION REPORT")
    print(sep)

    results = []
    for desc, q, expected, is_edge, edge_type, mitigation in TEST_CASES:
        tag = "[EDGE]" if is_edge else "[NORM]"
        print(f"\n{tag} {desc}")
        print(f"  Query: {q[:70]}")

        t0 = time.time()
        try:
            result = _query(q)
            elapsed = time.time() - t0
            actual_src = _top_source(result)
            score = _top_score(result)
            answer_snip = result.get("answer", "")[:120].replace("\n", " ")

            if expected is None:
                # Info-only — we just want to observe the (wrong) output
                status = "INFO"
                rank = "-"
            else:
                rank = _rank_of(result, expected)
                status = "PASS" if rank == 1 else "FAIL"

            print(f"  Top source : {actual_src}  (score {score:.3f})")
            if expected:
                print(f"  Expected   : {expected}  (ranked #{rank})")
            print(f"  Answer     : {answer_snip}...")
            print(f"  Time       : {elapsed:.1f}s   Status: [{status}]")

            if status == "FAIL" or status == "INFO":
                if edge_type:
                    print(f"  Failure    : {edge_type}")
                    print(f"  Mitigation : {mitigation}")

            results.append({"desc": desc, "status": status, "score": score,
                            "rank": rank, "is_edge": is_edge})

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ERROR: {exc}")
            results.append({"desc": desc, "status": "ERROR", "is_edge": is_edge})

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    norm  = [r for r in results if not r["is_edge"]]
    edges = [r for r in results if r["is_edge"]]
    norm_pass = sum(1 for r in norm if r["status"] == "PASS")
    print(f"Normal queries : {norm_pass}/{len(norm)} passed")
    print(f"Edge cases     : {len(edges)} observed (failures are expected)")
    if norm:
        avg_score = sum(r.get("score", 0) for r in norm) / len(norm)
        print(f"Avg top score  : {avg_score:.3f}")

    print(f"\nEdge-case breakdown:")
    for i, (r, tc) in enumerate(
        [(r, tc) for r, tc in zip(results, TEST_CASES) if r["is_edge"]], 1
    ):
        print(f"  {i}. {r['desc']}")
        print(f"     Status: {r['status']}  |  Score: {r.get('score', 0):.3f}")

    failures = sum(1 for r in norm if r["status"] != "PASS")
    print(f"\n{'[ALL PASS]' if failures == 0 else f'[{failures} NORMAL QUERIES FAILED]'}")
    print(sep + "\n")
    return failures


if __name__ == "__main__":
    try:
        urllib.request.urlopen(f"{BASE_URL}/health", timeout=5)
    except Exception:
        print(f"ERROR: Cannot reach {BASE_URL}")
        print("       Start with:  docker compose up -d")
        sys.exit(1)
    sys.exit(run())
