"""Config A (dense) vs Config B (hybrid) retrieval eval on the pilot golden set.

Labels are resolved at run time: a chunk is relevant iff it contains the query's
'evidence' substring. Reports Recall@k, Hit@k, MRR, and per-query latency.

Run:  uv run python scripts/run_retrieval_eval.py
"""

from __future__ import annotations

import argparse
import json
import re
import time

from finsight.config import PROJECT_ROOT, CHUNKS_PATH
from finsight.eval.metrics import hit_at_k, mrr, recall_at_k
from finsight.indexing.embedder import Embedder
from finsight.retrieval.index import ChunkIndex
from finsight.retrieval.reranker import Reranker
from finsight.retrieval.retriever import Retriever

K = 5
GOLDEN_DIR = PROJECT_ROOT / "eval" / "golden"


def resolve_relevant(index: ChunkIndex, evidence: str) -> set[str]:
    """Relevant iff the chunk contains the evidence phrase on WORD BOUNDARIES.

    Naive substring matching is unsafe here: 'AWS' matches inside 'laws'/'draws',
    which silently mislabels ~300 chunks across every filer.
    """
    rx = re.compile(r"\b" + re.escape(evidence) + r"\b", re.I)
    return {c["chunk_id"] for c in index.chunks if rx.search(c["text"])}


def eval_mode(retriever: Retriever, index: ChunkIndex, queries: list[dict], mode: str, prefilter: bool = False) -> dict:
    recalls, hits, mrrs, lats, ceilings = [], [], [], [], []
    for q in queries:
        relevant = resolve_relevant(index, q["evidence"])
        if not relevant:
            print(f"  !! {q['id']}: evidence matched 0 chunks — skipping")
            continue
        # Recall@k is capped when a query has more than k relevant chunks.
        ceilings.append(min(K, len(relevant)) / len(relevant))
        t0 = time.perf_counter()
        results = retriever.retrieve(q["query"], top_k=K, mode=mode, prefilter=prefilter)
        lats.append((time.perf_counter() - t0) * 1000)
        ids = [r["chunk_id"] for r in results]
        recalls.append(recall_at_k(ids, relevant, K))
        hits.append(hit_at_k(ids, relevant, K))
        mrrs.append(mrr(ids, relevant))
    n = len(recalls)
    recall = sum(recalls) / n
    ceiling = sum(ceilings) / n
    return {
        "n": n,
        "recall@k": recall,
        "ceiling": ceiling,
        "recall_norm": recall / ceiling if ceiling else 0.0,
        "hit@k": sum(hits) / n,
        "mrr": sum(mrrs) / n,
        "p50_ms": sorted(lats)[len(lats) // 2],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="retrieval_10co.json",
                    help="golden set filename under eval/golden/")
    args = ap.parse_args()

    golden_path = GOLDEN_DIR / args.golden
    spec = json.loads(golden_path.read_text(encoding="utf-8"))
    queries = spec["queries"]
    print(f"Golden: {args.golden}")

    embedder = Embedder()
    index = ChunkIndex.build(CHUNKS_PATH, embedder)
    retriever = Retriever(index, embedder, reranker=Reranker())

    print(f"Golden set: {len(queries)} queries | corpus: {len(index)} chunks | k={K}\n")
    print(f"{'config':20} {'n':>3} {'Recall@5':>9} {'norm':>6} {'Hit@5':>7} {'MRR':>6} {'p50 ms':>7}")
    print("-" * 64)
    configs = [
        ("A: dense-only", "dense", False),
        ("B: hybrid (RRF)", "hybrid", False),
        ("C: hybrid+reranker", "rerank", False),
        ("C+ metadata filter", "rerank", True),
    ]
    ceiling = None
    for label, mode, prefilter in configs:
        m = eval_mode(retriever, index, queries, mode, prefilter=prefilter)
        ceiling = m["ceiling"]
        print(f"{label:20} {m['n']:>3} {m['recall@k']:>9.3f} {m['recall_norm']:>6.3f} "
              f"{m['hit@k']:>7.3f} {m['mrr']:>6.3f} {m['p50_ms']:>7.1f}")
    print("-" * 64)
    print(f"Recall@{K} ceiling for this golden set (multi-chunk evidence): {ceiling:.3f}")
    print("'norm' = Recall@5 as a fraction of that achievable ceiling.")


if __name__ == "__main__":
    main()
