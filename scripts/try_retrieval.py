"""Build the in-memory index from pilot chunks and compare dense vs hybrid.

Run:  uv run python scripts/try_retrieval.py
"""

from __future__ import annotations

import time

from finsight.config import CHUNKS_PATH
from finsight.indexing.embedder import Embedder
from finsight.retrieval.index import ChunkIndex
from finsight.retrieval.retriever import Retriever

QUERIES = [
    "What are the main risks related to supply chain and manufacturing in China?",
    "How did services revenue perform and what drove gross margin?",
    "competition from artificial intelligence and cloud computing rivals",
]


def show(label: str, results: list[dict]) -> None:
    print(f"  [{label}]")
    for r in results:
        m = r["metadata"]
        snippet = " ".join(r["text"].split())[:110]
        print(f"    - {m['ticker']} {m['section']:<22} {r['chunk_id']}")
        print(f"      {snippet}...")


def main() -> None:
    jsonl = CHUNKS_PATH
    print("Loading embedder (first run downloads bge-small ~130MB)...")
    embedder = Embedder()
    t0 = time.perf_counter()
    index = ChunkIndex.build(jsonl, embedder)
    print(f"Indexed {len(index)} child chunks (dim={embedder.dim}) in {time.perf_counter()-t0:.1f}s\n")

    retriever = Retriever(index, embedder)
    for q in QUERIES:
        print(f"Q: {q}")
        show("dense ", retriever.dense(q, top_k=3))
        show("hybrid", retriever.hybrid(q, top_k=3))
        print()


if __name__ == "__main__":
    main()
