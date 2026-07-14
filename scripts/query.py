"""Full routed pipeline demo: NUMERIC -> text-to-SQL, NARRATIVE -> retrieval+generation.

Run:  uv run python scripts/query.py "your question"
      uv run python scripts/query.py         # demo set
"""

from __future__ import annotations

import sys

from finsight.agent.orchestrator import Orchestrator
from finsight.config import CHUNKS_PATH
from finsight.generation.generator import Generator
from finsight.indexing.embedder import Embedder
from finsight.retrieval.index import ChunkIndex
from finsight.retrieval.reranker import Reranker
from finsight.retrieval.retriever import Retriever

DEMO = [
    "What was Apple's total revenue for fiscal year 2023?",
    "How much did Microsoft spend on research and development in FY2024?",
    "What was Google's net income in 2024?",
    "What supply chain risks does Apple disclose?",  # narrative
]


def print_routed(r) -> None:
    print("=" * 72)
    print(f"Q: {r.query}")
    print(f"ROUTE -> {r.route}")
    p = r.payload
    if r.route == "NUMERIC":
        print(f"SQL: {p.sql}")
        if p.error:
            print(f"ERROR: {p.error}")
        print("Answer:")
        print(p.text)
        print(f"({p.latency_ms:.0f} ms)")
    else:
        print(f"A: {p.text[:500]}")
        flag = "OK" if p.citation_valid else "CHECK"
        print(f"[citations {flag}] {p.cited}  ({p.latency_ms:.0f} ms)")


def main() -> None:
    embedder = Embedder()
    index = ChunkIndex.build(CHUNKS_PATH, embedder)
    orch = Orchestrator(Generator(Retriever(index, embedder, reranker=Reranker())))

    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO
    for q in questions:
        print_routed(orch.answer(q))


if __name__ == "__main__":
    main()
