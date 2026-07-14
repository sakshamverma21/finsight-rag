"""Ask a question against the pilot corpus with grounded citations.

Run:  uv run python scripts/ask.py "your question here"
      uv run python scripts/ask.py            # runs a couple of demo questions
"""

from __future__ import annotations

import sys

from finsight.config import CHUNKS_PATH
from finsight.generation.generator import Generator
from finsight.indexing.embedder import Embedder
from finsight.retrieval.index import ChunkIndex
from finsight.retrieval.reranker import Reranker
from finsight.retrieval.retriever import Retriever

DEMO_QUESTIONS = [
    "What are the main supply chain and manufacturing risks the company discloses?",
    "How did Microsoft's gross margin change and what drove it?",
]


def print_answer(ans) -> None:
    print(f"\nQ: {ans.query}")
    print(f"\nA: {ans.text}\n")
    flag = "OK" if ans.citation_valid else "CHECK"
    print(f"[citations {flag}] cited sources: {ans.cited}"
          + (f"  invalid: {ans.invalid_citations}" if ans.invalid_citations else "")
          + f"  | {ans.latency_ms:.0f} ms")
    print("Sources:")
    for s in ans.sources:
        mark = "*" if s.n in ans.cited else " "
        print(f"  {mark}[{s.n}] {s.ticker} {s.section:<24} {s.chunk_id}")
    print("-" * 70)


def main() -> None:
    embedder = Embedder()
    index = ChunkIndex.build(CHUNKS_PATH, embedder)
    gen = Generator(Retriever(index, embedder, reranker=Reranker()))

    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_QUESTIONS
    for q in questions:
        print_answer(gen.answer(q, top_k=5))


if __name__ == "__main__":
    main()
