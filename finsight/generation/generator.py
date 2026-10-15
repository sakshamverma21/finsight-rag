"""Grounded answer generation: retrieve -> answer citing only the sources.

Every claim is tagged to a numbered source that maps back to a chunk-id, so the
answer is auditable. Citation validity (do the [n] markers point at real sources?)
is computed here as a first faithfulness signal.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from finsight.config import LLM_MODEL
from finsight.llm import chat
from finsight.retrieval.retriever import Retriever

_CITATION = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "You are a financial-analyst assistant answering questions about SEC filings. "
    "Answer ONLY using the numbered SOURCES provided. After each claim, cite the "
    "supporting source(s) inline like [1] or [2][3]. Do not use outside knowledge. "
    "If the sources do not contain the answer, reply exactly: "
    "'The provided filings do not contain this information.' Be concise and factual."
)


@dataclass
class Source:
    n: int
    chunk_id: str
    ticker: str
    section: str
    text: str
    fiscal_year: int | None = None
    filing_date: str | None = None


@dataclass
class Answer:
    query: str
    text: str
    sources: list[Source]
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def citation_valid(self) -> bool:
        return self.cited and not self.invalid_citations


def _format_sources(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        text = " ".join(s.text.split())
        blocks.append(f"[{s.n}] ({s.ticker} {s.section})\n{text}")
    return "\n\n".join(blocks)


class Generator:
    def __init__(self, retriever: Retriever, model: str = LLM_MODEL):
        self.retriever = retriever
        self.model = model

    def answer(self, query: str, top_k: int = 5, mode: str | None = None, prefilter: bool = True) -> Answer:
        # Use the best available retrieval: reranked (Config C) if a reranker is
        # attached, else hybrid. Callers can still force a mode explicitly.
        if mode is None:
            mode = "rerank" if self.retriever.reranker is not None else "hybrid"
        chunks = self.retriever.retrieve(query, top_k=top_k, mode=mode, prefilter=prefilter)
        sources = [
            Source(
                n=i + 1,
                chunk_id=c["chunk_id"],
                ticker=c["metadata"]["ticker"],
                section=c["metadata"]["section"],
                text=c["text"],
                fiscal_year=c["metadata"].get("fiscal_year"),
                filing_date=c["metadata"].get("filing_date"),
            )
            for i, c in enumerate(chunks)
        ]

        t0 = time.perf_counter()
        text = chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"QUESTION: {query}\n\nSOURCES:\n{_format_sources(sources)}"},
            ],
            model=self.model,
        )
        latency = (time.perf_counter() - t0) * 1000

        cited = sorted({int(n) for n in _CITATION.findall(text)})
        valid_ns = {s.n for s in sources}
        invalid = [n for n in cited if n not in valid_ns]

        return Answer(
            query=query,
            text=text,
            sources=sources,
            cited=[n for n in cited if n in valid_ns],
            invalid_citations=invalid,
            latency_ms=latency,
        )
