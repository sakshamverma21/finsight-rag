"""MULTI_HOP path: agentic iterative retrieval for questions that need several hops.

Single-pass retrieval fails when one question spans several sub-topics or companies
(e.g. "how do Apple's and Microsoft's supply-chain risks differ?"): a single query
vector cannot sit close to every piece of evidence at once, so the reranker's top-k
is dominated by one hop and starves the others. The 2026 pattern is *iterative with
recovery*:

  decompose -> retrieve per sub-question -> check sufficiency -> re-retrieve the gap
  -> synthesize a grounded, cited answer.

The sub-question decomposition also becomes a visible reasoning trace (`steps`), and
the whole path is measured against single-pass narrative on a multi-hop golden set
(scripts/run_multihop_eval.py) so its win is a number, not a claim.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from finsight.config import LLM_MODEL
from finsight.generation.generator import (
    _CITATION, _format_sources, SYSTEM_PROMPT, Source,
)
from finsight.llm import chat
from finsight.retrieval.retriever import Retriever

_DECOMPOSE = """Break this financial question into the 2-3 focused sub-questions that must
each be answered to answer it fully. Each sub-question should target ONE company and ONE
topic, so it can be retrieved on its own. If the question is already single-hop, return it
unchanged as the only sub-question.

Question: {query}

Return ONLY JSON: {{"subquestions": ["...", "..."]}}"""

_SUFFICIENCY = """You are judging whether retrieved evidence is enough to fully answer a
question about SEC filings.

Question: {query}

Evidence (numbered snippets):
{evidence}

If the evidence is missing something needed to answer completely, give ONE focused
follow-up search query for the single most important missing piece; otherwise none.
Return ONLY JSON: {{"sufficient": true|false, "missing_query": "<query or null>"}}"""


@dataclass
class MultiHopAnswer:
    query: str
    text: str
    sources: list  # list[Source]
    cited: list = field(default_factory=list)
    invalid_citations: list = field(default_factory=list)
    subquestions: list = field(default_factory=list)
    steps: list = field(default_factory=list)   # [{"q": str, "n_sources": int, "recovery"?: bool}]
    latency_ms: float = 0.0

    @property
    def citation_valid(self) -> bool:
        return bool(self.cited) and not self.invalid_citations


def _decompose(query: str, model: str) -> list[str]:
    try:
        text = chat([{"role": "user", "content": _DECOMPOSE.format(query=query)}],
                    model=model, json_mode=True)
        subs = json.loads(text).get("subquestions") or []
        subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
        return subs[:3] or [query]
    except Exception:
        return [query]


def _retrieve_pool(retriever: Retriever, subqs: list[str], per_k: int):
    """Retrieve per sub-question; dedupe by chunk_id, preserving first-seen order."""
    seen: dict[str, dict] = {}
    steps: list[dict] = []
    for sq in subqs:
        chunks = retriever.retrieve(sq, top_k=per_k, mode="rerank", prefilter=True)
        added = 0
        for c in chunks:
            if c["chunk_id"] not in seen:
                seen[c["chunk_id"]] = c
                added += 1
        steps.append({"q": sq, "n_sources": added})
    return list(seen.values()), steps


def _evidence_digest(chunks: list[dict], limit: int) -> str:
    return "\n".join(
        f"[{i+1}] ({c['metadata']['ticker']} {c['metadata']['section']}) "
        + " ".join(c["text"].split())[:240]
        for i, c in enumerate(chunks[:limit])
    )


def answer_multihop(query: str, retriever: Retriever, model: str = LLM_MODEL,
                    per_k: int = 4, max_sources: int = 8) -> MultiHopAnswer:
    t0 = time.perf_counter()
    subqs = _decompose(query, model)
    chunks, steps = _retrieve_pool(retriever, subqs, per_k)

    # Sufficiency gate + one recovery retrieval (the "iterative with recovery" step).
    try:
        chk = json.loads(chat(
            [{"role": "user", "content": _SUFFICIENCY.format(
                query=query, evidence=_evidence_digest(chunks, max_sources))}],
            model=model, json_mode=True))
    except Exception:
        chk = {"sufficient": True}
    if not chk.get("sufficient") and chk.get("missing_query"):
        mq = str(chk["missing_query"]).strip()
        seen = {c["chunk_id"] for c in chunks}
        added = 0
        for c in retriever.retrieve(mq, top_k=per_k, mode="rerank", prefilter=True):
            if c["chunk_id"] not in seen:
                chunks.append(c)
                seen.add(c["chunk_id"])
                added += 1
        steps.append({"q": mq, "n_sources": added, "recovery": True})

    chunks = chunks[:max_sources]
    sources = [
        Source(n=i + 1, chunk_id=c["chunk_id"], ticker=c["metadata"]["ticker"],
               section=c["metadata"]["section"], text=c["text"],
               fiscal_year=c["metadata"].get("fiscal_year"),
               filing_date=c["metadata"].get("filing_date"))
        for i, c in enumerate(chunks)
    ]

    text = chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION: {query}\n\nSOURCES:\n{_format_sources(sources)}"},
        ],
        model=model,
    )

    cited = sorted({int(n) for n in _CITATION.findall(text)})
    valid = {s.n for s in sources}
    invalid = [n for n in cited if n not in valid]
    return MultiHopAnswer(
        query=query, text=text, sources=sources,
        cited=[n for n in cited if n in valid], invalid_citations=invalid,
        subquestions=subqs, steps=steps, latency_ms=(time.perf_counter() - t0) * 1000,
    )
