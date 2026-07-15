"""Route a query: NUMERIC -> text-to-SQL over facts; NARRATIVE -> retrieval + generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from finsight.agent.multihop import answer_multihop
from finsight.agent.router import route as route_query
from finsight.generation.generator import Generator
from finsight.retrieval.metadata_filter import extract_filters
from finsight.structured.text_to_sql import answer_numeric


@dataclass
class RoutedAnswer:
    query: str
    route: str          # "NUMERIC" | "NARRATIVE"
    routed_by_llm: bool  # False when the rule-based fast path decided (no router call)
    payload: Any        # NumericAnswer or Answer


class Orchestrator:
    def __init__(self, generator: Generator):
        self.generator = generator

    def answer(self, query: str) -> RoutedAnswer:
        route, used_llm = route_query(query)
        if route == "NUMERIC":
            filters = extract_filters(query)
            # Reuse the embedder already loaded for retrieval: schema linking must not
            # lazily construct a second copy of the model on the first request.
            payload = answer_numeric(
                query, tickers=filters["tickers"], fiscal_years=filters["fiscal_years"],
                embedder=self.generator.retriever.embedder)
        elif route == "MULTI_HOP":
            payload = answer_multihop(query, self.generator.retriever)
        else:
            payload = self.generator.answer(query)
        return RoutedAnswer(query, route, used_llm, payload)
