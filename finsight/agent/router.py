"""Adaptive router: classify a query as NUMERIC vs NARRATIVE (LLM, JSON output).

NUMERIC  -> answerable by a reported figure -> text-to-SQL over the facts table.
NARRATIVE -> qualitative/explanatory        -> hybrid retrieval + generation.
(MULTI_HOP is added in M3.)
"""

from __future__ import annotations

import json
import re

from finsight.config import LLM_MODEL
from finsight.llm import chat
from finsight.retrieval.metadata_filter import extract_filters

# --- Rule-based fast path (no LLM call) ------------------------------------
# The router LLM call is ~880ms (mostly network RTT). Most real queries carry an
# unambiguous signal, so we classify those by keyword and only fall back to the
# LLM when signals conflict or are absent. Validated on 188 labeled golden queries.
_METRIC = re.compile(
    r"\b(revenue|revenues|net sales|sales|r&d|research and development|net income|"
    r"net loss|earnings per share|eps|gross profit|gross margin|operating income|"
    r"operating loss|operating expense|cost of revenue|cost of sales|total assets|"
    r"total liabilities|cash and cash equivalents|free cash flow|diluted|"
    r"capital expenditure|capex)\b", re.I,
)
_NARRATIVE_CUE = re.compile(
    r"\b(risk|risks|why|how does|how do|how is|describe|discuss|explain|strateg\w*|"
    r"competit\w*|compete|overview|business model|outlook|challenge\w*|opportunit\w*|"
    r"concern\w*|trend\w*|driver\w*|approach)\b", re.I,
)
# Multi-hop signal: an explicit comparison/contrast word, or two+ named companies. Note
# "how do" is deliberately NOT here (it matches single-company "how does Apple ...").
_COMPARE_CUE = re.compile(
    r"\b(compare|compared to|comparison|versus|vs\.?|differ|differs|difference|"
    r"differences|contrast|which of|relative to|both companies)\b", re.I,
)


def _is_multihop(query: str) -> bool:
    """A narrative query is multi-hop if it compares, or spans two+ companies."""
    if _COMPARE_CUE.search(query):
        return True
    try:
        return len(extract_filters(query).get("tickers") or []) >= 2
    except Exception:
        return False


def heuristic_route(query: str) -> str | None:
    """NUMERIC / NARRATIVE / MULTI_HOP if unambiguous, else None (defer to LLM)."""
    metric = bool(_METRIC.search(query))
    narrative = bool(_NARRATIVE_CUE.search(query))
    if narrative and not metric and _is_multihop(query):
        return "MULTI_HOP"
    if metric and not narrative:
        return "NUMERIC"
    if narrative and not metric:
        return "NARRATIVE"
    return None


def route(query: str, model: str = LLM_MODEL) -> tuple[str, bool]:
    """Return (route, used_llm). Rule-based when confident, LLM otherwise."""
    r = heuristic_route(query)
    if r is not None:
        return r, False
    return classify(query, model), True


_RESOLVE_PROMPT = """Identify the single primary US-listed company a financial question is
about, for looking up its SEC 10-K filing. If the question does not name a specific company,
return null for both fields.

Query: {query}

Return ONLY JSON: {{"ticker": "<STOCK TICKER or null>", "name": "<company name or null>"}}"""

_TICKER_RE = re.compile(r"^[A-Za-z.]{1,6}$")


def resolve_company(query: str, model: str = LLM_MODEL) -> dict | None:
    """Best-effort {ticker, name} for the company a query is about, or None.

    Used only when the rule-based filter found no *known* company — so this LLM call
    happens on unknown-company (or company-less) queries, not on the common case.
    """
    text = chat([{"role": "user", "content": _RESOLVE_PROMPT.format(query=query)}],
                model=model, json_mode=True)
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    t = d.get("ticker")
    if isinstance(t, str) and _TICKER_RE.match(t):
        return {"ticker": t.upper().replace(".", ""), "name": (d.get("name") or t).strip()}
    return None


ROUTER_PROMPT = """Classify the financial query into exactly one type:
- NUMERIC: asks for a specific reported figure/metric (revenue, R&D, EPS, net income,
  gross margin, operating income, assets, cash, etc.) for a company/period. Answerable
  from a single financial-statement number.
- NARRATIVE: asks for qualitative or explanatory content about ONE company/topic (risks,
  strategy, why a metric changed, business descriptions, competition).
- MULTI_HOP: needs evidence from MULTIPLE companies or MULTIPLE distinct sub-topics to
  answer fully (e.g. comparing two companies' risks, or a question with several parts).

Query: {query}

Return ONLY JSON: {{"type": "NUMERIC" | "NARRATIVE" | "MULTI_HOP"}}"""


def classify(query: str, model: str = LLM_MODEL) -> str:
    text = chat(
        [{"role": "user", "content": ROUTER_PROMPT.format(query=query)}],
        model=model, json_mode=True,
    )
    try:
        t = json.loads(text).get("type", "NARRATIVE").upper()
        if t == "NUMERIC":
            return "NUMERIC"
        if t == "MULTI_HOP":
            return "MULTI_HOP"
        return "NARRATIVE"
    except (json.JSONDecodeError, AttributeError):
        return "NARRATIVE"
