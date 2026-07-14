"""Lightweight, deterministic query pre-filtering.

Parse company (ticker) and fiscal-year hints from a query so retrieval can be
constrained to the right filer/period *before* searching. Rule-based on purpose:
instant, no LLM call, no cost. The LLM router (M2/M3) can supply the same signals
for harder cases, but the common case ("Microsoft ... 2024") is pure regex.
"""

from __future__ import annotations

import re

# alias (lowercased, word-boundary) -> ticker
COMPANY_ALIASES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL",
    "meta": "META", "facebook": "META",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "intel": "INTC",
    "amd": "AMD", "advanced micro devices": "AMD",
    "salesforce": "CRM",
}
_TICKERS = set(COMPANY_ALIASES.values())


def register_company(ticker: str, name: str | None = None) -> None:
    """Teach the filter a company ingested on demand (so its name resolves too)."""
    ticker = ticker.upper()
    _TICKERS.add(ticker)
    if name:
        # first word of the legal name is usually the recognizable alias ("Oracle Corporation")
        COMPANY_ALIASES.setdefault(name.split()[0].lower(), ticker)
    COMPANY_ALIASES.setdefault(ticker.lower(), ticker)

_FY_FULL = re.compile(r"\bFY\s?(20\d{2})\b|\bfiscal(?:\s+year)?\s+(20\d{2})\b|\b(20\d{2})\b", re.I)
_FY_SHORT = re.compile(r"\bFY\s?(\d{2})\b", re.I)

_MIN_YEAR, _MAX_YEAR = 2018, 2026


def extract_filters(query: str) -> dict:
    """Return {'tickers': [...], 'fiscal_years': [...]} parsed from the query."""
    q = query.lower()

    tickers: set[str] = set()
    for alias, ticker in COMPANY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            tickers.add(ticker)
    # explicit uppercase ticker symbols, e.g. "AAPL"
    for sym in re.findall(r"\b[A-Z]{2,5}\b", query):
        if sym in _TICKERS:
            tickers.add(sym)

    years: set[int] = set()
    for m in _FY_FULL.finditer(query):
        y = next(g for g in m.groups() if g)
        years.add(int(y))
    for m in _FY_SHORT.finditer(query):
        years.add(2000 + int(m.group(1)))
    years = {y for y in years if _MIN_YEAR <= y <= _MAX_YEAR}

    return {"tickers": sorted(tickers), "fiscal_years": sorted(years)}
