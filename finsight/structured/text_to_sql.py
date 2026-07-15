"""NUMERIC path: natural language -> DuckDB SQL over the `facts` table -> exact rows.

The LLM is grounded with the schema and the actual (concept|label) pairs available,
so it filters on canonical us-gaap concepts. Generated SQL is validated read-only
before execution. Answers are templated from the returned rows (never re-generated),
so numbers are exact and every figure cites its (ticker, period, statement, concept).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import duckdb
import numpy as np

from finsight.config import DB_PATH, LLM_MODEL
from finsight.llm import chat, last_throttle_wait_ms

_STARTS_SELECT = re.compile(r"^\s*(WITH|SELECT)\b", re.I)

# --- Schema linking -------------------------------------------------------
# Sending every concept for a ticker costs ~1700 prompt tokens. Instead we embed
# the question and keep only the most similar concepts. Measured on the 168-pair
# golden set: K=20 already gives 100% concept-recall (the needed concept is always
# present) at ~4x fewer tokens. We use K=25 for margin, because the golden queries
# are templated and mirror the labels — free-form phrasing is harder — and we always
# union in the core metrics so the most-asked figures can never fall out of the window.
SCHEMA_TOP_K = 25

CORE_CONCEPTS = {
    "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap_Revenues",
    "us-gaap_NetIncomeLoss",
    "us-gaap_GrossProfit",
    "us-gaap_OperatingIncomeLoss",
    "us-gaap_ResearchAndDevelopmentExpense",
    "us-gaap_EarningsPerShareDiluted",
    "us-gaap_EarningsPerShareBasic",
}

_embedder = None
_concept_cache: dict[tuple, tuple[list[tuple[str, str]], np.ndarray]] = {}


def _get_embedder():
    global _embedder
    if _embedder is None:
        from finsight.indexing.embedder import Embedder
        _embedder = Embedder()
    return _embedder
_FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|COPY|PRAGMA|REPLACE)\b", re.I)

SCHEMA_DOC = """Table `facts` — one row per reported numeric fact:
  ticker VARCHAR            -- 'AAPL','MSFT','GOOGL', ...
  statement VARCHAR         -- 'income_statement' | 'balance_sheet' | 'cashflow_statement'
  concept VARCHAR           -- canonical us-gaap tag; CONSISTENT across companies (prefer filtering on this)
  label VARCHAR             -- human label; VARIES by company
  dimension_member VARCHAR  -- NULL = statement total; non-null = a segment breakdown
  period_fy INTEGER         -- fiscal year of THIS fact (filter years on this, NOT fiscal_year)
  period_end VARCHAR        -- 'YYYY-MM-DD'
  value DOUBLE              -- native units: full dollars (e.g. 416161000000.0) or per-share for EPS

Rules:
- Read-only, a single SELECT.
- The `concept` you filter on MUST be one that appears in the provided concept|label
  list. Do NOT invent a concept id (e.g. do not write 'us-gaap_Revenues' if it is not
  listed) — pick the closest one that IS listed.
- Use the EXACT fiscal year(s) named in the question for `period_fy`.
- Different companies may tag the SAME metric under different concept ids. To compare a
  metric across companies, use `concept IN (...)` covering every relevant concept from
  the list (e.g. revenue may be 'us-gaap_Revenues' for one filer and
  'us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax' for another).
- For a company-level total, require `dimension_member IS NULL`.
- Prefer matching `concept` (cross-company consistent) over `label`.
- String literals MUST use single quotes ('...'). In DuckDB double quotes denote an
  identifier (a column name), so "us-gaap_X" is a column reference, not a string.
- "Net income" means the figure attributable to the parent's common stockholders:
  prefer `us-gaap_NetIncomeLoss` over `us-gaap_ProfitLoss` (which consolidates
  noncontrolling interests), unless the question explicitly asks for consolidated profit.
- Always SELECT ticker, period_fy, label, value (and period_end) so the answer is citable."""


@dataclass
class NumericAnswer:
    query: str
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    text: str = ""
    latency_ms: float = 0.0
    error: str | None = None
    trend: list = field(default_factory=list)  # multi-year series for a single (ticker, concept)


def _concept_rows(con, tickers: list[str] | None, embedder=None) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Distinct (concept, label) for the ticker(s), deduped by concept, with embeddings.

    Cached per ticker-set: the schema doesn't change between queries.
    `embedder` is injected by the service so we reuse the ONE loaded model rather
    than lazily constructing a second copy on the first request.
    """
    embedder = embedder or _get_embedder()
    key = tuple(tickers or [])
    if key not in _concept_cache:
        sql = "SELECT DISTINCT concept, label FROM facts WHERE dimension_member IS NULL"
        params: list = []
        if tickers:
            sql += " AND ticker IN (" + ",".join("?" * len(tickers)) + ")"
            params += tickers
        sql += " ORDER BY concept, label"
        seen: dict[str, tuple[str, str]] = {}
        for c, lab in con.execute(sql, params).fetchall():
            seen.setdefault(c, (c, lab))
        rows = list(seen.values())
        vecs = embedder.embed_documents([f"{c} | {lab}" for c, lab in rows])
        _concept_cache[key] = (rows, vecs)
    return _concept_cache[key]


def _reference(con, tickers: list[str] | None, query: str | None = None,
               top_k: int = SCHEMA_TOP_K, embedder=None) -> str:
    """Concept|label lines for the prompt, narrowed to those relevant to `query`."""
    embedder = embedder or _get_embedder()
    rows, vecs = _concept_rows(con, tickers, embedder)
    if query is None or len(rows) <= top_k:
        kept = rows
    else:
        sims = vecs @ embedder.embed_query(query)
        kept = [rows[i] for i in np.argsort(-sims)[:top_k]]
        present = {c for c, _ in kept}
        kept += [(c, lab) for c, lab in rows if c in CORE_CONCEPTS and c not in present]
    return "\n".join(f"  {c} | {lab}" for c, lab in kept)


def _hints(tickers: list[str] | None, fiscal_years: list[int] | None) -> str:
    """Inject filters we already parsed, so the model doesn't guess them wrong."""
    out = ""
    if tickers:
        vals = ", ".join(f"'{t}'" for t in tickers)
        out += f"\nRestrict to ticker(s) {vals}. Use exactly these ticker values verbatim."
    if fiscal_years:
        ys = ", ".join(str(y) for y in fiscal_years)
        out += f"\nThe question refers to fiscal year(s) {ys}. Filter `period_fy IN ({ys})`."
    return out + "\n" if out else ""


def generate_sql(query: str, con, tickers=None, model: str = LLM_MODEL,
                 embedder=None, fiscal_years=None) -> str:
    prompt = (
        f"{SCHEMA_DOC}\n{_hints(tickers, fiscal_years)}\n"
        f"Available concept | label pairs:\n{_reference(con, tickers, query, embedder=embedder)}\n\n"
        f"Write one DuckDB SQL SELECT that answers:\n{query}\n\n"
        'Return ONLY JSON: {"sql": "<query>"}'
    )
    text = chat([{"role": "user", "content": prompt}], model=model, json_mode=True)
    return json.loads(text)["sql"].strip().rstrip(";")


_CONCEPT_IN_SQL = re.compile(r"concept\s*=\s*'([^']+)'", re.I)


def repair_sql(query: str, tickers, model: str, bad_sql: str, problem: str,
               fiscal_years=None, con=None, embedder=None) -> str:
    """One recovery attempt — with the concept list in front of the model this time.

    The first attempt often fails by picking a concept that doesn't exist for this
    company (e.g. 'us-gaap_Revenues' for a filer that tags revenue differently). The
    repair re-shows the available concepts and, if we can tell the used concept is
    invalid, says so explicitly.
    """
    note = ""
    used = _CONCEPT_IN_SQL.search(bad_sql)
    if used and con is not None:
        c = used.group(1)
        exists = con.execute(
            "SELECT 1 FROM facts WHERE concept = ? " + (
                "AND ticker IN (" + ",".join("?" * len(tickers)) + ")" if tickers else "") + " LIMIT 1",
            [c, *(tickers or [])],
        ).fetchone()
        if not exists:
            note = (f"\nThe concept '{c}' does NOT exist for this company. You MUST pick a "
                    f"concept from the list below.")

    reference = _reference(con, tickers, query, embedder=embedder) if con is not None else ""
    prompt = (
        f"{SCHEMA_DOC}\n{_hints(tickers, fiscal_years)}{note}\n"
        f"Available concept | label pairs:\n{reference}\n\n"
        f"This SQL was generated for the question below but {problem}:\n{bad_sql}\n\n"
        f"Question: {query}\n\nReturn corrected SQL. ONLY JSON: {{\"sql\": \"<query>\"}}"
    )
    text = chat([{"role": "user", "content": prompt}], model=model, json_mode=True)
    return json.loads(text)["sql"].strip().rstrip(";")


def _validate(sql: str) -> None:
    if not _STARTS_SELECT.match(sql):
        raise ValueError("query must be a single SELECT/WITH")
    if _FORBIDDEN.search(sql):
        raise ValueError("non-read-only SQL rejected")
    if ";" in sql:
        raise ValueError("multiple statements not allowed")


def _fmt_value(v, concept: str) -> str:
    if v is None:
        return "n/a"
    if "PerShare" in concept or abs(v) < 1000:
        return f"{v:,.2f}"
    if abs(v) >= 1e9:
        return f"${v / 1e9:,.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.2f}M"
    return f"${v:,.0f}"


def _format_rows(columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "The structured financial data contains no matching figure."
    idx = {c: i for i, c in enumerate(columns)}
    lines = []
    for r in rows:
        tkr = r[idx["ticker"]] if "ticker" in idx else ""
        fy = r[idx["period_fy"]] if "period_fy" in idx else ""
        lab = r[idx["label"]] if "label" in idx else ""
        val = r[idx["value"]] if "value" in idx else None
        concept = r[idx["concept"]] if "concept" in idx else ""
        lines.append(f"  {tkr} FY{fy}  {lab}: {_fmt_value(val, concept or '')}")
    return "\n".join(lines)


def _run(con, sql: str) -> tuple[list[str], list[tuple]]:
    _validate(sql)
    cur = con.execute(sql)
    return [d[0] for d in cur.description], cur.fetchmany(25)


# --- Multi-year trend (deterministic, no LLM) -----------------------------
# When an answer resolves to a single (ticker, concept), the facts table already
# holds every fiscal year that filing reported (~3 years). We surface that series so
# a single figure gains context — a sparkline + YoY — without a second model call.
_CONCEPTS_ANY = re.compile(r"concept\s*(?:=\s*'([^']+)'|IN\s*\(([^)]*)\))", re.I)


def _sql_concepts(sql: str) -> set[str]:
    cs: set[str] = set()
    for m in _CONCEPTS_ANY.finditer(sql):
        if m.group(1):
            cs.add(m.group(1))
        elif m.group(2):
            cs |= {x.strip().strip("'") for x in m.group(2).split(",") if x.strip()}
    return cs


def _metric_series(con, ticker: str, concept: str) -> list[dict]:
    q = ("SELECT period_fy, value, label, period_end FROM facts "
         "WHERE ticker = ? AND concept = ? AND dimension_member IS NULL "
         "AND period_fy IS NOT NULL AND value IS NOT NULL ORDER BY period_fy")
    return [
        {"ticker": ticker, "period_fy": fy, "value": val, "label": lab, "period_end": pe}
        for fy, val, lab, pe in con.execute(q, [ticker, concept]).fetchall()
    ]


def _trend_for(con, columns: list[str], rows: list[tuple], sql: str) -> list[dict]:
    """A year-over-year series, but only when the answer is one ticker + one concept."""
    try:
        if "ticker" not in columns:
            return []
        tickers_in = {r[columns.index("ticker")] for r in rows}
        concepts = _sql_concepts(sql)
        if len(tickers_in) == 1 and len(concepts) == 1:
            series = _metric_series(con, next(iter(tickers_in)), next(iter(concepts)))
            return series if len(series) >= 2 else []
    except Exception:
        return []
    return []


def answer_numeric(query: str, tickers: list[str] | None = None,
                   model: str = LLM_MODEL, db_path=DB_PATH, repair: bool = True,
                   embedder=None, fiscal_years=None) -> NumericAnswer:
    """Generate SQL, execute, and (once) repair on error or an empty result.

    An empty result is treated as a failure worth retrying: a query that silently
    returns no rows (e.g. a hallucinated ticker) is as wrong as one that crashes.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    t0 = time.perf_counter()
    sql = ""
    try:
        sql = generate_sql(query, con, tickers, model, embedder=embedder, fiscal_years=fiscal_years)
        try:
            columns, rows = _run(con, sql)
            problem = None if rows else "returned no rows"
            # Completeness: a comparison must not silently drop companies. If some
            # requested tickers are missing, they likely tag the metric under a
            # different concept — repair toward `concept IN (...)`.
            if rows and tickers and len(tickers) > 1 and "ticker" in columns:
                got = {r[columns.index("ticker")] for r in rows}
                missing = [t for t in tickers if t not in got]
                if missing:
                    problem = (f"returned no rows for {missing} — different companies may tag this "
                               f"metric under different concept ids; use `concept IN (...)` covering "
                               f"the relevant concepts for ALL requested companies")
        except Exception as e:
            columns, rows, problem = [], [], f"failed with error: {type(e).__name__}: {e}"

        if problem and repair:
            fixed = repair_sql(query, tickers, model, sql, problem,
                               fiscal_years=fiscal_years, con=con, embedder=embedder)
            try:
                columns, rows = _run(con, fixed)
                sql = fixed
            except Exception:
                pass  # keep the original sql/rows; reported below

        if not rows:
            return NumericAnswer(query, sql, columns, rows,
                                 "The structured financial data contains no matching figure.",
                                 _elapsed_ms(t0), error=problem)
        return NumericAnswer(query, sql, columns, rows, _format_rows(columns, rows), _elapsed_ms(t0),
                             trend=_trend_for(con, columns, rows, sql))
    except Exception as e:
        return NumericAnswer(query, sql, text="Could not answer from structured data.",
                             latency_ms=_elapsed_ms(t0), error=f"{type(e).__name__}: {e}")
    finally:
        con.close()


def _elapsed_ms(t0: float) -> float:
    """Wall time minus client-side throttle sleep (free-tier pacing is not system latency)."""
    return max((time.perf_counter() - t0) * 1000 - last_throttle_wait_ms(), 0.0)
