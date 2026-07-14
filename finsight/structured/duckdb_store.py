"""DuckDB store for the structured financial 'facts' table."""

from __future__ import annotations

import duckdb
import pandas as pd

from finsight.config import DB_PATH

FACTS_COLUMNS = [
    "ticker", "form", "fiscal_year", "fiscal_quarter", "filing_date", "accession",
    "statement", "concept", "label", "dimension_member", "period", "period_end",
    "period_fy", "value",
]

FACTS_DDL = """
CREATE TABLE IF NOT EXISTS facts (
    ticker VARCHAR,
    form VARCHAR,
    fiscal_year INTEGER,        -- fiscal year of the source FILING (provenance)
    fiscal_quarter INTEGER,
    filing_date VARCHAR,
    accession VARCHAR,
    statement VARCHAR,
    concept VARCHAR,            -- canonical us-gaap tag (consistent across filers)
    label VARCHAR,              -- human label (varies by filer)
    dimension_member VARCHAR,   -- NULL = statement total; else a segment breakdown
    period VARCHAR,
    period_end VARCHAR,
    period_fy INTEGER,          -- fiscal year of THIS fact (from period_end) -- query on this
    value DOUBLE                -- native units (full dollars, or per-share for EPS)
);
"""


def connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def init_schema(con) -> None:
    con.execute(FACTS_DDL)


def load_facts(con, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    frame = df.copy()
    for col in FACTS_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    frame = frame[FACTS_COLUMNS]
    con.register("_facts_stage", frame)
    con.execute("INSERT INTO facts SELECT * FROM _facts_stage")
    con.unregister("_facts_stage")
    return len(frame)
