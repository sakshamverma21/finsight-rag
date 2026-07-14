"""Turn XBRL financial statements into a long-format 'facts' table.

Each row = one (concept, period) numeric fact, carrying the canonical us-gaap
concept id, a human label, an optional dimension member (segment), and the
filing metadata. This is what the NUMERIC / text-to-SQL path queries.
"""

from __future__ import annotations

import re

import pandas as pd

_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")

STATEMENTS = ["income_statement", "balance_sheet", "cashflow_statement"]


def _period_cols(df: pd.DataFrame) -> list[str]:
    """Columns whose header contains a date, e.g. '2025-09-27 (FY)'."""
    return [c for c in df.columns if _DATE.search(str(c))]


def extract_facts(tenk, base_meta: dict) -> pd.DataFrame:
    """Melt all available statements into one long facts DataFrame."""
    frames: list[pd.DataFrame] = []
    fin = tenk.financials

    for stmt in STATEMENTS:
        try:
            df = getattr(fin, stmt)().to_dataframe()
        except Exception:
            continue

        period_cols = _period_cols(df)
        if not period_cols:
            continue

        id_cols = [c for c in ("concept", "label", "dimension_member") if c in df.columns]
        long = df.melt(
            id_vars=id_cols,
            value_vars=period_cols,
            var_name="period",
            value_name="value",
        )

        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["value"])
        long["period_end"] = long["period"].astype(str).str.extract(_DATE.pattern)
        # The fact's fiscal year comes from its OWN period end, not the filing's
        # fiscal year — one filing reports several years. This is the queryable year.
        long["period_fy"] = pd.to_numeric(long["period_end"].str[:4], errors="coerce").astype("Int64")
        long["statement"] = stmt
        for k, v in base_meta.items():
            long[k] = v

        frames.append(long)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
