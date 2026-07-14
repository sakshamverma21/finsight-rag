"""Auto-generate the numeric golden set from the DuckDB facts table.

Objective by construction: the expected answer IS the reported figure. Questions
are phrased naturally (metric name, not the us-gaap concept) so they test the
router + text-to-SQL's ability to map language -> the right structured fact.

Run:  uv run python scripts/build_numeric_golden.py
"""

from __future__ import annotations

import json

import duckdb

from finsight.config import DB_PATH, PROJECT_ROOT

OUT = PROJECT_ROOT / "eval" / "golden" / "numeric_pilot.json"

COMPANY = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "META": "Meta",
    "AMZN": "Amazon", "NVDA": "NVIDIA", "TSLA": "Tesla", "INTC": "Intel",
    "AMD": "AMD", "CRM": "Salesforce",
}

# (metric key, question template, candidate canonical concepts)
METRICS = [
    ("revenue", "What was {co}'s total revenue in fiscal year {fy}?",
     ["us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap_Revenues"]),
    ("rd", "How much did {co} spend on research and development in fiscal year {fy}?",
     ["us-gaap_ResearchAndDevelopmentExpense"]),
    ("net_income", "What was {co}'s net income in fiscal year {fy}?",
     ["us-gaap_NetIncomeLoss"]),
    ("gross_profit", "What was {co}'s gross profit in fiscal year {fy}?",
     ["us-gaap_GrossProfit"]),
    ("operating_income", "What was {co}'s operating income in fiscal year {fy}?",
     ["us-gaap_OperatingIncomeLoss"]),
    ("diluted_eps", "What was {co}'s diluted earnings per share in fiscal year {fy}?",
     ["us-gaap_EarningsPerShareDiluted"]),
]


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    years = [r[0] for r in con.execute(
        "SELECT DISTINCT period_fy FROM facts WHERE period_fy IS NOT NULL ORDER BY period_fy DESC"
    ).fetchall()]

    entries = []
    for ticker, co in COMPANY.items():
        for fy in years:
            for key, template, concepts in METRICS:
                row = con.execute(
                    f"""
                    SELECT concept, value FROM facts
                    WHERE ticker = ? AND period_fy = ? AND dimension_member IS NULL
                      AND concept IN ({",".join("?" * len(concepts))})
                    ORDER BY value DESC LIMIT 1
                    """,
                    [ticker, fy, *concepts],
                ).fetchone()
                if not row:
                    continue
                concept, value = row
                entries.append({
                    "id": f"{ticker.lower()}_{key}_{fy}",
                    "query": template.format(co=co, fy=fy),
                    "ticker": ticker,
                    "period_fy": int(fy),
                    "concept": concept,
                    "expected_value": float(value),
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"queries": entries}, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} numeric golden pairs -> {OUT.relative_to(PROJECT_ROOT)}")
    for e in entries[:5]:
        print(f"  {e['id']:24} {e['query']}  -> {e['expected_value']:,.2f}")


if __name__ == "__main__":
    main()
