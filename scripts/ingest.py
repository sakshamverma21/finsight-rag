"""Ingest the corpus: fetch latest 10-K per ticker, build chunks + facts.

Run:  uv run python scripts/ingest.py            # all 10 companies
      uv run python scripts/ingest.py --pilot    # 3 pilot companies
"""

from __future__ import annotations

import argparse

import duckdb

from finsight.config import ALL_TICKERS, CHUNKS_PATH, DB_PATH, PILOT_TICKERS, PROJECT_ROOT
from finsight.ingestion.pipeline import ingest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="ingest only the 3 pilot companies")
    args = ap.parse_args()
    tickers = PILOT_TICKERS if args.pilot else ALL_TICKERS

    print(f"Ingesting {len(tickers)} companies: {', '.join(tickers)}\n")
    stats, total_chunks = ingest(tickers)

    print(f"{'ticker':8} {'FY':>4} {'sections':>9} {'children':>9} {'facts':>7}  status")
    print("-" * 56)
    ok = 0
    for s in stats:
        if s.error:
            print(f"{s.ticker:8} {'—':>4} {'—':>9} {'—':>9} {'—':>7}  FAIL: {s.error}")
        else:
            ok += 1
            print(f"{s.ticker:8} {s.fiscal_year:>4} {s.sections:>9} {s.children:>9} {s.facts:>7}  ok")
    print("-" * 56)
    print(f"{ok}/{len(tickers)} companies ok | {total_chunks} chunks -> {CHUNKS_PATH.relative_to(PROJECT_ROOT)}")

    # cross-company sanity: R&D latest FY per company
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        """
        SELECT ticker, period_fy, value FROM facts
        WHERE concept ILIKE '%ResearchAndDevelopmentExpense%' AND dimension_member IS NULL
        QUALIFY row_number() OVER (PARTITION BY ticker ORDER BY period_fy DESC) = 1
        ORDER BY value DESC
        """
    ).fetchall()
    print("\nR&D expense (latest FY), all companies:")
    for t, fy, v in rows:
        print(f"  {t:6} FY{fy}  ${v / 1e9:,.2f}B")
    con.close()


if __name__ == "__main__":
    main()
