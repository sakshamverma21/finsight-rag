"""Reusable ingestion: tickers -> parent/child chunks (JSONL) + XBRL facts (DuckDB).

Per-ticker isolation: a single filing that fails to fetch/parse is recorded and
skipped rather than aborting the whole corpus build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from finsight.config import CHUNKS_PATH, SECTIONS_10K, ensure_dirs
from finsight.ingestion.chunker import chunk_section
from finsight.ingestion.edgar_fetcher import fetch_latest_10k
from finsight.ingestion.section_extractor import extract_sections
from finsight.structured import duckdb_store as store
from finsight.structured.table_extractor import extract_facts


@dataclass
class IngestStat:
    ticker: str
    fiscal_year: int | None = None
    sections: int = 0
    children: int = 0
    facts: int = 0
    error: str | None = None


def ingest(tickers: list[str], chunks_path: Path = CHUNKS_PATH) -> tuple[list[IngestStat], int]:
    ensure_dirs()
    con = store.connect()
    store.init_schema(con)
    con.execute("DELETE FROM facts")  # full rebuild each run (idempotent)

    all_chunks = []
    stats: list[IngestStat] = []

    for ticker in tickers:
        try:
            ref = fetch_latest_10k(ticker)
            base_meta = ref.base_meta()
            sections = extract_sections(ref.obj)
            t_chunks = []
            for item, text in sections.items():
                t_chunks += chunk_section(item, SECTIONS_10K[item], text, base_meta)
            n_facts = store.load_facts(con, extract_facts(ref.obj, base_meta))
            all_chunks += t_chunks
            stats.append(IngestStat(
                ticker, ref.fiscal_year, len(sections),
                sum(1 for c in t_chunks if c.level == "paragraph"), n_facts,
            ))
        except Exception as e:  # noqa: BLE001 - isolate per-ticker failures
            stats.append(IngestStat(ticker, error=f"{type(e).__name__}: {e}"))

    with chunks_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps({
                "chunk_id": c.chunk_id,
                "parent_chunk_id": c.parent_chunk_id,
                "level": c.level,
                "metadata": c.metadata,
                "text": c.text,
            }, ensure_ascii=False) + "\n")

    con.close()
    return stats, len(all_chunks)


def ingest_company(ticker: str):
    """Fetch one company's latest 10-K → (chunk dicts, facts DataFrame, company name).

    Pure fetch/extract — touches no shared state (no DuckDB, no index). The caller
    commits the facts and chunks under a lock, so this slow part (SEC fetch + parse)
    runs without blocking queries. Additive by design (never wipes existing data).
    """
    ref = fetch_latest_10k(ticker)  # raises if no original 10-K
    base_meta = ref.base_meta()

    chunks: list[dict] = []
    for item, text in extract_sections(ref.obj).items():
        for c in chunk_section(item, SECTIONS_10K[item], text, base_meta):
            chunks.append({
                "chunk_id": c.chunk_id, "parent_chunk_id": c.parent_chunk_id,
                "level": c.level, "metadata": c.metadata, "text": c.text,
            })

    return chunks, extract_facts(ref.obj, base_meta), ref.company_name
