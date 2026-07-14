"""Central config: paths, corpus definition, and EDGAR identity setup."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from edgar import set_identity

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "finsight.duckdb"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"  # the corpus artifact (last ingestion)
INDEX_PATH = DATA_DIR / "index.npz"      # prebuilt embeddings (build-time artifact)

# --- EDGAR ---
# SEC requires a declared identity (any real email). No API key needed.
EDGAR_IDENTITY = os.environ.get("EDGAR_IDENTITY", "research@example.com")

# --- LLM (via LiteLLM) ---
# Gemini free tier: gemini-3.1-flash-lite is fast (~1s) and doesn't "think" (the
# full 3.x flash models default to thinking → 9-29s, unusable). Free tier gives
# 15 RPM / 1500 RPD / 250k TPM. Swap providers via FINSIGHT_LLM_MODEL:
#   groq/qwen/qwen3-32b  (set FINSIGHT_TPM_LIMIT=5200)  |  openrouter/...  |  H100
LLM_MODEL = os.environ.get("FINSIGHT_LLM_MODEL", "gemini/gemini-3.1-flash-lite")

# --- Corpus ---
PILOT_TICKERS = ["AAPL", "MSFT", "GOOGL"]
ALL_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN",
    "NVDA", "TSLA", "INTC", "AMD", "CRM",
]

# 10-K sections indexed for the narrative path (item number -> readable name).
# Item 8 (financial statements) is intentionally excluded here: its numbers are
# handled by the structured/XBRL path (DuckDB), and its text is table-noise.
SECTIONS_10K = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 7": "MD&A",
    "Item 7A": "Market Risk",
}

_identity_set = False


def init_edgar() -> None:
    """Set the EDGAR identity once per process (idempotent)."""
    global _identity_set
    if not _identity_set:
        set_identity(EDGAR_IDENTITY)
        _identity_set = True


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
