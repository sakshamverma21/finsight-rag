# FinSight RAG

**A retrieval *system* over SEC 10-K filings** — not "RAG that works," but one where numbers are
exact (not fuzzy vector matches), every answer is citable, the corpus grows on demand, and every
design choice is justified by a measured ablation.

Numeric questions are routed to a **text-to-SQL path over structured XBRL data** (exact figures, to
the dollar); narrative questions go through **hybrid retrieval + cross-encoder reranking** with
grounded citations. It also compares companies with charts and ingests any SEC filer on demand.

<!-- Replace with your recording -->
![FinSight demo](docs/demo.gif)

---

## Why it's different

| Question | Path | Why it matters |
|---|---|---|
| **Numeric** — "Apple's FY2023 revenue?" | Router → **text-to-SQL over DuckDB** (XBRL facts) | Vector search returns an *approximate* number — in finance, approximate is wrong. We answer from structured data, exactly, and cite the source row. |
| **Narrative** — "supply-chain risks?" | Hybrid (dense + BM25) → **rerank** → grounded citations | Every claim maps to a source chunk; a faithfulness check flags unsupported ones. |
| **Comparison** — "AAPL vs MSFT vs GOOGL revenue" | text-to-SQL across companies → **table + chart** | Structured data makes comparison and plotting trivial; a vector RAG can't do it reliably. |
| **Unknown company** — "Oracle's revenue?" | **Auto-detect → fetch 10-K → embed → answer** (~30s) | The corpus expands on demand over all ~8,000 SEC filers, and persists. |

## Measured, not asserted

Every component earns its place by a delta on an offline eval — full tables in
[BENCHMARK.md](BENCHMARK.md):

- **Retrieval ablation** on a company-scoped golden set: dense → +hybrid → +reranker → +metadata-filter, **monotonic** gains; the reranker is the single biggest lever.
- **Numeric accuracy**: exact-match against reported figures on a 168-pair auto-generated golden set.
- **Latency** (warm): numeric ~1.6 s, narrative ~2.6–3.1 s, **cached repeat ~29 ms** — instrumented as `system_ms` (ours) vs `llm_ms` (provider) so the breakdown is honest, not hidden.

Full results are in [BENCHMARK.md](BENCHMARK.md); the design rationale and component deep-dives are
in [finsight_rag_architecture.md](finsight_rag_architecture.md).

## Architecture

```
Query
 → L1 cache (Redis, exact-match; degrades to a no-op if absent)
 → Adaptive router (rule-based fast path decides 94% of queries; LLM fallback): NUMERIC | NARRATIVE
      • NUMERIC   → schema-linked text-to-SQL over DuckDB XBRL facts → exact rows (+ self-repair loop)
      • NARRATIVE → hybrid (bge-small + BM25) → RRF → cross-encoder rerank → parent expansion
 → Generation (LiteLLM → Gemini flash-lite) with grounded citations
 → Response (+ table/chart for numeric; auto-ingest for unknown companies)
```

Design principle throughout: **do work at the right time.** Embeddings are a *build-time* artifact
(150 s once → 0.2 s to load), models load *once* at startup, and only per-request work runs per
request. The retriever is brute-force in-memory on purpose — at a few thousand vectors it's ~20 ms,
so a vector DB would optimize the wrong thing (that's a measured decision, not laziness).

## Tech stack

`edgartools` (SEC XBRL) · `DuckDB` (structured facts) · `bge-small` + `BM25` (hybrid retrieval) ·
`ms-marco-MiniLM` cross-encoder (rerank) · `LiteLLM` → **Gemini flash-lite** (model-agnostic) ·
`FastAPI` · `Redis` (cache) · **Docker** (multi-stage, CPU-only torch — 4.9 GB → 2.7 GB) · `uv` (locked env).

## Run it locally

```bash
# 1. Environment
uv sync
cp .env.example .env          # add GEMINI_API_KEY (free: https://aistudio.google.com)

# 2. Build the corpus + index (fetches 10 companies' latest 10-K from SEC)
uv run python scripts/ingest.py
uv run python scripts/build_index.py

# 3. Serve
uv run uvicorn finsight.api.main:app --port 8000     # → http://localhost:8000
```

Or the whole stack (API + Redis) in one command: **`docker compose up --build`**.

## Repository map

```
finsight/
  agent/        router (rule-based + LLM), orchestrator
  retrieval/    hybrid retriever, reranker, in-memory index, metadata filter
  structured/   XBRL → DuckDB facts, text-to-SQL (schema linking + repair)
  ingestion/    SEC fetch, section extraction, chunking, on-demand ingest
  generation/   grounded generation + citation checking
  cache/        Redis L1 cache (graceful no-op without Redis)
  api/          FastAPI service + single-file UI
scripts/        ingest, build-index, evals
eval/           golden sets + eval harness
```

## License

MIT — see [LICENSE](LICENSE). Built with public SEC EDGAR data.
