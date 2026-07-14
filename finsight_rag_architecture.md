# FinSight RAG — Architecture Guide & Build Phases

> **A latency-aware, numerically-correct, agentic retrieval system for SEC financial filings.**
>
> *Built to demonstrate what production retrieval engineering looks like in 2026 — not "RAG that works," but a system that routes numeric questions to structured data, reasons iteratively over multi-document queries, cites every claim, and justifies every component with a measured ablation.*

---

## 1. Why This Project Exists

Financial analysts spend a large share of their time simply *locating* information scattered across hundreds of filings. Keyword search (the default in most terminals) misses semantic matches; naive semantic search misses exact financial terms and — critically — gets **numbers** wrong. Companies like AlphaSense and Hebbia have built large businesses on solving retrieval over filings, which validates the problem space.

This project is not competing with them. It demonstrates that you understand the **systems problems** they solve: numeric correctness over tables, multi-hop reasoning across documents, honest citations, latency budgets, and — above all — a pipeline where every design choice is *measured*, not asserted.

**Why not just stuff everything into a long-context LLM?** At ~400 filings, dumping the corpus into context on every query is economically absurd (dollars per query) and gives no citation guarantees or numeric determinism. The retrieval layer exists for **cost, consistency, and verifiable citations** — none of which long context solves.

---

## 2. Project Goal (One Sentence)

Build a retrieval system over SEC 10-K/10-Q filings that **routes numeric questions to a structured (text-to-SQL) path and narrative questions to hybrid vector retrieval**, reasons iteratively over multi-hop queries, grounds every claim in a citable source, and proves each component's value through an offline ablation study — deployed as a live, containerized demo.

---

## 3. Data Strategy

### Source
**SEC EDGAR** — free, public, programmatic. No scraping, no licensing.

Use **`edgartools`** (MIT-licensed, no API key):
```bash
pip install edgartools
```
```python
from edgar import Company
apple = Company("AAPL")
filing = apple.get_filings(form="10-K").latest(1)
```

### Corpus Definition
| Parameter | Value |
|---|---|
| Companies | 10 large-cap tech: AAPL, MSFT, GOOGL, META, AMZN, NVDA, TSLA, INTC, AMD, CRM |
| Filing types | 10-K (annual) + 10-Q (quarterly) |
| Time range | FY2023 – FY2025 (latest 3 fiscal years via XBRL; older years available from prior filings) |
| Total corpus | ~400 documents |
| Why this scope | Hard enough to stress multi-document and numeric retrieval; small enough to iterate fast and re-index cheaply |

### What Makes This Data Hard
1. **Numbers live in tables.** The most valuable facts (revenue, R&D, EPS, guidance) are in financial statements. Fuzzy vector search over serialized table text returns *approximately* the right number — which in finance is just *wrong*. This is the core challenge and the project's main differentiator addresses it directly (see §5.4).
2. **Temporal ambiguity.** "Q3 revenue" could mean Q3 FY2022 or FY2023. Chunk metadata must preserve fiscal period, and the router must filter on it.
3. **Repetitive boilerplate.** Filings reuse legal language; dense-only retrieval over-retrieves near-duplicates and sparse-only over-retrieves keyword noise. Hybrid + rerank is needed.
4. **Cross-document queries.** "How did AAPL R&D change vs MSFT over 3 years?" needs multi-hop retrieval across 6+ documents plus a reasoning loop that recovers when the first pass is insufficient.

---

## 4. System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                 │
└───────────────────────────────────┬───────────────────────────────────┘
                                     ▼
┌───────────────────────────────────────────────────────────────────────┐
│  L1 EXACT-HASH CACHE (Redis, ~1ms)   →  HIT → return                    │
│  (safe by construction: exact string match only)                       │
└───────────────────────────────────┬───────────────────────────────────┘
                                     │ miss
                                     ▼
┌───────────────────────────────────────────────────────────────────────┐
│                  ADAPTIVE ROUTER  (LLM, structured output)              │
│   Classifies: NUMERIC | NARRATIVE | MULTI-HOP  + extracts entities,     │
│   fiscal periods (used as metadata pre-filters)                         │
└───────┬───────────────────────┬───────────────────────────┬────────────┘
        │ NUMERIC               │ NARRATIVE                  │ MULTI-HOP
        ▼                       ▼                            ▼
┌───────────────┐  ┌────────────────────────────┐  ┌──────────────────────┐
│ TEXT-TO-SQL   │  │ HYBRID RETRIEVAL (Qdrant)  │  │ AGENTIC LOOP          │
│ over DuckDB   │  │ BGE-M3 dense + sparse      │  │ decompose → parallel  │
│ financial-    │  │ → RRF → top-50             │  │ retrieve → sufficiency│
│ facts store   │  │ → BGE-reranker-v2 → top-5  │  │ check → re-retrieve   │
│ → exact rows  │  │ → parent-chunk expansion   │  │ on gaps → merge       │
└───────┬───────┘  └─────────────┬──────────────┘  └───────────┬──────────┘
        └────────────────────────┴─────────────────────────────┘
                                     ▼
┌───────────────────────────────────────────────────────────────────────┐
│                 GENERATION + GROUNDING  (LiteLLM, model-agnostic)       │
│  Every claim tagged to a source: chunk-id (narrative) or SQL row        │
│  (numeric). Faithfulness self-check (LLM-judge) → flag/regenerate.      │
└───────────────────────────────────┬───────────────────────────────────┘
                                     ▼
┌───────────────────────────────────────────────────────────────────────┐
│  RESPONSE (answer + citations)  →  write L1 cache  →  return            │
│  [optional] evaluated semantic cache (see §5.5)                         │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep Dive

### 5.1 Data Ingestion Pipeline

**Responsibilities:** Pull filings → extract text **and tables separately** → chunk text → extract table rows into structured facts → enrich with metadata → push text to Qdrant and facts to DuckDB.

```
SEC EDGAR
    │  edgartools → filing object
    │
    ├── Section text (Item 1, 7, 7A, 8...) → parent/child text chunks → Qdrant
    │
    └── Financial tables → typed rows → DuckDB "facts" table
            (company, ticker, form_type, fiscal_year, fiscal_quarter,
             line_item, value, unit, filing_date, source_chunk_id)
```

Every text chunk carries metadata:
```json
{
  "company": "AAPL", "ticker": "AAPL", "form_type": "10-K",
  "fiscal_year": 2023, "fiscal_quarter": null,
  "section": "Item 7 - MD&A",
  "chunk_id": "aapl_10k_2023_item7_chunk_042",
  "parent_chunk_id": "aapl_10k_2023_item7_section",
  "is_table": false, "filing_date": "2023-11-03"
}
```

**Why extract tables into a structured store instead of only serializing them to text:** serialized tables ("Revenue FY2023: $394.3B") are searchable but retrieval still returns them *approximately* — the model may pick a neighboring row or year. Numeric questions deserve **deterministic, exact** answers from typed rows, not a similarity guess. Tables are *also* serialized to text and indexed in Qdrant so they remain available to the narrative path.

### 5.2 Chunking Strategy

Use **hierarchical parent-child chunking**:

| Level | Size | Purpose |
|---|---|---|
| Document | Full filing | Never retrieved directly |
| Section | ~2000 tokens | Parent chunk — passed to LLM |
| Paragraph | ~200 tokens | Child chunk — indexed for retrieval precision |

Retrieve small (precise), generate with large (context-rich) — this addresses the "lost in the middle" problem.

**Do NOT use fixed-size character chunking.** It breaks mid-sentence, severs temporal context ("fiscal year 20…" gets cut), and is the single biggest cause of retrieval inconsistency in financial docs.

### 5.3 Indexing in Qdrant

```python
# Each text chunk becomes one point with two named vectors:
#   dense  — BGE-M3 dense embedding
#   sparse — BGE-M3 learned-sparse weights (stronger than raw BM25)
client.upsert(
    collection_name="finsight",
    points=[PointStruct(
        id=chunk_id,
        vector={"dense": dense_embedding, "sparse": sparse_vector},
        payload=metadata_dict,
    )],
)
```

**Embedding model — hardware-driven, two-tier strategy** (see §11 for the verified environment):

- **Default / online-serving model: `nomic-embed-text`** (137M, 274 MB). Runs comfortably on the local box (7.7 GB RAM, CPU), fast enough to embed a single query at request time. Sparse side via **BM25** (`rank_bm25`, already installed with edgartools). This is what the live demo serves with.
- **Quality-upgrade / batch model: `BGE-M3`** (dense + learned-sparse + multi-vector, single MIT model). Too heavy to serve online here, but ideal for a **one-time batch re-index on the university H100** (Jupyter). It becomes a *measured ablation row* (nomic vs BGE-M3 recall/nDCG), not a hard dependency.

This decouples **GPU-optimized batch indexing** from **CPU-optimized online serving** — the way production retrieval systems are actually built. The one constraint: the query must be embedded with the *same* model the corpus was indexed with, so if you promote BGE-M3 after the ablation, query-time embedding of the single query still runs locally (one short text on CPU is fine; it's the 400-doc batch that needs the GPU). Qdrant runs hybrid search natively in one query.

### 5.4 Retrieval Paths

The adaptive router (§5.6) sends each query down one of three paths.

**NUMERIC → text-to-SQL over DuckDB (the differentiator).**
The `facts` table is populated from **edgartools' XBRL-tagged financials** (`.financials.income_statement()` etc. → `to_dataframe()`, melted long), *not* fragile HTML parsing — so each row carries a canonical `us-gaap_*` concept, a human label, a typed value, and a source period. For "AAPL FY2023 R&D expense," a text-to-SQL step (a **Qwen-Coder** model works well here) generates a parameterized query against `facts` and returns the exact row(s). Deterministic, exact, and every number is traceable to a filing + line item. Scope the store to XBRL statement facts (high-confidence); footnote/MD&A tables that aren't cleanly tagged fall through to the narrative path — a boundary that maps naturally onto the NUMERIC-vs-NARRATIVE router split. *(Extraction validated in §11.)*

**NARRATIVE → hybrid retrieval + rerank.**
- **Stage 1 — Hybrid + RRF fusion:** merge dense and sparse rankings, retrieve **top-50** candidates. Fast (ANN, ~5–15ms).
  ```
  hybrid_score = 1/(k + rank_dense) + 1/(k + rank_sparse)
  ```
- **Stage 2 — Cross-encoder rerank:** score each (query, chunk) pair and return **top-5**. Use a **lightweight cross-encoder** (`bge-reranker-base` or a MiniLM cross-encoder) for online serving to stay in the latency budget on CPU; **`BGE-reranker-v2`** is the heavier, higher-quality option to evaluate in the H100 batch ablation. Typically +15–30% precision over first-stage alone.
- **Parent expansion:** retrieve the precise child chunk, pass its parent section to the LLM.

**Metadata pre-filtering (both paths):** the router parses company/fiscal period *before* searching, so "Apple's FY2023 revenue" filters `ticker=AAPL, fiscal_year=2023` — shrinking the search space and eliminating temporal cross-contamination.

### 5.5 Caching (honest scope)

**L1 — exact-hash cache (Redis).** MD5 of the normalized query string → exact match, ~1ms. Safe *by construction*: it can only return an answer to the identical question.

**L2 — semantic cache (optional, and evaluated, not assumed).** Semantic caching is tempting but *dangerous on financial facts*: "AAPL revenue FY23" and "AAPL revenue FY24" can exceed 0.92 cosine similarity, so a naive semantic cache will confidently return the **wrong year**. Therefore L2 is treated as an **experiment that must earn its place**:
- Guard it with metadata equality (same ticker + fiscal period) *in addition* to a high similarity threshold.
- Measure its **false-hit rate** on a set of near-miss temporal pairs. Ship it only if false-hit rate ≈ 0; otherwise cut it.
- Do **not** claim a cache hit rate — a single-user portfolio project has no realistic query distribution to measure one honestly.

Invalidation: when a new filing is ingested for a company, invalidate cached answers referencing that company.

### 5.6 Adaptive Router + Iterative Agentic Retrieval

A lightweight LLM call classifies each query and extracts entities/periods with structured output:

```python
ROUTER_PROMPT = """
Classify this financial query and extract structured filters.
- NUMERIC:   a specific figure/metric for a company + period (→ SQL path)
- NARRATIVE: qualitative/explanatory ("why did margins fall?") (→ hybrid retrieval)
- MULTI_HOP: comparison across companies/periods, or multi-part (→ agentic loop)

Query: {query}
Return JSON: {
  "type": "NUMERIC|NARRATIVE|MULTI_HOP",
  "entities": [...], "fiscal_periods": [...]
}
"""
```

**MULTI_HOP path — iterative loop (the 2026 bar, not one-shot decomposition):**
1. **Decompose** into sub-queries.
2. **Retrieve in parallel** (async) — each sub-query routes to NUMERIC or NARRATIVE.
3. **Sufficiency check:** does the gathered evidence actually answer every sub-part?
4. **Recover:** if not, reformulate and **re-retrieve** the gaps (bounded iterations).
5. **Merge + synthesize** with citations from all sources.

The recovery loop is the key differentiator from naive decomposition: naive pipelines fail at retrieval on a large fraction of complex queries precisely because they have *no second try*.

**Faithfulness self-check (all paths):** after generation, extract claims and verify each maps to a supporting chunk or SQL row. If groundedness is low, flag or regenerate.

### 5.7 Evaluation Infrastructure

This is the crown jewel — the thing that separates the project from a tutorial. **Every config change must be measurable.**

**Golden set (build this first):** ~100 labeled QA pairs:
- 40 numeric (revenue, EPS, R&D, guidance) — each with the exact expected value + source row
- 30 narrative/temporal (FY22 vs FY23 explanations)
- 30 multi-hop (cross-company, cross-year)

For each: `{query, type, expected_source_ids, expected_answer}`.

**Metrics tracked per config:**

| Metric | Measures | Target |
|---|---|---|
| `Context Recall@5` | Retrieval finds the right chunks | > 0.85 |
| `nDCG / MRR` | Best evidence ranked first | > 0.75 |
| `Numeric Accuracy` | Exact-match on financial figures (NUMERIC path) | > 0.95 |
| `Faithfulness` (LLM-judge) | Answer sticks to retrieved evidence | > 0.90 |
| `Temporal Accuracy` | Correct fiscal year/quarter | > 0.95 |
| `Citation Validity` | Every claim maps to a real source | > 0.95 |
| `P95 Latency` | End-to-end response time | < 2s |

**Ablation study (your key contribution):** run the golden set across each config and log all metrics. Each addition must beat the previous.

```
Config A: Dense-only retrieval (nomic-embed-text)
Config B: + Hybrid (dense + BM25 sparse) + RRF
Config C: + Cross-encoder rerank
Config D: + Numeric / text-to-SQL routing          ← finance differentiator isolated
Config E: + Agentic iterative retrieval (multi-hop) ← full system
Config F: (batch/H100) swap nomic → BGE-M3          ← embedding-quality upgrade, measured
```

Report Config D's gain **specifically on numeric queries** and Config E's gain **specifically on multi-hop queries** — isolating each differentiator. The cache is measured separately (latency + false-hit rate), not as a quality config.

---

## 6. Tech Stack

| Component | Tool | Why |
|---|---|---|
| Data ingestion | `edgartools` | Free, MIT, typed Python API, no key |
| Vector DB | `Qdrant` | Native hybrid search, free self-hosted, production-grade |
| Embeddings (online) | `nomic-embed-text` + BM25 | Light enough to serve on 7.7 GB / CPU; BM25 sparse via `rank_bm25` |
| Embeddings (batch upgrade) | `BGE-M3` | Higher-quality dense+sparse; run on H100 as a measured ablation (Config F) |
| Reranker | `bge-reranker-base` (online) / `BGE-reranker-v2` (batch) | Small cross-encoder for latency; heavy one for the ablation |
| Structured store | `DuckDB` | In-process SQL over **XBRL** financial facts; text-to-SQL target |
| Exact cache | `Redis` | Sub-ms, safe exact-match L1 |
| LLM | Open models (**Qwen / GLM / Kimi**) via **OpenRouter** + `LiteLLM` | Cheap (pennies/run), model-agnostic; Qwen-Coder for text-to-SQL |
| API server | `FastAPI` + `asyncio` | Async throughout; clean latency measurement |
| Eval framework | `RAGAS` + custom | Standard metrics + numeric-accuracy & citation-validity |
| Logging / latency | Structured logs + per-stage timing | Honest observability without Prometheus/Grafana overhead |
| Deployment | `Docker Compose` + hosted demo | First-class: a live, reproducible system |
| Demo UI | Lightweight web UI (Streamlit or a small React app) | A real, clickable demo — not an afterthought |

**Local infra cost: $0** (Qdrant, Redis, DuckDB, nomic all self-hosted). Only LLM API calls cost money, and with open models via OpenRouter that's cents, not dollars, per full eval run. The H100 (university Jupyter) is used only for optional batch re-indexing/ablation — never a runtime dependency.

---

## 7. Build Milestones

Leaner than a rigid week-by-week plan — reach a working, deployed demo fast, then deepen the two differentiators.

### M0 — Ingestion & Structured Extraction
- Repo scaffold; `edgartools` pull for 3 pilot companies (AAPL, MSFT, GOOGL).
- Section extraction → parent/child text chunks with metadata.
- **Table extraction → DuckDB `facts` table** (typed rows).
- Manually inspect 20–30 chunks and 20–30 facts rows — does temporal/numeric context survive?
- Qdrant up locally (`docker run -p 6333:6333 qdrant/qdrant`); index pilot corpus (dense first).
- **Deliverable:** pilot corpus indexed in Qdrant + facts in DuckDB; basic similarity search + a basic SQL lookup both work.

### M1 — Baseline Hybrid RAG + Eval Harness (the floor)
- FastAPI `/query` endpoint; hybrid retrieval (BGE-M3 dense + sparse) + RRF.
- LiteLLM generation with grounded chunk-id citations.
- Build the **golden set** (start ~50 pairs) and wire **RAGAS + custom metrics**; log P95 latency.
- Expand corpus to all 10 companies.
- **Deliverable:** Config A/B numbers. This is the measurable floor — everything after beats it.

### M2 — Rerank + Numeric Routing (the two differentiators)
- Add BGE-reranker-v2 (top-50 → top-5) and parent-chunk expansion.
- Add the **adaptive router** + **text-to-SQL numeric path** over DuckDB.
- Metadata pre-filtering from parsed entities/periods.
- Re-run the ablation; report Config C and Config D gains (D isolated on numeric queries).
- **Deliverable:** the retrieval + numeric-correctness story, with measured deltas.

### M3 — Agentic Iterative Layer (multi-hop)
- Query decomposer + parallel async sub-query retrieval.
- **Sufficiency check + bounded re-retrieval recovery loop**; context merger/synthesizer.
- Faithfulness self-check (LLM-judge) post-generation.
- Extend golden set with 30 multi-hop queries; re-run full ablation (Config E).
- **Deliverable:** full-system metrics; agentic layer's win isolated on multi-hop.

### M4 — Deploy & Portfolio Polish
- Dockerize; **Docker Compose** for the full stack (API + Qdrant + Redis + DuckDB).
- **Hosted live demo** + a real, clickable demo UI with a curated query set that shows the system at its best.
- `BENCHMARK.md` (the artifact interviewers ask about) + README with architecture diagram and one-command setup.
- Optional: 3-minute demo video.
- (Optional) evaluate the L2 semantic cache; ship only if false-hit rate ≈ 0.
- **Deliverable:** a reproducible, deployed project with a benchmark story.

---

## 8. Repo Structure

```
finsight-rag/
├── data/
│   ├── ingestion/
│   │   ├── edgar_fetcher.py       # edgartools wrapper, pull filings
│   │   ├── section_extractor.py   # extract Item 7, Item 8, etc.
│   │   ├── table_serializer.py    # table → structured text (for narrative path)
│   │   └── chunker.py             # hierarchical parent-child chunking
│   └── raw/                       # local filing cache (gitignored)
│
├── structured/                    # ← the numeric differentiator
│   ├── table_extractor.py         # tables → typed rows
│   ├── duckdb_store.py            # facts schema, load, query
│   └── text_to_sql.py            # NL numeric query → parameterized SQL
│
├── indexing/
│   ├── embedder.py                # BGE-M3 dense + sparse
│   └── qdrant_client.py           # collection setup, upsert, hybrid search
│
├── retrieval/
│   ├── hybrid_retriever.py        # dense + sparse + RRF fusion
│   ├── reranker.py                # BGE-reranker-v2 cross-encoder
│   ├── metadata_filter.py         # entities/periods → payload filters
│   └── context_preparer.py        # parent expansion, ordering
│
├── agent/
│   ├── router.py                  # NUMERIC | NARRATIVE | MULTI_HOP
│   ├── query_decomposer.py        # multi-hop → sub-queries
│   ├── iterative_loop.py          # sufficiency check + re-retrieval recovery
│   └── faithfulness_checker.py    # post-generation self-check (LLM-judge)
│
├── generation/
│   └── generator.py               # LiteLLM wrapper, grounded citation linker
│
├── cache/
│   └── l1_cache.py                # Redis exact-hash cache (L2 optional/evaluated)
│
├── eval/
│   ├── dataset/eval_set.json      # ~100 labeled QA pairs (typed)
│   ├── metrics.py                 # RAGAS + numeric-accuracy + citation-validity
│   ├── ablation.py                # run all configs, log results
│   └── results/                   # benchmark outputs (CSV)
│
├── api/
│   ├── main.py                    # FastAPI app, /query endpoint
│   └── logging.py                 # structured logs + per-stage latency
│
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml         # API + Qdrant + Redis
│   └── ui/                        # demo UI
│
├── scripts/{ingest_all.py, run_eval.py}
├── BENCHMARK.md                   # key artifact — ablation results
├── finsight_rag_architecture.md   # this document
└── README.md
```

---

## 9. What to Say in Interviews

**The one-liner:**
> "I built a retrieval system over SEC filings that routes numeric questions to a text-to-SQL path over extracted financial tables and narrative questions to hybrid vector retrieval, with an iterative agentic loop for multi-hop comparisons. Every architectural decision is justified by a measured ablation — including *why I deliberately did not* ship a naive semantic cache."

**System design answer:**
> "The core insight is that finance retrieval has two different failure modes. Numbers must be exact, so I extract tables into DuckDB and answer numeric queries with generated SQL — deterministic and citable to a line item. Narrative queries go through hybrid retrieval (BGE-M3 dense + learned-sparse), RRF fusion to top-50, then a BGE cross-encoder reranker to top-5. Multi-hop queries run an agentic loop that decomposes, retrieves in parallel, checks sufficiency, and re-retrieves on gaps — because the biggest cause of failure in complex RAG is one-shot retrieval with no recovery."

**The "RAG is dead / just use long context" answer:**
> "Stuffing 400 filings into context is dollars per query with no citation or numeric guarantees. Retrieval exists for cost, consistency, and verifiable citations. And for numbers specifically, similarity search returns *approximately* the right figure — which in finance is just wrong. That's why I route numeric questions to structured SQL, not a vector index."

**The "why no semantic cache" answer (shows judgment):**
> "I prototyped one and measured its false-hit rate. On financial facts, adjacent fiscal years embed too closely — a semantic cache confidently returns the wrong year. I kept a safe exact-match L1 cache and cut L2 unless metadata guards drove false hits to zero. Shipping a feature that returns wrong numbers to look sophisticated is the opposite of engineering."

---

## 10. What Already Exists vs What You're Building

| What exists | What you're doing differently |
|---|---|
| AlphaSense — keyword + semantic search over filings | Open, reproducible pipeline with a measured ablation |
| Hebbia — agentic multi-agent architecture | Lightweight adaptive router + *iterative* agentic retrieval, isolated eval on multi-hop |
| Generic RAG tutorials (LangChain/LlamaIndex) | Systems-first: numeric text-to-SQL path + every component justified by ablation |
| FinQA / FinanceBench datasets | Your own typed golden set on real, current SEC EDGAR filings |

**Your unique contribution:** a rigorous ablation showing exactly how much each component — hybrid retrieval, reranking, **numeric/text-to-SQL routing**, and **iterative agentic retrieval** — contributes to final performance, on a real financial corpus with temporal and numeric retrieval challenges, deployed as a live demo.

---

## 11. Build Readiness & Verified Environment (as of July 2026)

Everything below was empirically checked on the target machine before committing to the build — no assumption is load-bearing without evidence.

### Verified working ✅
| Assumption | Result |
|---|---|
| edgartools install | v5.40.1, Python 3.11.9 |
| **Numeric extraction (XBRL)** | `.financials.income_statement().to_dataframe()` → typed rows with canonical `us-gaap_*` concepts. AAPL FY23 net sales `$383,285M` and R&D `$29,915M`, MSFT FY23 revenue `$211,915M` — **exact matches** to known values |
| **Narrative extraction** | `tenk['Item 7']`, `.business`, `.risk_factors`, `.management_discussion` return clean **UTF-8** text (18K chars MD&A; apostrophe verified U+2019, zero replacement chars) |
| 10-Q support | 99 AAPL 10-Qs available; financials extract to a DataFrame |
| Docker | v28.3.2 installed → Qdrant + Redis viable |

### Hardware profile & consequences ⚠️
| Fact | Consequence baked into the design |
|---|---|
| **RAM 7.7 GB** | Don't run Docker services + heavy models at full tilt simultaneously; keep online models light (nomic) |
| **GPU: RTX 2050, 4 GB — but PyTorch is CPU-only** (`2.9.1+cpu`, `cuda.is_available()==False`) | Local serving is CPU. Enabling the local GPU needs a CUDA torch reinstall; 4 GB is small |
| **University H100 via Jupyter link (session-based, online-only)** | Batch-only asset: use for BGE-M3 corpus indexing + heavy ablations (Config F). **Never** the live-demo inference server |
| **Disk 28.5 GB free** | Watch model weights + Docker/WSL2 images + filing cache |

### Resolved decisions
- **LLM (dev):** **Groq free tier** via LiteLLM — verified working. Serves **`qwen/qwen3-32b`** (matches the open-model preference), plus `openai/gpt-oss-120b/20b`, `llama-3.3-70b-versatile`, `llama-4-scout`. Fast (LPU) and $0. Free-tier rate limits mean heavy Config A–F/F sweeps route to **OpenRouter or the H100** instead — a one-line model-string change through LiteLLM.
- **Embeddings:** **nomic-embed-text + BM25** for online serving; **BGE-M3** as an H100 batch ablation (Config F).
- **Reranker:** small cross-encoder online; BGE-reranker-v2 in the batch ablation.
- **Architecture principle confirmed:** decouple **GPU batch indexing** from **CPU online serving**.

### LLM layer — VERIFIED ✅
`smoke_test_llm.py` runs a router-style JSON call through LiteLLM → Groq/qwen3-32b and gets correct structured output (`MULTI_HOP`, entities, fiscal periods). Every architecture layer is now empirically green; nothing load-bearing is unverified. (Env: keys in gitignored `.env` via `python-dotenv`; Groq's Cloudflare needs a `User-Agent` header on the raw `/models` call; keep script stdout ASCII — the Windows console is cp1252.)

---

*Architecture version 2.1 — July 2026 (hardware-verified)*
