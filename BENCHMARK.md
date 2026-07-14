# FinSight RAG — Benchmark Results

The ablation artifact: each component earns its place by a measured delta. Numbers
are reproducible via `uv run python scripts/run_retrieval_eval.py`.

> **Scope note.** Results below are on the **3-company pilot** (AAPL, MSFT, GOOGL;
> latest 10-K; 499 child chunks) with a **10-query starter golden set** labeled by
> evidence-substring. Absolute numbers will move as the corpus and golden set grow
> and the chunker is refined — the point is the *relative* effect of each component.

## Retrieval ablation — 3-company pilot (531 chunks, k=5)

| Config | Retrieval | Recall@5 | Hit@5 | MRR | p50 |
|---|---|---|---|---|---|
| A | Dense-only (bge-small-en-v1.5) | 0.408 | 0.700 | 0.650 | 22 ms |
| B | Hybrid: dense + BM25, RRF | 0.475 | 0.700 | 0.633 | 25 ms |
| C | Hybrid + reranker (MiniLM-L-6) | **0.704** | **0.900** | 0.653 | 1835 ms |
| C+ | C + metadata pre-filter | 0.704 | 0.900 | **0.703** | 1875 ms |

At pilot scale: hybrid > dense (+16% Recall@5); reranker is the big lever (+48% Recall@5, Hit@5→0.90); metadata filter lifts MRR +7.7%.

## Retrieval ablation — 10-company corpus (2,308 chunks, k=5)

Golden set: `eval/golden/retrieval_10co.json` — 20 company-scoped queries (2 per filer).
Relevance = chunk contains the query's evidence phrase, matched on **word boundaries**.
Every evidence phrase occurs in exactly one filer, so cross-company hits are true negatives.

| Config | Recall@5 | norm | Hit@5 | MRR | p50 |
|---|---|---|---|---|---|
| A: dense-only | 0.235 | 0.353 | 0.650 | 0.588 | 25 ms |
| B: hybrid (RRF) | 0.267 | 0.401 | 0.750 | 0.550 | 33 ms |
| C: + cross-encoder reranker | 0.325 | 0.487 | 0.850 | 0.692 | 1889 ms |
| C+: + metadata pre-filter | **0.337** | **0.507** | **0.850** | **0.717** | 1901 ms |

**Recall@5 ceiling = 0.666.** Several evidence phrases match 14–20 chunks, so Recall@5 is
structurally capped; `norm` = Recall@5 ÷ ceiling. C+ reaches **51% of achievable recall**.

**Read:**
- Improvement is **monotonic**: A → B → C → C+ on Recall, normalized recall, and Hit@5.
- **Reranker is the largest single lever** (norm 0.401 → 0.487; Hit@5 0.75 → 0.85).
- **Metadata pre-filter** adds recall and lifts **MRR 0.692 → 0.717** by dropping cross-filer distractors.
- B dips MRR slightly: BM25 pulls more relevant chunks into top-5 (Hit@5 ↑) but can rank a different one first. The reranker then repairs ordering.

> **Correction (superseded result).** An earlier 10-company run reused the *3-company*
> golden set and appeared to show hybrid degrading below dense (0.293 vs 0.370). That was a
> **measurement artifact**: generic evidence phrases ("climate change") matched many filers,
> and naive substring matching mislabeled chunks (`'AWS'` matches inside `'laws'`). With a
> company-scoped golden set and word-boundary matching, hybrid beats dense as expected.
> Fixed in `resolve_relevant()`.

**Chunker fix (size-tails):** child sizes `1/128/–/534` → `13/127/172/218` words (min/median/p95/max).
On this 10-query set it was mostly hygiene (B recall held at 0.475) — cleaner chunks for citations, slightly faster — not itself a recall win.

## Generation
Grounded answers via Groq `qwen3-32b` (LiteLLM), every claim cited `[n]`→chunk-id;
citation-validity checked inline. Demo answers reconcile with independently-extracted XBRL figures. ~2 s/query.

## Config D — numeric / text-to-SQL path (numeric golden set)

Objective golden set: 51 pairs auto-generated from the facts table (expected value =
the reported figure). Metric = exact-match accuracy of the returned figure.

| Run | Model + pipeline | Coverage | Exact-match | SQL errors | latency (median) |
|---|---|---|---|---|---|
| 1 | qwen3-32b, un-hardened SQL, no repair | 168/168 | 164/168 = 0.976 | 2 | 1475 ms (mean) |
| **3 (current)** | **Gemini flash-lite, hardened SQL + repair loop** | **168/168** | **168/168 = 1.000** | **0** | **1236 ms** |

**Reported Config D result: `168/168 = 1.000`** — full coverage, on the model actually shipped
(Gemini flash-lite), with the hardened text-to-SQL path.

**Two honest caveats — this is not a clean model-vs-model comparison:**
1. Run 1's 0.976 was qwen3-32b on the *un-hardened* pipeline (before concept-grounding rules,
   fiscal-year injection, and the concept-aware repair loop). The jump to 1.000 is **part model,
   part pipeline hardening** — not "flash-lite is better than qwen3."
2. It's 1.000 on the **auto-generated golden set** (templated queries, objective values). Real
   free-form phrasing is harder (see the BlackRock/Netflix cases in DEVLOG) — this is not a claim
   of universal perfection, just that the SQL path resolves the objective set exactly.

*(An intermediate run on Groq hit the daily token cap at 74% coverage and was discarded — a biased
subset is not a benchmark. The current run is full-coverage thanks to a proper requests-per-minute
throttle that respects Gemini's 15 RPM.)*

What earned the jump from 0.976 → 1.000 (all validated on the failing pairs):
- **Concept grounding**: prompt forbids inventing concept ids not in the reference; the repair step
  re-shows the concept list and flags a concept that doesn't exist for the company.
- **Fiscal-year injection**: the parsed year is passed as a hard `period_fy IN (...)` hint (the model
  was occasionally guessing the wrong year).
- **Repair loop on empty results**: a zero-row answer is treated as a failure worth one correction.

Example: "What was Apple's total net sales for fiscal year 2023?" → routed NUMERIC →
`… concept='us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax' AND period_fy=2023
AND dimension_member IS NULL` → **$383.29B** (exact), ~1 s.

### Failure analysis (all 4 misses)
| Miss | Cause |
|---|---|
| `googl_rd_2025`, `crm_rd_2025` | LLM emitted **double-quoted** string literals (`concept = "us-gaap_…"`). In DuckDB double quotes denote an *identifier*, so it resolved as a column → `BinderException`. |
| `crm_gross_profit_2026`, `crm_operating_income_2026` | LLM used the **stale ticker `'SFDC'`** instead of `'CRM'` → 0 rows, no error, **silent** miss. |

**Fixes applied** (each verified on the failing pair; full confirmatory re-run pending):
1. Prompt states the SQL quoting rule explicitly.
2. Prompt injects the **already-resolved ticker(s)** — the model was never told which ticker to use, so it invented one.
3. **Repair retry**: on a SQL error *or an empty result set*, the failure is fed back for one corrected attempt. An empty result is treated as a failure worth retrying — a silent zero-row answer is as wrong as a crash.

All 4 previously-failing pairs now pass. **Measured figure above remains 164/168** until the full sweep is re-run.

### Run 2's single genuine miss: an accounting ambiguity, not a bug
`tsla_net_income_2025` — the model selected `us-gaap_ProfitLoss` (consolidated profit, **including**
noncontrolling interests) where the golden set expects `us-gaap_NetIncomeLoss` ($3,794M, attributable
to common stockholders; the two differ by Tesla's $61M NCI). Both are defensible readings of
"net income." Analysts mean the attributable-to-common figure (it is what EPS is computed from),
so the prompt now states that preference. Filed as a **domain-semantics** gap, not a model error.

## Schema linking (prompt-size optimization)

Sending every concept for a ticker costs ~1,700 prompt tokens. Embedding the question and keeping
the top-K most similar concepts was validated **offline, with zero LLM calls**, against the
ground-truth concept stored in each golden pair:

| K | concept-recall | ref tokens |
|---|---|---|
| 5 | 0.946 | 109 |
| 10 | 0.988 | 220 |
| 15 | 0.994 | 326 |
| **20** | **1.000** | 429 |
| 25 | 1.000 | 528 |
| all | 1.000 | 1683 |

Shipped at **K=25 + a always-included core-concept union** (revenue, net income, EPS, gross profit,
operating income, R&D). The margin over K=20 is deliberate: golden queries are *templated* and mirror
the concept labels, so `recall = 1.000` is optimistic for free-form analyst phrasing.

## Latency optimizations (serving)

Measured warm, per request, split into system (ours) vs LLM (provider, swappable):

| change | NUMERIC | NARRATIVE |
|---|---|---|
| baseline (pool=50, LLM router) | ~1.9 s | ~10.3 s |
| + reranker pool 50→20 | — | rerank ~2.4× faster, quality same/better (C+ Recall 0.337→0.350) |
| + rule-based fast-path router (skips router LLM on 94% of queries, 100% acc on 188 labeled) | ~1.65 s | **~2.6–3.1 s** |
| **+ L1 cache hit (Redis)** | **29 ms** | **29 ms** |

Cache hit is **~85× faster** than a miss and spends **zero LLM tokens**. Server-side lookup is 2 ms
(the ~29 ms is HTTP round-trip). Exact-match only — normalization (case/whitespace) hits, a
different question misses. Degrades to a no-op if Redis is down.

Honest note: LLM inference (~1–2 s) is the remaining floor and is provider-dependent, not system
compute — instrumented as `llm_ms` vs `system_ms` per request.

## Pending configs
- **D (re-run)** — confirm 168/168 after the repair loop.
- **E** — agentic iterative retrieval (measured on multi-hop queries)
- **F** — BGE-M3 embeddings + BGE-reranker-v2 (H100 batch) vs the light online stack

## Known levers (tracked in DEVLOG)
- Reranker latency on CPU (~1.8 s) → top-20 pool / GPU / batch.
- Expand golden set beyond 10 queries; add hand-audited relevance labels.
