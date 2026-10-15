"""FastAPI service for FinSight.

Three deployment ideas made concrete (see DEPLOYMENT_GUIDE.md):

1. **Startup vs request time.** Models and the prebuilt index load ONCE in `lifespan`.
   Rebuilding the index instead would cost ~150s on every restart.
2. **Liveness vs readiness.** `/health` says the process is alive; `/ready` says the
   models finished loading. For ~7s after boot we are alive but not ready.
3. **Concurrency.** `/query` is a *sync* `def`, so FastAPI runs it in a threadpool.
   The reranker is CPU-bound (~1.8s of forward passes) and would otherwise block the
   event loop and stall every other request.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).parent / "static"

from finsight.agent.orchestrator import Orchestrator
from finsight.agent.router import resolve_company
from finsight.cache.l1_cache import L1Cache
from finsight.config import CHUNKS_PATH, DB_PATH, INDEX_PATH, LLM_MODEL
from finsight.generation.generator import Generator
from finsight.indexing.embedder import Embedder
from finsight.ingestion.pipeline import ingest_company
from finsight.llm import ThrottleTimeout, pop_llm_ms, pop_throttle_wait_ms
from finsight.retrieval.index import ChunkIndex
from finsight.retrieval.metadata_filter import extract_filters, register_company
from finsight.retrieval.reranker import Reranker
from finsight.retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("finsight")

# Public-demo guardrail: cap how many companies can be ingested on demand, so a
# visitor can't exhaust RAM or hammer SEC EDGAR under our contact identity.
MAX_COMPANIES = int(os.environ.get("FINSIGHT_MAX_COMPANIES", "30"))

state: dict = {}


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class SourceOut(BaseModel):
    n: int
    chunk_id: str
    ticker: str
    section: str
    cited: bool
    text: str = ""
    fiscal_year: int | None = None
    filing_date: str | None = None


class QueryResponse(BaseModel):
    question: str
    route: str
    answer: str
    latency_ms: float        # total wall time the caller experienced
    system_ms: float         # OUR compute: retrieval + rerank + SQL exec + orchestration
    llm_ms: float            # provider LLM inference (swappable: Groq / Gemini / ...)
    throttle_ms: float       # time we deliberately slept to respect rate limits
    cached: bool = False     # served from the L1 cache (no LLM, no retrieval)
    pending_ticker: str | None = None  # set when route == "INGESTING": poll /ingest/{ticker}
    sources: list[SourceOut] = []
    sql: str | None = None
    data: list[dict] = []    # structured rows behind a NUMERIC answer (for tables/charts)
    trend: list[dict] = []   # multi-year series for a single figure (deterministic, no LLM)
    subquestions: list[str] = []  # MULTI_HOP: the agentic decomposition
    steps: list[dict] = []        # MULTI_HOP: per-hop retrieval trace (query, n_sources)
    citations_valid: bool | None = None


def _upstream_status(err: Exception) -> tuple[int, str]:
    """Map LLM-provider failures onto honest HTTP semantics.

    A provider rate limit is not a bug in this service -- returning 500 would tell
    the caller (and any autoscaler) the wrong thing. 429 means 'retry later'.
    """
    if isinstance(err, ThrottleTimeout):
        return 429, f"rate limited: {err}"
    text = str(err)
    if "RateLimitError" in type(err).__name__ or "rate_limit" in text:
        return 429, "LLM provider rate limit reached; retry shortly"
    return 502, "LLM provider request failed"


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    embedder = Embedder()

    if INDEX_PATH.exists():
        index = ChunkIndex.load(INDEX_PATH)
        source = "prebuilt"
    else:  # fall back, but this is a ~150s penalty -- run scripts/build_index.py
        log.warning(json.dumps({"event": "index_missing", "path": str(INDEX_PATH)}))
        index = ChunkIndex.build(CHUNKS_PATH, embedder)
        source = "rebuilt"

    retriever = Retriever(index, embedder, reranker=Reranker())
    state["embedder"] = embedder
    state["index"] = index
    state["orchestrator"] = Orchestrator(Generator(retriever))
    state["cache"] = L1Cache()  # no-op if REDIS_URL unset/unreachable
    state["chunks"] = len(index)
    # On-demand ingestion mutates the shared index at runtime; guard it.
    state["index_lock"] = threading.Lock()
    state["ingest_jobs"] = {}  # ticker -> "ingesting" | "ready" | "error: ..."
    state["companies"] = {c["metadata"]["ticker"] for c in index.chunks}
    state["ready"] = True

    log.info(json.dumps({
        "event": "startup", "cold_start_s": round(time.perf_counter() - t0, 2),
        "index": source, "chunks": len(index), "llm": LLM_MODEL,
        "cache": "on" if state["cache"].enabled else "off",
    }))
    yield
    state.clear()


app = FastAPI(title="FinSight RAG", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the research UI from the same origin as the API."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up. Failing this should trigger a restart."""
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: models loaded. Failing this should stop traffic, NOT restart."""
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="models still loading")
    return {"status": "ready", "chunks": state["chunks"], "llm": LLM_MODEL}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, background: BackgroundTasks) -> QueryResponse:
    # sync `def` on purpose: FastAPI offloads it to a threadpool so the CPU-bound
    # reranker doesn't block the event loop.
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="models still loading")

    # L1 cache: exact-match hit returns the prior answer with no LLM / no retrieval.
    cache = state["cache"]
    t_cache = time.perf_counter()
    hit = cache.get(req.question)
    if hit is not None:
        lookup_ms = (time.perf_counter() - t_cache) * 1000
        log.info(json.dumps({"event": "query", "route": hit["route"],
                             "cached": True, "latency_ms": round(lookup_ms)}))
        return QueryResponse(**hit, cached=True, latency_ms=lookup_ms,
                             system_ms=lookup_ms, llm_ms=0.0, throttle_ms=0.0)

    pop_throttle_wait_ms(); pop_llm_ms()  # clear any residue on this pooled thread
    t0 = time.perf_counter()

    # Unknown-company detection: if no company we know is named, try to resolve the
    # intended ticker. If it's a real filer we don't have yet, start ingesting it and
    # tell the client to poll — the "pull the numbers automatically" flow.
    if not extract_filters(req.question)["tickers"]:
        resolved = resolve_company(req.question)
        if resolved and resolved["ticker"] not in state["companies"]:
            tk = resolved["ticker"]
            if len(state["companies"]) >= MAX_COMPANIES:
                total_ms = (time.perf_counter() - t0) * 1000
                return QueryResponse(
                    question=req.question, route="NARRATIVE",
                    answer=(f"{resolved['name']} isn't in this demo's data, and it's reached its "
                            f"{MAX_COMPANIES}-company limit for on-demand additions. Try one of the "
                            f"companies already loaded."),
                    latency_ms=total_ms, system_ms=max(total_ms - pop_llm_ms(), 0.0),
                    llm_ms=0.0, throttle_ms=pop_throttle_wait_ms())
            if state["ingest_jobs"].get(tk) != "ingesting":
                state["ingest_jobs"][tk] = "ingesting"
                background.add_task(_ingest_job, tk)
            total_ms = (time.perf_counter() - t0) * 1000
            llm_ms = pop_llm_ms()
            log.info(json.dumps({"event": "auto_ingest", "ticker": tk, "name": resolved["name"]}))
            return QueryResponse(
                question=req.question, route="INGESTING", pending_ticker=tk,
                answer=f"\U0001F4C4 {resolved['name']} isn't in our data yet — pulling its "
                       f"SEC filing now (~30s), then I'll answer.",
                latency_ms=total_ms, llm_ms=llm_ms,
                system_ms=max(total_ms - llm_ms, 0.0), throttle_ms=pop_throttle_wait_ms(),
            )

    try:
        # Serialize with on-demand ingestion so a concurrent index commit can't be
        # read mid-splice. Coarse but correct for a single-user demo; production would
        # snapshot the index or use a vector DB with real concurrent writes.
        with state["index_lock"]:
            routed = state["orchestrator"].answer(req.question)
    except Exception as e:  # noqa: BLE001 - upstream provider failures
        status, detail = _upstream_status(e)
        # Truncating an error at 200 chars once hid *which* rate limit was hit
        # (per-minute vs per-day) -- the only part that mattered. Keep the tail.
        log.warning(json.dumps({"event": "upstream_error", "status": status, "error": str(e)[:600]}))
        raise HTTPException(status_code=status, detail=detail) from e

    total_ms = (time.perf_counter() - t0) * 1000
    throttle_ms = pop_throttle_wait_ms()
    llm_ms = pop_llm_ms()
    common = {
        "question": req.question, "route": routed.route,
        "latency_ms": total_ms, "throttle_ms": throttle_ms, "llm_ms": llm_ms,
        # our compute is whatever's left after provider inference and pacing
        "system_ms": max(total_ms - throttle_ms - llm_ms, 0.0),
    }
    payload = routed.payload

    if routed.route == "NUMERIC":
        # Expose the SQL result rows so the UI can render a table / chart.
        data = [dict(zip(payload.columns, r)) for r in payload.rows] if payload.rows else []
        resp = QueryResponse(**common, answer=payload.text, sql=payload.sql or None, data=data,
                             trend=getattr(payload, "trend", []) or [])
    else:  # NARRATIVE or MULTI_HOP — both are grounded, cited answers
        resp = QueryResponse(
            **common, answer=payload.text,
            citations_valid=bool(payload.citation_valid),
            sources=[
                SourceOut(n=s.n, chunk_id=s.chunk_id, ticker=s.ticker,
                          section=s.section, cited=s.n in payload.cited,
                          text=s.text, fiscal_year=s.fiscal_year,
                          filing_date=s.filing_date)
                for s in payload.sources
            ],
            subquestions=getattr(payload, "subquestions", []),
            steps=getattr(payload, "steps", []),
        )

    # Cache only successful answers — never a failed/empty numeric result, or we'd
    # serve a stale "no matching figure" forever and block a retry from succeeding.
    numeric_failed = routed.route == "NUMERIC" and (routed.payload.error or not routed.payload.rows)
    if not numeric_failed:
        cache.set(req.question, resp.model_dump(exclude={
            "latency_ms", "system_ms", "llm_ms", "throttle_ms", "cached"}))

    log.info(json.dumps({
        "event": "query", "route": routed.route, "cached": False, "latency_ms": round(total_ms),
        "system_ms": round(resp.system_ms), "llm_ms": round(llm_ms), "throttle_ms": round(throttle_ms),
        "n_sources": len(resp.sources), "citations_valid": resp.citations_valid,
    }))
    return resp


# --- On-demand ingestion: add a company outside the fixed corpus at runtime -----

class IngestRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=6)


def _ingest_job(ticker: str) -> None:
    """Fetch + extract + embed a company, then splice it into the live index.

    Runs in a background thread (~30-50s). The slow parts (SEC fetch, embedding)
    happen OUTSIDE the index lock; only the quick commit + BM25 re-fit holds it.
    """
    try:
        # Slow, unlocked: SEC fetch + parse + embed. Touches no shared state.
        chunks, facts, name = ingest_company(ticker)
        children = [c for c in chunks if c["level"] == "paragraph"]
        embeddings = state["embedder"].embed_documents([c["text"] for c in children])

        # Fast, locked: commit facts (DB write) and index together, so no query's
        # read-only DuckDB connection overlaps this write, and no read sees a
        # half-spliced index.
        with state["index_lock"]:
            con = duckdb.connect(str(DB_PATH))
            from finsight.structured import duckdb_store as store
            store.load_facts(con, facts)
            con.close()
            state["index"].commit(children, embeddings)
        register_company(ticker, name)
        state["companies"].add(ticker)

        # Persist so the ingested company survives a restart.
        with CHUNKS_PATH.open("a", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        state["index"].save(INDEX_PATH)

        state["ingest_jobs"][ticker] = "ready"
        log.info(json.dumps({"event": "ingest", "ticker": ticker, "status": "ready",
                             "children": len(children), "name": name}))
    except Exception as e:  # noqa: BLE001
        state["ingest_jobs"][ticker] = f"error: {type(e).__name__}: {e}"[:200]
        log.warning(json.dumps({"event": "ingest", "ticker": ticker, "status": "error", "error": str(e)[:200]}))


@app.post("/ingest", status_code=202)
def ingest(req: IngestRequest, background: BackgroundTasks) -> dict:
    """Kick off ingestion of a company not in the corpus. Poll GET /ingest/{ticker}."""
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="models still loading")
    ticker = req.ticker.strip().upper()
    if ticker in state["companies"]:
        return {"ticker": ticker, "status": "ready", "note": "already available"}
    if state["ingest_jobs"].get(ticker) == "ingesting":
        return {"ticker": ticker, "status": "ingesting"}
    if len(state["companies"]) >= MAX_COMPANIES:
        raise HTTPException(status_code=429,
                            detail=f"demo limited to {MAX_COMPANIES} companies; can't add more")
    state["ingest_jobs"][ticker] = "ingesting"
    background.add_task(_ingest_job, ticker)
    return {"ticker": ticker, "status": "ingesting", "note": "~30-50s; poll /ingest/{ticker}"}


@app.get("/ingest/{ticker}")
def ingest_status(ticker: str) -> dict:
    ticker = ticker.strip().upper()
    if ticker in state["companies"]:
        return {"ticker": ticker, "status": "ready"}
    return {"ticker": ticker, "status": state["ingest_jobs"].get(ticker, "unknown")}


@app.get("/companies")
def companies() -> dict:
    return {"companies": sorted(state["companies"])}


@app.get("/coverage")
def coverage() -> dict:
    """Available fiscal periods for each indexed filer, for the comparison UI."""
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="models still loading")
    with state["index_lock"]:
        filing_years: dict[str, int] = {}
        for chunk in state["index"].chunks:
            meta = chunk["metadata"]
            ticker, year = meta["ticker"], meta.get("fiscal_year")
            if year is not None:
                filing_years[ticker] = max(year, filing_years.get(ticker, year))

        years: dict[str, list[int]] = {}
        if DB_PATH.exists():
            con = duckdb.connect(str(DB_PATH), read_only=True)
            try:
                for ticker, year in con.execute(
                    "SELECT DISTINCT ticker, period_fy FROM facts "
                    "WHERE period_fy IS NOT NULL ORDER BY ticker, period_fy DESC"
                ).fetchall():
                    years.setdefault(ticker, []).append(year)
            finally:
                con.close()

        return {"companies": [
            {"ticker": ticker, "filing_year": filing_years.get(ticker),
             "years": years.get(ticker, [])}
            for ticker in sorted(state["companies"])
        ]}
