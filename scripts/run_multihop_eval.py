"""Config E — agentic multi-hop retrieval vs single-pass narrative.

A multi-hop question names two companies, so a *complete* answer must draw evidence
from BOTH. Single-pass retrieval fills its top-k from whichever company the combined
query vector is most similar to, starving the other. Config E decomposes the question,
retrieves per company, checks sufficiency, and re-retrieves the gap.

Three metrics, because the obvious one saturates:
  1. naive coverage   - standard single-pass (one query vector, NO metadata prefilter,
     top-k rerank over the whole corpus): does it even surface both companies?
  2. company coverage - single-pass WITH prefilter vs multi-hop: fraction of required
     companies present. Prefilter alone lifts single-pass here, so this saturates at 1.0.
  3. evidence balance - min per-company source count. Single-pass fills its top-k
     lopsidedly (the reranker favours whichever company the blended query is closer to),
     starving the comparison; per-hop retrieval returns balanced evidence. THIS is the
     discriminating metric under production settings.

Run:  uv run python scripts/run_multihop_eval.py
"""

from __future__ import annotations

import argparse
import json
import time

from finsight.agent.multihop import answer_multihop
from finsight.config import CHUNKS_PATH, INDEX_PATH, PROJECT_ROOT
from finsight.generation.generator import Generator
from finsight.indexing.embedder import Embedder
from finsight.retrieval.index import ChunkIndex
from finsight.retrieval.reranker import Reranker
from finsight.retrieval.retriever import Retriever

GOLDEN_DIR = PROJECT_ROOT / "eval" / "golden"
RESULTS = PROJECT_ROOT / "eval" / "results" / "config_e.json"
TOP_K = 8  # single-pass gets the same source budget the multi-hop pool caps at


def coverage(found_tickers, required) -> float:
    req = set(required)
    return len(req & set(found_tickers)) / len(req) if req else 0.0


def minority(found_tickers, required) -> int:
    """Sources from the LEAST-represented required company — the comparison's weak leg."""
    return min(found_tickers.count(t) for t in required) if required else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="multihop.json")
    args = ap.parse_args()

    spec = json.loads((GOLDEN_DIR / args.golden).read_text(encoding="utf-8"))
    queries = spec["queries"]

    embedder = Embedder()
    index = ChunkIndex.load(INDEX_PATH) if INDEX_PATH.exists() else ChunkIndex.build(CHUNKS_PATH, embedder)
    retriever = Retriever(index, embedder, reranker=Reranker())
    generator = Generator(retriever)

    print(f"Golden: {args.golden} | {len(queries)} multi-hop queries | corpus {len(index)} chunks | k={TOP_K}\n")
    print(f"  {'query id':26} {'required':11} {'naive':>6} {'sp cov/bal':>11} {'mh cov/bal':>11}  hops")
    print("  " + "-" * 76)

    rows = []
    naive_cov, sp_cov, mh_cov, sp_bal, mh_bal, sp_lat, mh_lat = [], [], [], [], [], [], []
    for q in queries:
        req = q["companies"]

        # 1. naive single-pass: one query vector, NO metadata prefilter (standard RAG).
        naive_src = [c["metadata"]["ticker"]
                     for c in retriever.retrieve(q["query"], top_k=TOP_K, mode="rerank", prefilter=False)]

        # 2. production single-pass narrative (prefilter on).
        t0 = time.perf_counter()
        sp = generator.answer(q["query"], top_k=TOP_K)
        sp_ms = (time.perf_counter() - t0) * 1000
        sp_src = [s.ticker for s in sp.sources]

        # 3. Config E — agentic multi-hop.
        t0 = time.perf_counter()
        mh = answer_multihop(q["query"], retriever)
        mh_ms = (time.perf_counter() - t0) * 1000
        mh_src = [s.ticker for s in mh.sources]

        r = {
            "id": q["id"], "companies": req,
            "naive_coverage": coverage(naive_src, req),
            "single_coverage": coverage(sp_src, req), "single_minority": minority(sp_src, req),
            "multi_coverage": coverage(mh_src, req), "multi_minority": minority(mh_src, req),
            "single_ms": round(sp_ms), "multi_ms": round(mh_ms),
            "hops": len(mh.steps), "recovery": any(s.get("recovery") for s in mh.steps),
        }
        rows.append(r)
        naive_cov.append(r["naive_coverage"])
        sp_cov.append(r["single_coverage"]); mh_cov.append(r["multi_coverage"])
        sp_bal.append(r["single_minority"]); mh_bal.append(r["multi_minority"])
        sp_lat.append(sp_ms); mh_lat.append(mh_ms)
        print(f"  {q['id']:26} {'+'.join(req):11} {r['naive_coverage']:>6.2f} "
              f"{r['single_coverage']:.2f}/{r['single_minority']}".rjust(11)
              + f"{r['multi_coverage']:.2f}/{r['multi_minority']}".rjust(12)
              + f"  {r['hops']}{'+rec' if r['recovery'] else ''}")

    n = len(rows)
    avg = lambda a: sum(a) / n if n else 0.0
    med = lambda a: sorted(a)[n // 2] if n else 0.0
    print("\n" + "=" * 74)
    print(f"  {'metric':38} {'single-pass':>12} {'multi-hop':>12}")
    print("  " + "-" * 64)
    print(f"  {'naive coverage (no prefilter)':38} {avg(naive_cov):>12.3f} {'n/a':>12}")
    print(f"  {'company coverage (prefilter on)':38} {avg(sp_cov):>12.3f} {avg(mh_cov):>12.3f}")
    print(f"  {'evidence balance (min src / company)':38} {avg(sp_bal):>12.2f} {avg(mh_bal):>12.2f}")
    print(f"  {'median latency (ms)':38} {med(sp_lat):>12.0f} {med(mh_lat):>12.0f}")
    print("=" * 74)

    summary = {
        "n": n,
        "naive_single_pass": {"coverage": avg(naive_cov)},
        "single_pass": {"coverage": avg(sp_cov), "evidence_balance": avg(sp_bal), "median_ms": med(sp_lat)},
        "multi_hop": {"coverage": avg(mh_cov), "evidence_balance": avg(mh_bal), "median_ms": med(mh_lat)},
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"queries": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
