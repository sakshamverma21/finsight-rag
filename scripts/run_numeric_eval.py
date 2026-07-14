"""Config D: numeric accuracy of the text-to-SQL path on the numeric golden set.

For each pair, run the SQL path (ticker-filtered) and check the returned value
against the known figure within a tight tolerance. Reports exact-match accuracy,
SQL-error rate, and latency.

Robustness: every pair is checkpointed to a JSONL as soon as it completes, so a
crash (or a rate-limit death) never loses prior work. `--resume` skips pairs that
are already in the checkpoint. LLM calls are paced by the token throttle in
finsight.llm, so this is safe against free-tier providers -- it just takes longer.

Run:  uv run python scripts/run_numeric_eval.py              # full set
      uv run python scripts/run_numeric_eval.py --resume     # continue after a stop
      uv run python scripts/run_numeric_eval.py --limit 4    # quick check
"""

from __future__ import annotations

import argparse
import json
import time

from finsight.config import PROJECT_ROOT
from finsight.structured.text_to_sql import answer_numeric

GOLDEN = PROJECT_ROOT / "eval" / "golden" / "numeric_pilot.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
CHECKPOINT = RESULTS_DIR / "config_d_partial.jsonl"
RESULTS = RESULTS_DIR / "config_d.json"


def value_hit(ans, expected: float, rel: float = 0.005) -> bool:
    if ans.error or "value" not in ans.columns:
        return False
    vi = ans.columns.index("value")
    tol = rel * max(abs(expected), 1.0)
    return any(r[vi] is not None and abs(r[vi] - expected) <= tol for r in ans.rows)


def is_transient(error: str | None) -> bool:
    """Provider throttling — the pair was never really evaluated."""
    return bool(error) and ("RateLimitError" in error or "rate_limit" in error)


def is_daily_cap(error: str | None) -> bool:
    """Groq's per-DAY token budget. Retrying today is futile; stop the sweep."""
    return bool(error) and ("tokens per day" in error or "TPD" in error)


def load_checkpoint() -> dict[str, dict]:
    """Completed pairs only. Transient (rate-limited) pairs are NOT treated as done,
    so `--resume` retries them instead of silently accepting a throttling artifact."""
    if not CHECKPOINT.exists():
        return {}
    done = {}
    for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not is_transient(r.get("error")):
                done[r["id"]] = r
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="run only the first N pairs")
    ap.add_argument("--resume", action="store_true", help="skip pairs already checkpointed")
    args = ap.parse_args()

    pairs = json.loads(GOLDEN.read_text(encoding="utf-8"))["queries"]
    if args.limit:
        pairs = pairs[: args.limit]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = load_checkpoint() if args.resume else {}
    if not args.resume and CHECKPOINT.exists():
        CHECKPOINT.unlink()  # fresh run -> fresh checkpoint

    todo = [p for p in pairs if p["id"] not in done]
    n = len(pairs)
    print(f"Config D sweep: {n} pairs ({len(done)} already done, {len(todo)} to run)", flush=True)
    print("Throttled for free-tier TPM; expect ~1.5-2 calls/min.\n", flush=True)

    t_start = time.perf_counter()
    aborted = False
    with CHECKPOINT.open("a", encoding="utf-8") as ck:
        for i, p in enumerate(todo, 1):
            ans = answer_numeric(p["query"], tickers=[p["ticker"]])
            rec = {
                "id": p["id"], "query": p["query"], "ticker": p["ticker"],
                "expected_value": p["expected_value"], "hit": value_hit(ans, p["expected_value"]),
                "sql": ans.sql, "error": ans.error, "latency_ms": ans.latency_ms,
            }
            if is_daily_cap(ans.error):
                # Every remaining pair would fail identically and pollute the results.
                print(f"\n!! Daily token cap reached at {p['id']}. Stopping.", flush=True)
                print("   Re-run with --resume once the provider's daily budget resets.", flush=True)
                aborted = True
                break
            done[p["id"]] = rec
            ck.write(json.dumps(rec) + "\n")
            ck.flush()  # checkpoint immediately: a crash must not lose this pair
            status = "ok " if rec["hit"] else ("ERR" if rec["error"] else "MISS")
            acc = sum(r["hit"] for r in done.values()) / len(done)
            print(f"[{i:3}/{len(todo)}] {status} {p['id']:28} acc={acc:.3f}", flush=True)

    elapsed = time.perf_counter() - t_start
    records = [done[p["id"]] for p in pairs if p["id"] in done]
    if not records:
        print("No pairs completed.", flush=True)
        return
    correct = sum(r["hit"] for r in records)
    errors = sum(bool(r["error"]) for r in records)
    n = len(records)
    acc = correct / n
    lats = sorted(r["latency_ms"] for r in records)
    mean_lat = sum(lats) / n
    median_lat = lats[n // 2]
    coverage = n / len(pairs)

    RESULTS.write_text(json.dumps({
        "config": "D", "n": n, "total_pairs": len(pairs), "coverage": coverage,
        "complete": not aborted and n == len(pairs),
        "exact_match_accuracy": acc, "sql_errors": errors,
        "median_latency_ms": median_lat, "mean_latency_ms": mean_lat,
        "elapsed_s": elapsed, "records": records,
    }, indent=2), encoding="utf-8")

    print("\n=== Config D — numeric / text-to-SQL ===", flush=True)
    print(f"  pairs evaluated:      {n}/{len(pairs)}  (coverage {coverage:.1%})", flush=True)
    if aborted or n < len(pairs):
        print("  !! PARTIAL RUN — do not report as a full-coverage result", flush=True)
    print(f"  exact-match accuracy: {correct}/{n} = {acc:.3f}", flush=True)
    print(f"  SQL errors:           {errors}/{n}", flush=True)
    print(f"  latency median:       {median_lat:.0f} ms  (mean {mean_lat:.0f} ms, throttle excluded)", flush=True)
    print(f"  wall clock this run:  {elapsed / 60:.1f} min", flush=True)
    print(f"  results -> {RESULTS.relative_to(PROJECT_ROOT)}", flush=True)

    misses = [r for r in records if not r["hit"]]
    if misses:
        print(f"\n  {len(misses)} misses (first 12):", flush=True)
        for r in misses[:12]:
            print(f"    - {r['id']:28} {r['error'] or 'wrong/no value'}", flush=True)


if __name__ == "__main__":
    main()
