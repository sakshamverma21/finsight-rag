"""Single choke-point for LLM calls via LiteLLM.

Centralizes: model default, JSON mode, <think>-block stripping (qwen3 is a
reasoning model), retry/backoff, and a client-side **token-per-minute throttle**.

The throttle exists because free-tier providers (Groq: 6k TPM) meter tokens on a
rolling minute. Retries alone can't beat that ceiling on a long eval sweep — the
right fix is to pace our own egress. We estimate a call's cost before sending,
wait until it fits the window, then record the provider's *actual* usage.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import deque

from litellm import completion

from finsight.config import LLM_MODEL

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

_NUM_RETRIES = 4

# Stay under the provider's tokens-per-minute ceiling. Default suits Gemini free
# (250k TPM); for Groq free (6000 TPM) set FINSIGHT_TPM_LIMIT=5200.
_TPM_LIMIT = int(os.environ.get("FINSIGHT_TPM_LIMIT", "220000"))
# Requests-per-minute ceiling. Gemini flash-lite free = 15 RPM (the binding limit
# there); a token-only throttle ignores it and floods the endpoint. 0 = disabled.
_RPM_LIMIT = int(os.environ.get("FINSIGHT_RPM_LIMIT", "14"))
_WINDOW_SECONDS = 60.0
_COMPLETION_ALLOWANCE = 400  # tokens we assume the model will generate

# Sleeping to respect a rate limit is right for a BATCH sweep and wrong in a REQUEST
# path -- nobody waits 50s for an answer. Set FINSIGHT_THROTTLE_MAX_WAIT_S in the
# service so long waits fail fast as 429 instead of stalling the caller.
# 0 = unbounded (the default, appropriate for offline eval).
_MAX_WAIT_S = float(os.environ.get("FINSIGHT_THROTTLE_MAX_WAIT_S", "0"))


class ThrottleTimeout(Exception):
    """Serving the request would require sleeping longer than we're willing to."""

_lock = threading.Lock()
_window: deque[tuple[float, int]] = deque()  # (monotonic_ts, tokens)
_req_window: deque[float] = deque()  # monotonic timestamps of recent requests (for RPM)
_local = threading.local()  # per-thread: seconds slept by the throttle on last call


def last_throttle_wait_ms() -> float:
    """Throttle sleep for the most recent chat() on this thread.

    Callers subtract this from wall time so benchmarks report the *system's*
    latency, not an artifact of free-tier pacing.
    """
    return getattr(_local, "last_wait_s", 0.0) * 1000.0


def pop_throttle_wait_ms() -> float:
    """Total throttle sleep on this thread since the last pop, then reset.

    Lets a request handler report *pacing* separately from *work* — otherwise a
    service reports queueing delay as if it were compute, and you optimize the
    wrong thing.
    """
    total = getattr(_local, "total_wait_s", 0.0)
    _local.total_wait_s = 0.0
    return total * 1000.0


def pop_llm_ms() -> float:
    """Total time spent inside provider LLM calls on this thread, then reset.

    Separates *provider* latency (swappable — Groq vs Gemini vs ...) from *our*
    compute (retrieval, rerank, SQL exec). system_ms = total - llm_ms - throttle_ms.
    """
    total = getattr(_local, "llm_s", 0.0)
    _local.llm_s = 0.0
    return total * 1000.0


def _record_wait(seconds: float) -> None:
    _local.last_wait_s = seconds
    _local.total_wait_s = getattr(_local, "total_wait_s", 0.0) + seconds


def _record_llm(seconds: float) -> None:
    _local.llm_s = getattr(_local, "llm_s", 0.0) + seconds


def _prune(now: float) -> None:
    while _window and now - _window[0][0] > _WINDOW_SECONDS:
        _window.popleft()


def _tokens_in_window() -> int:
    return sum(t for _, t in _window)


def _estimate(messages: list[dict]) -> int:
    chars = sum(len(m.get("content") or "") for m in messages)
    return chars // 4 + _COMPLETION_ALLOWANCE


def _throttle(estimated: int) -> float:
    """Block until this call fits BOTH the TPM and RPM windows. Returns seconds slept.

    A token-only throttle floods providers whose binding limit is requests/min
    (Gemini flash-lite = 15 RPM), so we gate on both and reserve a request slot.
    """
    waited = 0.0
    while True:
        with _lock:
            now = time.monotonic()
            _prune(now)
            while _req_window and now - _req_window[0] > _WINDOW_SECONDS:
                _req_window.popleft()
            # `not _window` guard: a single oversized call must still go through.
            token_ok = not _window or _tokens_in_window() + estimated <= _TPM_LIMIT
            req_ok = not _RPM_LIMIT or len(_req_window) < _RPM_LIMIT
            if token_ok and req_ok:
                _req_window.append(now)  # reserve this request's slot
                return waited
            sleeps = []
            if not token_ok and _window:
                sleeps.append(_WINDOW_SECONDS - (now - _window[0][0]) + 0.1)
            if not req_ok and _req_window:
                sleeps.append(_WINDOW_SECONDS - (now - _req_window[0]) + 0.1)
            sleep_for = max(sleeps) if sleeps else 0.1
        naptime = max(sleep_for, 0.1)
        if _MAX_WAIT_S and waited + naptime > _MAX_WAIT_S:
            raise ThrottleTimeout(
                f"rate limit: would need to wait {waited + naptime:.1f}s "
                f"(cap {_MAX_WAIT_S:.1f}s)"
            )
        time.sleep(naptime)
        waited += naptime


def _record(tokens: int) -> None:
    with _lock:
        _window.append((time.monotonic(), tokens))


def chat(messages: list[dict], model: str = LLM_MODEL,
         json_mode: bool = False, temperature: float = 0.0) -> str:
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "num_retries": _NUM_RETRIES,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    estimated = _estimate(messages)
    _record_wait(_throttle(estimated))

    t_call = time.perf_counter()
    try:
        resp = completion(**kwargs)
    except Exception:
        _record_llm(time.perf_counter() - t_call)  # failed calls still cost latency
        _record(estimated)  # assume it cost us; stay conservative
        raise
    _record_llm(time.perf_counter() - t_call)

    usage = getattr(resp, "usage", None)
    _record(getattr(usage, "total_tokens", None) or estimated)
    return _THINK.sub("", resp.choices[0].message.content or "").strip()
