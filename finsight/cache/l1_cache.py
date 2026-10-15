"""L1 exact-match answer cache (Redis).

Key = hash of the normalized question. A hit returns the full prior answer without
touching the LLM or the retrieval stack — turning a ~2-10s request into a <5ms one,
and (crucially on a free tier) spending zero tokens on repeats.

Safe by construction: exact string match only, so it can never return one question's
answer for a different question (the trap that killed the semantic-cache idea).

Degrades gracefully: if Redis is unreachable the cache is a silent no-op. A cache is
an optimization, never a dependency — the service must answer with Redis down.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

DEFAULT_TTL_SECONDS = int(os.environ.get("FINSIGHT_CACHE_TTL", str(24 * 3600)))
_WS = re.compile(r"\s+")


def _key(question: str) -> str:
    norm = _WS.sub(" ", question.strip().lower())
    # Response v2 includes filing excerpts. Old cached responses lack that evidence.
    return "finsight:l1:v2:" + hashlib.md5(norm.encode()).hexdigest()


class L1Cache:
    def __init__(self, url: str | None = None, ttl: int = DEFAULT_TTL_SECONDS):
        self.ttl = ttl
        self.client = None
        url = url or os.environ.get("REDIS_URL")
        if not url:
            return
        try:
            import redis
            client = redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
            client.ping()  # fail fast if unreachable
            self.client = client
        except Exception:
            self.client = None  # no cache, but the service still runs

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def get(self, question: str) -> dict | None:
        if not self.client:
            return None
        try:
            raw = self.client.get(_key(question))
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set(self, question: str, payload: dict) -> None:
        if not self.client:
            return
        try:
            self.client.set(_key(question), json.dumps(payload), ex=self.ttl)
        except Exception:
            pass
