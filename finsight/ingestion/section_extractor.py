"""Extract narrative sections (Item 1, 1A, 7, 7A ...) from a 10-K object."""

from __future__ import annotations

from finsight.config import SECTIONS_10K


def extract_sections(tenk) -> dict[str, str]:
    """Return {item -> clean text} for the configured sections that exist."""
    out: dict[str, str] = {}
    available = set(getattr(tenk, "items", []))
    for item in SECTIONS_10K:
        if item not in available:
            continue
        try:
            text = str(tenk[item]).strip()
        except Exception:
            continue
        if text:
            out[item] = text
    return out
