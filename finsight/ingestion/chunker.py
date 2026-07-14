"""Hierarchical parent-child chunking.

Parent = full section (context passed to the LLM later).
Child  = ~200-token units (indexed for retrieval precision).

Children are built by packing "units" up to a target size, where a unit is a
paragraph — or, if a paragraph is oversized, its sentence-split pieces. Tiny
trailing fragments (stray headers, one-liners) merge into the previous child so
we never emit 1-word chunks. Token counts are approximated (words / 0.75).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_CHILD_TOKENS = 200
MAX_UNIT_TOKENS = 300  # a single paragraph larger than this gets sentence-split
MIN_CHILD_TOKENS = 25  # a finished child smaller than this merges into the previous

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    parent_chunk_id: str
    text: str
    level: str  # "section" (parent) or "paragraph" (child)
    metadata: dict


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75) if text else 0


def _slug(item: str) -> str:
    return item.lower().replace(" ", "")


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _pack(pieces: list[str], target: int) -> list[str]:
    """Greedily pack text pieces into groups up to `target` tokens."""
    groups: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for piece in pieces:
        pt = _approx_tokens(piece)
        if buf and buf_tokens + pt > target:
            groups.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(piece)
        buf_tokens += pt
    if buf:
        groups.append(" ".join(buf))
    return groups


def _units(text: str) -> list[str]:
    """Paragraphs, with oversized paragraphs sentence-split into ~target pieces."""
    units: list[str] = []
    for para in _split_paragraphs(text):
        if _approx_tokens(para) <= MAX_UNIT_TOKENS:
            units.append(para)
        else:
            units.extend(_pack(_SENTENCE.split(para), TARGET_CHILD_TOKENS))
    return units


def _child_texts(text: str) -> list[str]:
    groups = _pack(_units(text), TARGET_CHILD_TOKENS)
    merged: list[str] = []
    for g in groups:
        if merged and _approx_tokens(g) < MIN_CHILD_TOKENS:
            merged[-1] = merged[-1] + "\n\n" + g
        else:
            merged.append(g)
    return merged


def chunk_section(item: str, name: str, text: str, base_meta: dict) -> list[Chunk]:
    """Turn one section into a parent chunk + N child chunks."""
    fy = base_meta["fiscal_year"]
    form_slug = base_meta["form"].replace("-", "").lower()
    prefix = f"{base_meta['ticker'].lower()}_{form_slug}_{fy}_{_slug(item)}"
    parent_id = f"{prefix}_section"
    section_label = f"{item} - {name}"

    def meta(level: str) -> dict:
        return {**base_meta, "section": section_label, "is_table": False, "level": level}

    chunks: list[Chunk] = [Chunk(parent_id, parent_id, text, "section", meta("section"))]
    for idx, ctext in enumerate(_child_texts(text)):
        chunks.append(Chunk(f"{prefix}_chunk_{idx:03d}", parent_id, ctext, "paragraph", meta("paragraph")))
    return chunks
