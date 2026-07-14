"""Cross-encoder reranker (Config C): rescore (query, chunk) pairs, keep top-k.

Default is a small MiniLM cross-encoder — fast on CPU, the right choice for
online serving on this hardware. BGE-reranker-v2 is the heavier, higher-quality
alternative reserved for the H100 batch ablation (architecture §5.4).
"""

from __future__ import annotations

import os

import numpy as np
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = os.environ.get("FINSIGHT_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


class Reranker:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks:
            return []
        scores = self.model.predict([(query, c["text"]) for c in chunks])
        order = np.argsort(-np.asarray(scores))
        return [chunks[i] for i in order[:top_k]]
