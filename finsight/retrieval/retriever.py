"""Dense / sparse / hybrid retrieval over a ChunkIndex.

Config A = dense-only; Config B = hybrid (dense + BM25 fused with RRF). Both are
exposed here so the eval harness can measure them against the same index.
"""

from __future__ import annotations

import os

import numpy as np

from finsight.indexing.embedder import Embedder
from finsight.retrieval.index import ChunkIndex, _tokenize
from finsight.retrieval.metadata_filter import extract_filters

RRF_K = 60
# Reranker candidate pool. Every candidate is a CPU cross-encoder forward pass, so
# this trades latency for recall almost linearly. 20 keeps the relevant chunk in
# range (hybrid usually surfaces it in the top-20) at ~2.5x less rerank time than 50.
RERANK_POOL = int(os.environ.get("FINSIGHT_RERANK_POOL", "20"))


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Fuse multiple ranked lists of chunk indices into one RRF-scored order."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self, index: ChunkIndex, embedder: Embedder, reranker=None):
        self.index = index
        self.embedder = embedder
        self.reranker = reranker  # optional; required for mode="rerank"

    def _mask(self, filters: dict | None) -> np.ndarray | None:
        """Boolean mask of eligible chunks; None if no effective filter.

        Falls back to no filter if the constraints would eliminate every chunk
        (an over-eager parse shouldn't zero out results).
        """
        if not filters:
            return None
        tickers = set(filters.get("tickers") or [])
        years = set(filters.get("fiscal_years") or [])
        if not tickers and not years:
            return None
        mask = np.ones(len(self.index.chunks), dtype=bool)
        if tickers:
            mask &= np.array([c["metadata"]["ticker"] in tickers for c in self.index.chunks])
        if years:
            mask &= np.array([c["metadata"]["fiscal_year"] in years for c in self.index.chunks])
        return mask if mask.any() else None

    def _resolve_mask(self, query: str, filters: dict | None, prefilter: bool) -> np.ndarray | None:
        if filters is None and prefilter:
            filters = extract_filters(query)
        return self._mask(filters)

    def _dense_rank(self, query: str, top_k: int, mask: np.ndarray | None = None) -> list[int]:
        q = self.embedder.embed_query(query)
        sims = self.index.embeddings @ q  # normalized vectors -> cosine similarity
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
        return list(np.argsort(-sims)[:top_k])

    def _sparse_rank(self, query: str, top_k: int, mask: np.ndarray | None = None) -> list[int]:
        scores = self.index.bm25.get_scores(_tokenize(query))
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        return list(np.argsort(-scores)[:top_k])

    def dense(self, query: str, top_k: int = 5, filters: dict | None = None, prefilter: bool = False) -> list[dict]:
        mask = self._resolve_mask(query, filters, prefilter)
        return [self.index.chunks[i] for i in self._dense_rank(query, top_k, mask)]

    def hybrid(self, query: str, top_k: int = 5, pool: int = 50, filters: dict | None = None, prefilter: bool = False) -> list[dict]:
        mask = self._resolve_mask(query, filters, prefilter)
        dense = self._dense_rank(query, pool, mask)
        sparse = self._sparse_rank(query, pool, mask)
        fused = reciprocal_rank_fusion([dense, sparse])[:top_k]
        return [self.index.chunks[i] for i in fused]

    def rerank(self, query: str, top_k: int = 5, pool: int = RERANK_POOL, filters: dict | None = None, prefilter: bool = False) -> list[dict]:
        if self.reranker is None:
            raise ValueError("mode='rerank' requires a reranker on the Retriever")
        candidates = self.hybrid(query, top_k=pool, pool=pool, filters=filters, prefilter=prefilter)
        return self.reranker.rerank(query, candidates, top_k)

    def retrieve(self, query: str, top_k: int = 5, mode: str = "hybrid",
                 filters: dict | None = None, prefilter: bool = False) -> list[dict]:
        if mode == "dense":
            return self.dense(query, top_k, filters, prefilter)
        if mode == "rerank":
            return self.rerank(query, top_k, filters=filters, prefilter=prefilter)
        return self.hybrid(query, top_k, filters=filters, prefilter=prefilter)
