"""In-memory chunk index: dense embeddings + BM25, built from the chunks JSONL.

Deliberately infra-free for the pilot corpus (a few thousand chunks). Behind the
`Retriever` interface so a Qdrant-backed index can replace it when corpus scale
justifies ANN — without touching retrieval logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from finsight.indexing.embedder import Embedder


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class ChunkIndex:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray, bm25: BM25Okapi):
        self.chunks = chunks
        self.embeddings = embeddings  # [N, d], L2-normalized
        self.bm25 = bm25

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def build(cls, jsonl_path: Path, embedder: Embedder, level: str = "paragraph") -> "ChunkIndex":
        records = [json.loads(line) for line in jsonl_path.open(encoding="utf-8")]
        chunks = [r for r in records if r["level"] == level]
        texts = [r["text"] for r in chunks]
        embeddings = embedder.embed_documents(texts)
        bm25 = BM25Okapi([_tokenize(t) for t in texts])
        return cls(chunks, embeddings, bm25)

    def add(self, chunks: list[dict], embedder: Embedder, level: str = "paragraph") -> int:
        """Convenience (scripts/tests): embed the new children and commit them."""
        new = [c for c in chunks if c["level"] == level]
        if not new:
            return 0
        self.commit(new, embedder.embed_documents([c["text"] for c in new]))
        return len(new)

    def commit(self, new_children: list[dict], new_embeddings: np.ndarray) -> None:
        """Splice pre-embedded chunks into the live index and re-fit BM25.

        Kept separate from embedding so a caller can do the slow embed *outside* a
        lock and only hold it for this quick splice. rank_bm25 has no incremental
        add, so we re-fit over the whole corpus (~seconds) — the moment an in-memory
        index starts wishing it were a vector DB with real online writes.
        """
        self.chunks.extend(new_children)
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            chunks=np.array([json.dumps(c) for c in self.chunks], dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> "ChunkIndex":
        data = np.load(path, allow_pickle=True)
        chunks = [json.loads(c) for c in data["chunks"]]
        embeddings = data["embeddings"]
        bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
        return cls(chunks, embeddings, bm25)
