"""Dense embedder for online serving.

Default model is a small, fast, CPU-friendly encoder (bge-small-en-v1.5) chosen
for this hardware (7.7 GB RAM, CPU torch). It's swappable via FINSIGHT_EMBED_MODEL;
BGE-M3 is the H100 batch-upgrade alternative (see architecture §5.3 / Config F).
"""

from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.environ.get("FINSIGHT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# bge-v1.5 retrieval works best with a short instruction on the *query* side only.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, query_instruction: str = _QUERY_INSTRUCTION):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.model = SentenceTransformer(model_name, device="cpu")

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(
            self.model.encode(self.query_instruction + text, normalize_embeddings=True),
            dtype=np.float32,
        )
