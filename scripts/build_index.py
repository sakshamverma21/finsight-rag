"""Build the vector index once and persist it as a deployable artifact.

Embedding the corpus is *build-time* work, not *request-time* work. A service that
rebuilds its index on every start pays that cost on every deploy, restart, crash-loop,
and autoscale event. We do it once, write index.npz, and the API just loads it.

Run:  uv run python scripts/build_index.py
"""

from __future__ import annotations

import time

from finsight.config import CHUNKS_PATH, INDEX_PATH, PROJECT_ROOT
from finsight.indexing.embedder import Embedder
from finsight.retrieval.index import ChunkIndex


def main() -> None:
    t0 = time.perf_counter()
    embedder = Embedder()
    t_model = time.perf_counter() - t0

    t0 = time.perf_counter()
    index = ChunkIndex.build(CHUNKS_PATH, embedder)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    index.save(INDEX_PATH)
    t_save = time.perf_counter() - t0

    t0 = time.perf_counter()
    reloaded = ChunkIndex.load(INDEX_PATH)
    t_load = time.perf_counter() - t0

    size_mb = INDEX_PATH.stat().st_size / 1e6
    print(f"embedder load : {t_model:6.2f}s")
    print(f"index build   : {t_build:6.2f}s   ({len(index)} chunks, dim={embedder.dim})")
    print(f"index save    : {t_save:6.2f}s   ({size_mb:.1f} MB -> {INDEX_PATH.relative_to(PROJECT_ROOT)})")
    print(f"index LOAD    : {t_load:6.2f}s   ({len(reloaded)} chunks)")
    print()
    print(f"=> cold start goes from ~{t_model + t_build:.1f}s (build) to ~{t_model + t_load:.1f}s (load)")


if __name__ == "__main__":
    main()
