# FinSight API image — multi-stage build.
#
# WHY TWO STAGES (guide §10): installing dependencies leaves ~1.5 GB of downloaded
# wheels in uv's cache. In a single stage that cache is baked into the image next
# to the venv it produced — we'd ship the wheels AND the unpacked copy. So we
# install in a "builder" stage and copy ONLY the finished venv into a clean
# "runtime" stage. The wheel cache stays behind in the builder and never ships.
#
# Both stages use the SAME base image so the venv's interpreter path stays valid
# after the copy (a venv records the absolute path of its base Python).

# ---------- builder: install deps, the project, and pre-fetch models ----------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/app/.cache/huggingface

# Deps first (cached unless pyproject.toml / uv.lock change) — so a code edit does
# not trigger a 3-minute torch reinstall.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# App source, then install the project itself (editable link into /app/finsight).
COPY finsight ./finsight
COPY README.md ./
RUN uv sync --frozen --no-dev

# Bake model weights so startup needs no network. --no-sync: don't let `uv run`
# re-add the dev group we deliberately excluded.
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# ---------- runtime: only what serving needs ----------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    FINSIGHT_THROTTLE_MAX_WAIT_S=8

# The installed venv (no wheel cache) and the pre-fetched models.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.cache/huggingface /app/.cache/huggingface
# The project is installed editable, so its source must exist at the same path.
COPY finsight ./finsight
# Build-time artifacts (index MUST match the embedding model or retrieval is silently wrong).
COPY data/index.npz data/chunks.jsonl data/finsight.duckdb ./data/

EXPOSE 8000

# Liveness for the runtime (no curl in slim; use the venv python now on PATH).
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Bind 0.0.0.0 — inside a container 127.0.0.1 is unreachable from the host.
CMD ["uvicorn", "finsight.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
