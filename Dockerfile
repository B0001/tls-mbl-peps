# syntax=docker/dockerfile:1
#
# Batch image for the TLS-MBL PEPS solver. Runs as a Kubernetes Job, not a
# service -- `tlsmbl run <config>` is run-to-completion and writes a .zarr.
#
#   docker build -t tls-mbl-peps:0.0.1 .
#   docker run --rm -v "$PWD/runs:/app/runs" tls-mbl-peps:0.0.1 \
#       tlsmbl run configs/smoke.yaml
#
# Python is 3.14 to match .python-version and uv.lock. It is also the ceiling:
# per CLAUDE.md, torch 2.13.0 publishes cp310-cp314 only, so 3.15 has no wheel.

# --- Stage 1: build the environment ---
FROM python:3.14-slim-bookworm AS builder

# uv from the official distroless image -- pinned, no curl|sh bootstrap.
COPY --from=ghcr.io/astral-sh/uv:0.6.9 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.14

WORKDIR /app

# Dependency layer first so editing src/ does not invalidate the solve.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Then the source, then the project itself into the same venv. LICENSE.md is
# required, not decorative: pyproject sets `license-files = ["LICENSE.md"]` and
# setuptools fails the build if it is missing.
COPY src/ ./src/
COPY LICENSE.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# torch comes from the CPU index, NOT default PyPI. Per CLAUDE.md the default
# linux wheel is CUDA-linked: it pulls ~2GB of nvidia-* shared libraries that a
# CPU-only Job will never load. This is a separate `uv pip install` because it
# is deliberately outside the lock (CLAUDE.md documents it as a manual step).
#
# It MUST come after the last `uv sync`. Being outside the lock is exactly what
# makes `uv sync` prune it: installing torch earlier builds a green image whose
# `tlsmbl run` dies at `import torch` in orchestrate.py. Same trap as the local
# `uv sync --extra dev` one CLAUDE.md warns about, one layer down.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# --- Stage 2: runtime ---
FROM python:3.14-slim-bookworm AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src/ ./src/
COPY pyproject.toml ./
# Configs are baked in so the image is runnable standalone; the Job overrides
# them with a ConfigMap mount at /etc/tlsmbl for anything non-default.
COPY configs/ ./configs/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Pin the math libraries to one thread per process. Parallelism here is
# process-level: run.workers in the config forks N realization workers, and
# without this each one would also spawn NCPU BLAS threads, oversubscribing the
# pod's CPU limit by workers x NCPU and slowing the run down rather than up.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Non-root. Kubernetes pins runAsUser/fsGroup to this same 10001 so the runs/
# PVC is writable; keep the two in sync.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app

# Configs write to a relative `run.out` (e.g. runs/smoke.zarr), resolved against
# the working directory, so this is where the PVC mounts.
RUN mkdir -p /app/runs && chown 10001:10001 /app/runs

USER 10001:10001

ENTRYPOINT ["tlsmbl"]
CMD ["--help"]
