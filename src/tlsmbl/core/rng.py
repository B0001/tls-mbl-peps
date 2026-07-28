"""Reproducibility gate: INV-6 (ARCHITECTURE.md §3).

Every realization is keyed `(master_seed, realization_index)`. All RNG streams --
numpy for disorder sampling, torch for init and sketch randomness -- are derived from
that key via `np.random.SeedSequence.spawn()`, never touched directly. Determinism
requires building a *fresh* `SeedSequence(master_seed)` and spawning from it in one
shot per call: `SeedSequence` tracks how many children it has already spawned, so
reusing a live object across calls would silently break bit-reproducibility (T-DET).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MissingSeedError(RuntimeError):
    """INV-6: the build must refuse to run without a master seed."""


def require_master_seed(master_seed: int | None) -> int:
    """Gate: raise unless a master seed is present. Never default one silently."""
    if master_seed is None:
        raise MissingSeedError(
            "INV-6: master_seed is required; refusing to run unseeded."
        )
    return master_seed


def realization_seed_sequence(
    master_seed: int, realization_index: int
) -> np.random.SeedSequence:
    """The seed sequence for one realization, keyed (master_seed, realization_index).

    Deterministic: a fresh root SeedSequence is constructed and spawned once, so the
    result does not depend on call order or prior spawns from other realizations.
    """
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(realization_index + 1)
    return children[realization_index]


@dataclass(frozen=True)
class RealizationStreams:
    """The three independent RNG streams a single realization consumes."""

    disorder: np.random.Generator  # model/sampling.py: eps, delta, J
    torch_init_seed: int  # optimize/init.py: PEPS random init
    torch_sketch_seed: int  # kernels/rsvd.py: Gaussian test matrices


def realization_streams(master_seed: int, realization_index: int) -> RealizationStreams:
    """Derive the (disorder, torch_init, torch_sketch) streams for one realization."""
    base = realization_seed_sequence(master_seed, realization_index)
    disorder_seq, torch_init_seq, torch_sketch_seq = base.spawn(3)
    return RealizationStreams(
        disorder=np.random.default_rng(disorder_seq),
        torch_init_seed=int(torch_init_seq.generate_state(1, dtype=np.uint64)[0]),
        torch_sketch_seed=int(torch_sketch_seq.generate_state(1, dtype=np.uint64)[0]),
    )


def enable_deterministic_mode() -> None:
    """Certification-run switch: torch.use_deterministic_algorithms(True) (INV-6)."""
    import torch

    torch.use_deterministic_algorithms(True)
