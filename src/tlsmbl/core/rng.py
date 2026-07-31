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
    """The four independent RNG streams a single realization consumes.

    ORDER IS FROZEN. `SeedSequence.spawn` derives children by index, so appending a
    stream leaves the earlier ones bit-identical, but *reordering* or inserting one would
    silently change every golden fixture, the frozen baselines in `prototypes/baselines/`
    and T-DET. New streams go on the end, never in the middle.
    """

    disorder: np.random.Generator  # model/sampling.py: eps, delta, J
    torch_init_seed: int  # optimize/init.py: PEPS random init
    torch_sketch_seed: int  # kernels/rsvd.py: Gaussian test matrices
    tail_seed: int  # model/hartree.py: lazy r > R_c couplings (§7.4, NR-5)


def realization_streams(master_seed: int, realization_index: int) -> RealizationStreams:
    """Derive the (disorder, torch_init, torch_sketch, tail) streams for one realization.

    `spawn(4)`, not `spawn(3)`: the tail stream was appended when the §7.4 Hartree loop
    landed. Children 0-2 are unchanged by the extension (verified across seeds in
    tests/unit/test_hartree_loop.py::test_existing_three_streams_are_unchanged), so no
    stored artifact moves.
    """
    base = realization_seed_sequence(master_seed, realization_index)
    disorder_seq, torch_init_seq, torch_sketch_seq, tail_seq = base.spawn(4)
    return RealizationStreams(
        disorder=np.random.default_rng(disorder_seq),
        torch_init_seed=int(torch_init_seq.generate_state(1, dtype=np.uint64)[0]),
        torch_sketch_seed=int(torch_sketch_seq.generate_state(1, dtype=np.uint64)[0]),
        tail_seed=int(tail_seq.generate_state(1, dtype=np.uint64)[0]),
    )


def enable_deterministic_mode() -> None:
    """Certification-run switch: torch.use_deterministic_algorithms(True) (INV-6)."""
    import torch

    torch.use_deterministic_algorithms(True)
