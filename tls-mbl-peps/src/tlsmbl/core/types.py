"""Dtype threading (CLAUDE.md non-negotiables; ARCHITECTURE.md §4 dtype policy).

`complex128` everywhere on certification paths; `complex64` allowed only in `bench`
mode. Every tensor-producing call takes a `TensorSpec` and reads `.dtype` from it --
no module hardcodes `torch.complex128`/`torch.complex64` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Mode = Literal["certified", "bench"]

_DTYPE_BY_MODE: dict[Mode, torch.dtype] = {
    "certified": torch.complex128,
    "bench": torch.complex64,
}
_REAL_DTYPE_BY_COMPLEX: dict[torch.dtype, torch.dtype] = {
    torch.complex128: torch.float64,
    torch.complex64: torch.float32,
}


@dataclass(frozen=True)
class TensorSpec:
    """The dtype/device a run's tensors are constructed with. One instance per run,
    threaded down through every constructor -- never rebuilt ad hoc mid-pipeline."""

    mode: Mode = "certified"
    device: str = "cpu"

    @property
    def dtype(self) -> torch.dtype:
        return _DTYPE_BY_MODE[self.mode]

    @property
    def real_dtype(self) -> torch.dtype:
        return _REAL_DTYPE_BY_COMPLEX[self.dtype]
