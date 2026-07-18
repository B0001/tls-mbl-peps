"""Double-layer tensors, normative einsums E-1 / E-2 (ARCHITECTURE.md §6).

Bra/ket legs fused per leg with the ket index fastest; copy the strings, do not
re-derive them.
"""

from __future__ import annotations

import torch


def double_layer(A: torch.Tensor, op: torch.Tensor | None = None) -> torch.Tensor:
    """E-1 (op None) / E-2 (one-site operator acting on the ket). Output legs
    ((l lbar), (u ubar), (r rbar), (d dbar))."""
    if op is None:
        t = torch.einsum("plurd,pLURD->lLuUrRdD", A, A.conj())
    else:
        t = torch.einsum("pq,qlurd,pLURD->lLuUrRdD", op.to(A.dtype), A, A.conj())
    s = A.shape
    return t.reshape(s[1] ** 2, s[2] ** 2, s[3] ** 2, s[4] ** 2)
