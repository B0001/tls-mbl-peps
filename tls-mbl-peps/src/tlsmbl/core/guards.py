"""No-silent-NaN gate: INV-7 (ARCHITECTURE.md §3).

Every kernel's output tensors must be finite; a `nan`/`inf` raises immediately with
full provenance rather than propagating silently into an energy or gradient. Applied
as `@finite` on every kernel function starting in Phase 2 (peps/kernels/); this module
only defines the mechanism.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

import torch

_FnT = TypeVar("_FnT", bound=Callable[..., Any])

# Parameter names, if present on the wrapped function's signature, echoed into the
# NumericalCorruption message so a failure can be traced to (site, sweep, bond).
_PROVENANCE_KEYS = ("site", "sweep", "bond", "x", "y", "row")


class NumericalCorruption(RuntimeError):
    """INV-7: a kernel produced a non-finite output."""


def _iter_tensors(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_tensors(item)


def finite(fn: _FnT) -> _FnT:
    """INV-7 gate: raise `NumericalCorruption` if any output tensor is non-finite."""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        bad = [t for t in _iter_tensors(result) if not torch.isfinite(t).all()]
        if bad:
            try:
                bound = sig.bind_partial(*args, **kwargs).arguments
            except TypeError:
                bound = {}
            provenance = {k: bound[k] for k in _PROVENANCE_KEYS if k in bound}
            raise NumericalCorruption(
                f"INV-7: non-finite output from {fn.__qualname__} "
                f"({len(bad)} tensor(s) affected); provenance={provenance!r}"
            )
        return result

    return wrapper  # type: ignore[return-value]
