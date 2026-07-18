"""Kernel D-scaling microbenchmark (D3 deliverable; §14.9 T-PERF).

Times exact vs sketched truncation on the steady-state E-5 operand shape
(chi * D^2) square with chi = D^2, on a localized-phase-like fast-decaying
spectrum. Reports per-D medians and fitted log-log exponents; the CI gate is
the exponent gap (theory 2.0; prototype measured 3.30 across D=4..8).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from tlsmbl.kernels.rsvd import SketchedSVD
from tlsmbl.kernels.svd import ExactSVD


@dataclass
class BenchPoint:
    D: int
    n: int  # operand side chi * D^2 = D^4
    exact_s: float
    sketched_s: float
    gate_passed: bool


@dataclass
class BenchResult:
    points: list[BenchPoint]
    exact_exponent: float
    sketched_exponent: float

    @property
    def exponent_gap(self) -> float:
        return self.exact_exponent - self.sketched_exponent


def _operand(n: int, decay: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    w = a * np.exp(-decay * np.arange(1, n + 1))[None, :]
    return torch.from_numpy(w).to(torch.complex128)


def _median_time(fn: object, reps: int) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()  # type: ignore[operator]
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_kernel_bench(
    Ds: list[int], *, decay: float = 0.5, reps: int = 5, seed: int = 20260716
) -> BenchResult:
    """decay=0.5 mirrors ADR-009's validated localized-phase spectrum e^(-0.5k):
    the regime the solver operates in and the INV-3 gate certifies. Slow-decay
    operands legitimately fall back (gate_passed False) and time the exact path."""
    points = []
    for D in Ds:
        chi = D * D
        n = chi * D * D
        W = _operand(n, decay, seed + D)
        exact = ExactSVD()
        sketched = SketchedSVD(seed=seed + D)
        t_exact = _median_time(lambda: exact.truncate(W, chi), reps)
        res = sketched.truncate(W, chi)
        t_sketch = _median_time(lambda: sketched.truncate(W, chi), reps)
        points.append(
            BenchPoint(
                D=D,
                n=n,
                exact_s=t_exact,
                sketched_s=t_sketch,
                gate_passed=res.posterior_err is not None,
            )
        )
    if len(points) >= 2:
        logD = np.array([math.log(p.D) for p in points])
        ex = float(np.polyfit(logD, [math.log(p.exact_s) for p in points], 1)[0])
        sk = float(np.polyfit(logD, [math.log(p.sketched_s) for p in points], 1)[0])
    else:
        ex = sk = float("nan")
    return BenchResult(points=points, exact_exponent=ex, sketched_exponent=sk)
