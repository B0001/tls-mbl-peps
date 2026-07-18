"""Zarr persistence (ARCHITECTURE.md §5 layout; ADR-005: zarr for concurrent-writer
resumability). Each realization group is self-describing: a `stage` attr marks the
last completed pipeline stage, making `orchestrate.resume` a pure read."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import zarr

from tlsmbl.core.types import DisorderRealization, ModelParams, Site
from tlsmbl.peps.state import PEPSState

STAGES = ("sampled", "sdrg", "rung", "finalized")


def open_run(path: str | Path) -> zarr.Group:
    return zarr.open_group(str(path), mode="a")


def realization_group(root: zarr.Group, k: int) -> zarr.Group:
    return root.require_group(f"realizations/{k:05d}")


def get_stage(g: zarr.Group) -> str | None:
    return cast(str | None, g.attrs.get("stage"))


def read_attr_dict(g: zarr.Group, name: str) -> dict[str, Any]:
    """Typed boundary for JSON-valued attrs (report, observables, manifest, sdrg)."""
    return cast("dict[str, Any]", dict(cast("dict[str, Any]", g.attrs[name])))


def list_realizations(root: zarr.Group) -> list[str]:
    reals = root.get("realizations")
    if reals is None:
        return []
    return sorted(cast(zarr.Group, reals))


def _write_array(g: zarr.Group, name: str, data: np.ndarray) -> None:
    if name in g:
        del g[name]
    g.create_array(name, shape=data.shape, dtype=data.dtype)[:] = data


def write_disorder(g: zarr.Group, real: DisorderRealization) -> None:
    d = g.require_group("disorder")
    _write_array(d, "eps", real.eps)
    _write_array(d, "delta", real.delta)
    _write_array(d, "h_mf", real.h_mf)
    idx = np.array([[a[0], a[1], b[0], b[1]] for (a, b) in real.J], dtype=np.int64)
    vals = np.array(list(real.J.values()))
    _write_array(d, "J_indices", idx.reshape(-1, 4))
    _write_array(d, "J_values", vals)
    d.attrs["params"] = json.loads(real.params.model_dump_json())
    d.attrs["rng_fingerprint"] = real.rng_fingerprint
    g.attrs["stage"] = "sampled"


def subgroup(g: zarr.Group, name: str) -> zarr.Group:
    return cast(zarr.Group, g[name])


def read_disorder(g: zarr.Group) -> DisorderRealization:
    d = subgroup(g, "disorder")
    params = ModelParams.model_validate(d.attrs["params"])
    idx = np.asarray(d["J_indices"])
    vals = np.asarray(d["J_values"])
    J: dict[tuple[Site, Site], float] = {
        ((int(r[0]), int(r[1])), (int(r[2]), int(r[3]))): float(v)
        for r, v in zip(idx, vals)
    }
    return DisorderRealization(
        params=params,
        eps=np.asarray(d["eps"]),
        delta=np.asarray(d["delta"]),
        J=J,
        h_mf=np.asarray(d["h_mf"]),
        rng_fingerprint=str(d.attrs["rng_fingerprint"]),
    )


def write_sdrg(g: zarr.Group, summary: dict[str, Any]) -> None:
    g.attrs["sdrg"] = summary
    g.attrs["stage"] = "sdrg"


def write_rung(g: zarr.Group, D: int, state: PEPSState, record: dict[str, Any]) -> None:
    p = g.require_group(f"peps/D{D}")
    for y in range(state.L):
        for x in range(state.L):
            _write_array(p, f"t_{y}_{x}", state.tensors[y][x].detach().numpy())
    p.attrs["record"] = record
    p.attrs["L"] = state.L
    p.attrs["D"] = D
    rungs = rungs_done(g)
    if D not in rungs:
        rungs.append(D)
    g.attrs["rungs_done"] = rungs
    g.attrs["stage"] = "rung"


def read_rung(g: zarr.Group, D: int, dtype: torch.dtype) -> PEPSState:
    p = subgroup(g, f"peps/D{D}")
    L = int(cast(int, p.attrs["L"]))
    tensors = [
        [torch.from_numpy(np.asarray(p[f"t_{y}_{x}"])).to(dtype) for x in range(L)]
        for y in range(L)
    ]
    return PEPSState(tensors=tensors, L=L, D=int(cast(int, p.attrs["D"])))


def rungs_done(g: zarr.Group) -> list[int]:
    return [int(d) for d in cast("list[int]", g.attrs.get("rungs_done", []))]


def write_final(
    g: zarr.Group, report: dict[str, Any], observables: dict[str, Any]
) -> None:
    g.attrs["report"] = report
    g.attrs["observables"] = observables
    g.attrs["stage"] = "finalized"


def write_manifest(root: zarr.Group, manifest: dict[str, Any]) -> None:
    root.attrs["manifest"] = manifest
