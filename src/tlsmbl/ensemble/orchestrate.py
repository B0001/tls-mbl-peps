"""Ensemble orchestration (ARCHITECTURE.md §11): L1 parallelism over realizations,
checkpoint after every pipeline stage, resume by reading stage markers.

Pipeline per realization: sample -> SDRG (INV-8 auto-bypass) -> D-ladder with
per-rung checkpoints -> finalize (INV-2 + INV-5) -> observables -> store.
Crashes lose at most one ladder rung.
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from tlsmbl.core.config import Config
from tlsmbl.core.rng import realization_streams
from tlsmbl.core.types import DisorderRealization, HamiltonianTerms, ModelParams, TensorSpec
from tlsmbl.io import store
from tlsmbl.io.manifest import build_manifest
from tlsmbl.kernels.interface import TruncationBackend
from tlsmbl.kernels.rsvd import SketchedSVD
from tlsmbl.kernels.svd import ExactSVD
from tlsmbl.model.hamiltonian import build_terms
from tlsmbl.model.hartree import tail_bound, tail_certified
from tlsmbl.model.sampling import sample_realization
from tlsmbl.observables.static import measure_static
from tlsmbl.optimize.finalize import chi_extrapolation_check
from tlsmbl.optimize.init import product_init
from tlsmbl.optimize.lbfgs_driver import optimize_lbfgs
from tlsmbl.peps.energy import energy_certified
from tlsmbl.sdrg.transform import SDRGResult, sdrg_transform


def _backend(cfg: Config, sketch_seed: int) -> TruncationBackend:
    k = cfg.kernels
    if k.backend == "exact":
        return ExactSVD(eps_F=k.eps_F)
    return SketchedSVD(
        seed=sketch_seed,
        oversample=k.oversample,
        power_iters=k.power_iters,
        eta=k.eta,
        c_gate=k.c_gate,
        probes=k.probes,
        eps_F=k.eps_F,
        # INV-3 failure action (ADR-016). One backend per realization, so the
        # backend-scoped disable IS the "disable sketching for the realization"
        # the invariant calls for.
        disable_rate=k.fallback_disable_rate,
    )


def run_realization(
    cfg: Config,
    k: int,
    out: str | Path,
    *,
    _fail_after_stage: str | None = None,  # test hook: simulate a crash
) -> str:
    """Runs (or resumes) realization k; returns the final stage reached."""
    torch.set_num_threads(max(1, torch.get_num_threads() // max(1, cfg.run.workers)))
    root = store.open_run(out)
    g = store.realization_group(root, k)
    stage = store.get_stage(g)
    if stage == "finalized":
        return "finalized"

    streams = realization_streams(cfg.run.master_seed, k)
    spec = TensorSpec()
    backend = _backend(cfg, streams.torch_sketch_seed)

    def checkpoint(reached: str) -> bool:
        if _fail_after_stage == reached:
            raise RuntimeError(f"injected failure after stage {reached!r}")
        return True

    # --- stage: sample (cheap; regenerated deterministically if missing) ---
    if stage is None:
        params = ModelParams(
            L=cfg.model.L,
            delta_min=cfg.model.delta_min,
            g_J=cfg.model.g_J,
            R_c=cfg.model.R_c,
            polaron_kappa=cfg.model.polaron_kappa,
            seed_realization=k,
        )
        real = sample_realization(params, streams.disorder)
        store.write_disorder(g, real)
        checkpoint("sampled")
    else:
        real = store.read_disorder(g)
    terms = build_terms(real)

    # --- stage: SDRG (Stage A, quarantined) ---
    if cfg.sdrg.enabled:
        sdrg: SDRGResult | None = sdrg_transform(
            terms,
            omega_stop=cfg.sdrg.omega_stop,
            f_max=cfg.sdrg.f_max,
            keep_first_order=cfg.sdrg.keep_first_order,
            tau_sdrg=cfg.sdrg.tau_sdrg,
        )
        assert sdrg is not None
        if store.get_stage(g) in (None, "sampled"):
            store.write_sdrg(
                g,
                {
                    "bypassed": sdrg.bypassed,
                    "n_ops": 0 if sdrg.circuit is None else len(sdrg.circuit.ops),
                    "ledger_total": sdrg.ledger.total,
                    "E0": sdrg.E0,
                    "omega_sequence": sdrg.omega_sequence,
                },
            )
            checkpoint("sdrg")
        terms_opt = sdrg.terms
        circuit = sdrg.circuit
        e_offset = sdrg.E0
    else:
        sdrg = None
        terms_opt = terms
        circuit = None
        e_offset = 0.0

    # --- stage: D-ladder with per-rung checkpoints ---
    done = store.rungs_done(g)
    state = None
    seqs = np.random.SeedSequence(streams.torch_init_seed).spawn(len(cfg.peps.ladder))
    if done:
        last = max(d for d in done if d in cfg.peps.ladder)
        state = store.read_rung(g, last, spec.dtype)
    res = None
    for i, D in enumerate(cfg.peps.ladder):
        if D in done:
            continue
        chi = cfg.peps.chi_factor * D * D
        if state is None:
            init_real = real if circuit is None else _tilde_real(real, terms_opt)
            state = product_init(init_real, D, spec, seqs[i])
        elif state.D < D:
            state = state.grow(D, spec, seqs[i])
        t0 = perf_counter()
        res = optimize_lbfgs(
            state,
            terms_opt,
            chi,
            backend,
            max_outer=cfg.optimize.max_outer,
            inner_iters=cfg.optimize.inner_iters,
            tol_E=cfg.optimize.tol_E,
            tol_g_scale=cfg.optimize.tol_g_scale,
            eps_env=cfg.env.eps_env,
            eps_env_E=cfg.env.eps_env_E,
            retry_max=cfg.env.retry_max,
            factored=cfg.env.factored,
        )
        state = res.state
        store.write_rung(
            g,
            D,
            state,
            {
                "energy": res.energy,
                "grad_norm": res.grad_norm,
                "n_iters": res.n_iters,
                "chi": res.chi,
                "converged": res.converged,
                "wall_s": perf_counter() - t0,
            },
        )
        checkpoint("rung")

    # --- stage: finalize (INV-1 report, INV-2 stability, INV-5 tail) ---
    assert state is not None
    D_final = state.D
    chi = cfg.peps.chi_factor * D_final * D_final
    tb = tail_bound(cfg.model.g_J, cfg.model.R_c)
    report = energy_certified(
        state,
        terms_opt,
        chi,
        backend,
        eps_env=cfg.env.eps_env,
        eps_env_E=cfg.env.eps_env_E,
        tail_bound=tb,
        factored=cfg.env.factored,
    )
    report = chi_extrapolation_check(
        state,
        terms_opt,
        report,
        backend,
        tau_chi=cfg.invariants.tau_chi,
        eps_env=cfg.env.eps_env,
        eps_env_E=cfg.env.eps_env_E,
        factored=cfg.env.factored,
    )
    e_total = report.e_total + e_offset
    inv5_ok = tail_certified(
        cfg.model.g_J, cfg.model.R_c, e_total / cfg.model.L**2, cfg.invariants.tau_tail
    )
    report = dataclasses.replace(report, certified=report.certified and inv5_ok)
    obs = measure_static(
        state, real, chi, backend, circuit, factored=cfg.env.factored
    )
    store.write_final(
        g,
        {
            "e_total": e_total,
            "e_per_site": e_total / cfg.model.L**2,
            "e_offset_sdrg": e_offset,
            "chi": report.env.chi,
            "max_disc_weight": report.env.max_disc_weight,
            "updown_gap": report.env.updown_gap,
            "chi_stability": list(report.chi_stability or ()),
            "tail_bound": tb,
            "inv5_ok": inv5_ok,
            "certified": report.certified,
            # §11 audit: REPORT.md must echo fallback rates and bypass counts, so
            # they have to be persisted per realization, not just logged.
            "sketch_stats": report.env.sketch_stats,
            "sdrg_bypassed": bool(sdrg.bypassed) if sdrg is not None else None,
            "sdrg_ledger_total": float(sdrg.ledger.total) if sdrg is not None else None,
        },
        {
            "q_ea": obs.q_ea,
            "sz": {f"{x},{y}": v for (x, y), v in obs.sz.items()},
            "sx": {f"{x},{y}": v for (x, y), v in obs.sx.items()},
            "czz_r": {str(r): v for r, v in obs.czz_r.items()},
            "n_res_r": {str(r): v for r, v in obs.n_res_r.items()},
        },
    )
    return "finalized"


def _tilde_real(real: "DisorderRealization", terms_opt: "HamiltonianTerms") -> "DisorderRealization":
    """§10.1: product init reads H-tilde's on-site fields."""
    L = terms_opt.L
    eps_t, dlt_t = np.zeros((L, L)), np.zeros((L, L))
    for (x, y), op, c in terms_opt.onsite:
        if op == "z":
            eps_t[y, x] += 2 * c
        else:
            dlt_t[y, x] += 2 * c
    return dataclasses.replace(real, eps=eps_t, delta=dlt_t)


def run_ensemble(
    cfg: Config, *, progress: Callable[[int, str], None] | None = None
) -> Path:
    out = Path(cfg.run.out)
    root = store.open_run(out)
    store.write_manifest(root, asdict_manifest(cfg))
    ks = list(range(cfg.run.n_realizations))
    if cfg.run.workers <= 1:
        for k in ks:
            status = run_realization(cfg, k, out)
            if progress:
                progress(k, status)
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(
            max_workers=cfg.run.workers, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = {pool.submit(run_realization, cfg, k, out): k for k in ks}
            errors: dict[int, BaseException] = {}
            # as_completed, not futures.items(): iterating in submission order
            # blocks on whichever realization happens to be listed first, which
            # can silently sit on an already-raised exception in another future
            # for hours before it's ever retrieved (found running the L=8 pilot,
            # 2026-07-20 -- a crashed worker just idles, `.result()` is what
            # actually re-raises). as_completed surfaces each result the moment
            # its worker is done, success or failure.
            for fut in as_completed(futures):
                k = futures[fut]
                try:
                    status = fut.result()
                except Exception as exc:  # noqa: BLE001 -- collect, don't abort the batch
                    errors[k] = exc
                    status = f"FAILED: {exc}"
                if progress:
                    progress(k, status)
            if errors:
                worst = "; ".join(f"k={k}: {e}" for k, e in sorted(errors.items()))
                raise RuntimeError(
                    f"{len(errors)}/{len(ks)} realizations failed: {worst}"
                )
    return out


def asdict_manifest(cfg: Config) -> dict:  # type: ignore[type-arg]
    m = build_manifest(cfg)
    d = asdict(m) if dataclasses.is_dataclass(m) else m.model_dump()
    return d
