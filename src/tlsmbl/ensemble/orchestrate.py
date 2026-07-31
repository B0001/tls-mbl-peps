"""Ensemble orchestration (ARCHITECTURE.md §11): L1 parallelism over realizations,
checkpoint after every pipeline stage, resume by reading stage markers.

Pipeline per realization: sample -> [§7.4 Hartree outer loop over:] SDRG (INV-8
auto-bypass) -> D-ladder with per-rung checkpoints -> finalize (INV-2 + INV-5) ->
observables -> store. Crashes lose at most one ladder rung.

With `model.hartree.enabled`, SDRG + ladder + <sigma^z> measurement become the inner
problem of the mean-field loop, re-entered once per outer iteration with an updated
h_mf. Two consequences worth stating:

- Each outer iteration RE-RUNS the ladder from a product init rather than warm-starting
  from the previous field's optimum. A new h_mf changes H, so the previous optimum is a
  solution to a different problem; re-running keeps every stored rung a true optimum of
  the H recorded alongside it, at the cost of K_max ladders per realization.
- The resume contract is preserved exactly: the loop checkpoints its field BEFORE the
  ladder runs (`store.write_hartree`) and clears the previous iteration's rungs
  (`store.reset_rungs`), so the rungs on disk always belong to the recorded outer
  iteration. A kill still loses at most one rung, not one whole outer iteration.
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Callable, cast

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
from tlsmbl.model.hartree import HartreeResult, hartree_loop, tail_bound, tail_certified
from tlsmbl.model.sampling import sample_realization
from tlsmbl.observables.decoherence import inputs_from_config, tier2_record
from tlsmbl.observables.static import measure_static
from tlsmbl.optimize.finalize import chi_extrapolation_check
from tlsmbl.optimize.init import product_init
from tlsmbl.optimize.lbfgs_driver import optimize_lbfgs
from tlsmbl.peps.energy import energy_certified
from tlsmbl.peps.state import PEPSState
from tlsmbl.sdrg.circuit import SDRGCircuit
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
    def sdrg_and_ladder(
        real_o: DisorderRealization,
    ) -> tuple[PEPSState, SDRGResult | None, HamiltonianTerms, "SDRGCircuit | None", float]:
        """One inner problem: SDRG (if on) then the D-ladder, resuming from whatever
        rungs are already on disk. Factored out of the body because the §7.4 Hartree loop
        re-enters it once per outer iteration with a different mean field."""
        # INV-6: build a FRESH root SeedSequence and spawn in one shot per call, exactly
        # as core/rng.py's docstring requires. Hoisting this out of the function would
        # reuse one live SeedSequence across outer iterations -- and `spawn` is stateful
        # (PEPSState.from_product spawns L*L children per init), so the second outer
        # iteration would draw children L^2..2L^2-1 instead of 0..L^2-1. That made the
        # init depend on how many previous inits had run, which a resumed run skips:
        # measured as a 1.05e-6 relative energy difference between a resumed and an
        # uninterrupted run before this was fixed. Fresh-per-call makes every outer
        # iteration's init a pure function of the seed, so resume is bitwise again.
        seqs = np.random.SeedSequence(streams.torch_init_seed).spawn(len(cfg.peps.ladder))
        terms_o = build_terms(real_o)
        if cfg.sdrg.enabled:
            sdrg_o: SDRGResult | None = sdrg_transform(
                terms_o,
                omega_stop=cfg.sdrg.omega_stop,
                f_max=cfg.sdrg.f_max,
                keep_first_order=cfg.sdrg.keep_first_order,
                tau_sdrg=cfg.sdrg.tau_sdrg,
            )
            assert sdrg_o is not None
            # Rewritten every outer iteration: h_mf changes H, so the decimation sequence
            # is a property of the CURRENT mean field, not of the realization.
            store.write_sdrg(
                g,
                {
                    "bypassed": sdrg_o.bypassed,
                    "n_ops": 0 if sdrg_o.circuit is None else len(sdrg_o.circuit.ops),
                    "ledger_total": sdrg_o.ledger.total,
                    "E0": sdrg_o.E0,
                    "omega_sequence": sdrg_o.omega_sequence,
                },
            )
            checkpoint("sdrg")
            terms_opt_o, circuit_o, e_offset_o = sdrg_o.terms, sdrg_o.circuit, sdrg_o.E0
        else:
            sdrg_o, terms_opt_o, circuit_o, e_offset_o = None, terms_o, None, 0.0

        done = store.rungs_done(g)
        state_o: PEPSState | None = None
        if done:
            last = max(d for d in done if d in cfg.peps.ladder)
            state_o = store.read_rung(g, last, spec.dtype)
        for i, D in enumerate(cfg.peps.ladder):
            if D in done:
                continue
            chi_D = cfg.peps.chi_factor * D * D
            if state_o is None:
                init_real = real_o if circuit_o is None else _tilde_real(real_o, terms_opt_o)
                state_o = product_init(init_real, D, spec, seqs[i])
            elif state_o.D < D:
                state_o = state_o.grow(D, spec, seqs[i])
            t0 = perf_counter()
            res_o = optimize_lbfgs(
                state_o,
                terms_opt_o,
                chi_D,
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
            state_o = res_o.state
            store.write_rung(
                g,
                D,
                state_o,
                {
                    "energy": res_o.energy,
                    "grad_norm": res_o.grad_norm,
                    "n_iters": res_o.n_iters,
                    "chi": res_o.chi,
                    "converged": res_o.converged,
                    "wall_s": perf_counter() - t0,
                },
            )
            checkpoint("rung")
        assert state_o is not None
        return state_o, sdrg_o, terms_opt_o, circuit_o, e_offset_o

    # --- stages: [Hartree outer loop ->] SDRG -> D-ladder ---
    hartree_result: HartreeResult | None = None
    if not cfg.model.hartree.enabled:
        # v1 baseline (§7.4): h_mf = 0, single pass, INV-5 bound reported regardless.
        state, sdrg, terms_opt, circuit, e_offset = sdrg_and_ladder(real)
    else:
        hc = cfg.model.hartree
        # The inner solve returns m = <sigma^z> in the physical frame, measured on the
        # certified state -- the same measurement Tier-1 reports, not a second path.
        # `last` carries the objects belonging to the final inner solve so `finalize`
        # certifies exactly the state that was measured.
        last: dict[str, object] = {}

        def solve(h_mf: np.ndarray) -> np.ndarray:
            real_o = dataclasses.replace(real, h_mf=h_mf)
            st, sd, t_opt, circ, off = sdrg_and_ladder(real_o)
            chi_o = cfg.peps.chi_factor * st.D * st.D
            obs_o = measure_static(
                st, real_o, chi_o, backend, circ, factored=cfg.env.factored
            )
            last.update(
                real=real_o, state=st, sdrg=sd, terms_opt=t_opt, circuit=circ,
                e_offset=off, obs=obs_o,
            )
            m = np.zeros((cfg.model.L, cfg.model.L), dtype=np.float64)
            for (x, y), v in obs_o.sz.items():
                m[y, x] = v
            return m

        prior = store.read_hartree(g)
        resume_iter = int(prior["outer"]) if prior else 1
        h_mf0 = np.array(prior["h_mf"], dtype=np.float64) if prior else None

        def before_iteration(n: int, h_mf: np.ndarray, history: tuple[float, ...]) -> None:
            """Checkpoint BEFORE the ladder runs, so the rungs on disk always belong to
            the outer iteration recorded here -- this is what keeps the resume contract
            at 'loses at most one rung' instead of one whole outer iteration."""
            if n == resume_iter and prior is not None:
                return  # resuming into this iteration: its rungs are still valid
            if n > 1:
                store.reset_rungs(g)  # a new mean field invalidates the previous optimum
            store.write_hartree(
                g, {"outer": n, "h_mf": h_mf.tolist(), "history": list(history)}
            )
            checkpoint("hartree")

        hartree_result = hartree_loop(
            solve,
            L=cfg.model.L,
            R_c=cfg.model.R_c,
            g_J=cfg.model.g_J,
            tail_seed=streams.tail_seed,
            K_max=hc.K_max,
            alpha=hc.alpha,
            tol=hc.tol,
            h_mf0=h_mf0,
            start_iter=resume_iter,
            history0=[float(v) for v in prior["history"]] if prior else (),
            before_iteration=before_iteration,
        )
        # Certify the state that was actually measured, in the field it was solved in.
        # `hartree_loop` damps h_mf *after* `solve` returns, so `last["real"]` carries the
        # field the final state was optimized in, not the one-step-ahead field.
        real = cast(DisorderRealization, last["real"])
        # Persist that field: the disorder group was written at sample time with
        # h_mf = 0, so without this the stored artifact would describe a different H from
        # the one whose energy it reports. The rng fingerprint covers eps/delta/J only,
        # so re-writing does not disturb it. (write_disorder rewinds `stage` to
        # "sampled"; write_final below sets it to "finalized" again.)
        store.write_disorder(g, real)
        state = cast(PEPSState, last["state"])
        sdrg = cast("SDRGResult | None", last["sdrg"])
        terms_opt = cast(HamiltonianTerms, last["terms_opt"])
        circuit = cast("SDRGCircuit | None", last["circuit"])
        e_offset = cast(float, last["e_offset"])

    # --- stage: finalize (INV-1 report, INV-2 stability, INV-5 tail) ---
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
    # Tier-2 (§12), default off. Built from the *physical-frame* realization and the
    # measured local polarization, so the transverse weights come from the certified
    # state rather than the bare fields. A bad declared input raises rather than
    # silently emitting a Tier-2 number (see inputs_from_config).
    tier2: dict[str, object] | None = None
    if cfg.observables.tier2.enabled:
        t2 = cfg.observables.tier2
        tier2 = tier2_record(
            real,
            inputs_from_config(t2.omega_q, t2.g0, t2.gamma0, t2.T),
            sx=obs.sx,
            sz=obs.sz,
        ).to_json_dict()
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
            # §7.4: an unconverged mean field is reported, never hidden -- the energy is
            # still INV-1/2 certified, but it is the energy of a not-quite-self-consistent
            # H, and only this field says so.
            "hartree": None
            if hartree_result is None
            else {
                "converged": hartree_result.converged,
                "n_iters": hartree_result.n_iters,
                "max_delta": hartree_result.max_delta,
                "history": list(hartree_result.history),
                "bound_covers": hartree_result.bound_covers,
            },
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
            "tier2": tier2,  # None unless observables.tier2.enabled (§12)
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
