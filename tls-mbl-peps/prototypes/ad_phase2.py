#!/usr/bin/env python3
"""tlsmbl Phase-2 prototype: differentiable energy + T-AD-FD + LBFGS -> T-GOLD-ED.

Substrate: JAX (ADR-011; torch CPU wheels unreachable through the egress
allowlist and the PyPI torch wheel is CUDA-linked). The validated math is
framework-portable; production remains torch.

AD design (ADR-011/012, discovered constraints):
  * Canonicalization inside the AD graph is SVD-based, not QR/LQ: framework
    QR-VJPs (torch and JAX alike) do not support wide matrices, and the
    right-edge canonicalization matrix IS wide.
  * Truncation gradient: full economy SVD + slice. For losses in which U and
    S@Vh recombine bilinearly over the truncation index (exactly what the
    compression graph does), this vjp is EXACT, including kept<->discarded
    spectral coupling -- superseding the projector-term approximation of the
    original Section 8.5, which becomes the sketched-backend fallback.

Modes:
  consistency          jax energy == Phase-1-certified numpy energy (chi=D^2)
  adfd CHI [NC]        T-AD-FD: jax.grad vs central differences at given chi
  opt GJ D MAXITER     product-init + random-init LBFGS, compare to ED E0
"""
import json, sys, time
import numpy as np
from pathlib import Path as _P
_RESULTS = _P(__file__).resolve().parent / "results"; _RESULTS.mkdir(exist_ok=True)
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from scipy.optimize import minimize
from golden_3x3 import (L, Z, X, sample_realization, build_terms, build_H,
                        random_peps, bmps_observables)

MASTER = 20260716
RESULTS_PATH = str(_RESULTS / "phase2-results.json")
Zj, Xj = jnp.asarray(Z, dtype=jnp.complex128), jnp.asarray(X, dtype=jnp.complex128)


def record(rec):
    try:
        data = json.load(open(RESULTS_PATH))
    except FileNotFoundError:
        data = {"records": []}
    data["records"].append(rec)
    json.dump(data, open(RESULTS_PATH, "w"), indent=1)


# ---------- real parametrization ----------
def make_codec(A0):
    metas, off = [], 0
    for y in range(L):
        for x in range(L):
            sh = A0[y][x].shape
            n = int(np.prod(sh))
            metas.append((y, x, sh, n, off))
            off += 2 * n
    def flatten(A):
        parts = []
        for y, x, sh, n, o in metas:
            parts += [np.real(A[y][x]).ravel(), np.imag(A[y][x]).ravel()]
        return np.concatenate(parts)
    def unflatten(flat):
        A = [[None] * L for _ in range(L)]
        for y, x, sh, n, o in metas:
            A[y][x] = flat[o:o + n].reshape(sh) + 1j * flat[o + n:o + 2 * n].reshape(sh)
        return A
    return flatten, unflatten, off


# ---------- differentiable boundary-MPS machinery ----------
def dblj(A, O=None):
    if O is None:
        t = jnp.einsum("plurd,pLURD->lLuUrRdD", A, A.conj())
    else:
        t = jnp.einsum("pq,qlurd,pLURD->lLuUrRdD", O, A, A.conj())
    s = A.shape
    return t.reshape(s[1] ** 2, s[2] ** 2, s[3] ** 2, s[4] ** 2)


def absorb_top(M, row):
    out = []
    for x in range(L):
        t = jnp.einsum("lvr,avbw->lawrb", M[x], row[x])
        out.append(t.reshape(M[x].shape[0] * row[x].shape[0], row[x].shape[3],
                             M[x].shape[2] * row[x].shape[2]))
    return out


def absorb_bottom(M, row):
    out = []
    for x in range(L):
        t = jnp.einsum("lwr,avbw->lavrb", M[x], row[x])
        out.append(t.reshape(M[x].shape[0] * row[x].shape[0], row[x].shape[1],
                             M[x].shape[2] * row[x].shape[2]))
    return out


def compress_j(mps, chi):
    """ADR-010 canonical compression; canonical sweep via SVD (ADR-011)."""
    mps = list(mps)
    for x in range(L - 1, 0, -1):                     # exact right-canonicalization
        k, w, m = mps[x].shape
        U, S, Vh = jnp.linalg.svd(mps[x].reshape(k, w * m), full_matrices=False)
        mps[x] = Vh.reshape(Vh.shape[0], w, m)        # orthonormal rows
        mps[x - 1] = jnp.einsum("awk,kr->awr", mps[x - 1], U * S[None, :])
    carry, out = jnp.ones((1, 1), dtype=mps[0].dtype), []
    for T in mps:                                      # truncating sweep (E-5 operand)
        W = jnp.einsum("kK,Kwm->kwm", carry, T)
        k, w, m = W.shape
        U, S, Vh = jnp.linalg.svd(W.reshape(k * w, m), full_matrices=False)
        kk = min(chi, int(S.shape[0]))
        out.append(U[:, :kk].reshape(k, w, kk))        # exact vjp via slice (ADR-012)
        carry = S[:kk, None] * Vh[:kk, :]
    out[-1] = out[-1] * carry[0, 0]
    return out


TRIVj = lambda: [jnp.ones((1, 1, 1), dtype=jnp.complex128) for _ in range(L)]


def build_tops_j(A, chi, insert=None):
    tops = [TRIVj()]
    for y in range(L):
        row = [dblj(A[y][x], (insert or {}).get((x, y))) for x in range(L)]
        tops.append(compress_j(absorb_top(tops[-1], row), chi))
    return tops


def build_bottoms_j(A, chi):
    bots = [None] * (L + 1)
    bots[L] = TRIVj()
    for y in range(L - 1, -1, -1):
        row = [dblj(A[y][x]) for x in range(L)]
        bots[y] = compress_j(absorb_bottom(bots[y + 1], row), chi)
    return bots


def sandwich_j(T, A, y, B, ops=None):
    F = jnp.ones((1, 1, 1), dtype=jnp.complex128)
    for x in range(L):
        a = dblj(A[y][x], (ops or {}).get(x))
        F = jnp.einsum("tmb,tvT->mbvT", F, T[x])       # E-4a
        F = jnp.einsum("mbvT,mvnw->bTnw", F, a)        # E-4b
        F = jnp.einsum("bTnw,bwB->TnB", F, B[x])       # E-4c
    return F.reshape(())


def make_energy(unflatten, onsite, pair, chi):
    cross_sources = sorted({i for (i, j, _) in pair if i[1] != j[1]})
    def energy(flat):
        A = unflatten(flat)                             # nested complex (traced)
        tops = build_tops_j(A, chi)
        bots = build_bottoms_j(A, chi)
        nr = [sandwich_j(tops[y], A, y, bots[y + 1]) for y in range(L)]
        dressed = {s: build_tops_j(A, chi, insert={s: Zj}) for s in cross_sources}
        E = 0.0
        for (x, y), op, c in onsite:
            O = Zj if op == "z" else Xj
            E = E + c * jnp.real(sandwich_j(tops[y], A, y, bots[y + 1], {x: O}) / nr[y])
        for (i, j, J) in pair:
            (x1, y1), (x2, y2) = i, j
            if y1 == y2:
                v = sandwich_j(tops[y1], A, y1, bots[y1 + 1], {x1: Zj, x2: Zj}) / nr[y1]
            else:
                dt = dressed[i]
                v = sandwich_j(dt[y2], A, y2, bots[y2 + 1], {x2: Zj}) / nr[y2]
            E = E + J * jnp.real(v)
        return E
    return energy


# ---------- init (§10.1) ----------
def product_init(real, D, seed, noise=1e-2):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    A = random_peps(D, seed)                      # gets the right leg dims
    for y in range(L):
        for x in range(L):
            h2 = np.array([[real["eps"][y, x] / 2, real["delta"][y, x] / 2],
                           [real["delta"][y, x] / 2, -real["eps"][y, x] / 2]])
            v = np.linalg.eigh(h2)[1][:, 0]
            T = noise * (rng.standard_normal(A[y][x].shape) +
                         1j * rng.standard_normal(A[y][x].shape))
            T[:, 0, 0, 0, 0] += v
            A[y][x] = T
    return A


# ---------- modes ----------
def setup(gJ, seed_idx=0):
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(MASTER).spawn(2)]
    real = sample_realization(seeds[seed_idx], gJ)
    onsite, pair = build_terms(real)
    return real, onsite, pair, seeds[seed_idx]


def mode_consistency():
    real, onsite, pair, seed = setup(0.3)
    for D in (2, 3):
        A0 = random_peps(D, seed + 7 * D)
        flatten, unflatten, n = make_codec(A0)
        E_np = bmps_observables(A0, D * D, onsite, pair)["E"]
        E_jx = float(make_energy(unflatten, onsite, pair, D * D)(jnp.asarray(flatten(A0))))
        rel = abs(E_jx - E_np) / abs(E_np)
        print(f"consistency D={D}: jax {E_jx:+.14f}  numpy {E_np:+.14f}  rel {rel:.3e}")
        record({"mode": "consistency", "D": D, "rel_err": rel, "pass": bool(rel < 1e-11)})


def mode_adfd(chi, ncoord=24):
    real, onsite, pair, seed = setup(0.3)
    D = 2
    A0 = random_peps(D, seed + 7 * D)
    flatten, unflatten, n = make_codec(A0)
    x0 = jnp.asarray(flatten(A0))
    energy = make_energy(unflatten, onsite, pair, chi)
    t0 = time.time(); Ej = jax.jit(energy); E0 = float(Ej(x0))
    t1 = time.time(); g = np.asarray(jax.jit(jax.grad(energy))(x0))
    t2 = time.time()
    print(f"chi={chi}: E={E0:+.10f}  |g|_max={np.abs(g).max():.3e}  "
          f"(compile+eval: E {t1-t0:.1f}s, grad {t2-t1:.1f}s)")
    disc = bmps_observables(A0, chi, onsite, pair)["disc"]
    rng = np.random.default_rng(11)
    idx = rng.choice(n, size=min(ncoord, n), replace=False)
    h, errs = 1e-5, []
    for i in idx:
        xp, xm = np.asarray(x0).copy(), np.asarray(x0).copy()
        xp[i] += h; xm[i] -= h
        fd = (float(Ej(jnp.asarray(xp))) - float(Ej(jnp.asarray(xm)))) / (2 * h)
        errs.append(abs(g[i] - fd) / max(abs(fd), 1e-3 * np.abs(g).max()))
    print(f"T-AD-FD chi={chi} (truncating={disc>1e-14}, disc={disc:.2e}): "
          f"max rel err over {len(idx)} coords = {max(errs):.3e}")
    record({"mode": "adfd", "chi": chi, "D": D, "disc_weight": disc,
            "max_rel_err": float(max(errs)), "n_coords": len(idx),
            "pass": bool(max(errs) < 1e-6)})


def mode_opt(gJ, D, maxiter):
    real, onsite, pair, seed = setup(gJ)
    H = build_H(onsite, pair)
    E0 = float(np.linalg.eigh(H.toarray())[0][0])
    A0 = random_peps(D, seed + 7 * D)
    flatten, unflatten, n = make_codec(A0)
    energy = make_energy(unflatten, onsite, pair, D * D)
    t0 = time.time()
    vg = jax.jit(jax.value_and_grad(energy))
    _ = vg(jnp.asarray(flatten(A0)))                   # compile
    tc = time.time() - t0
    def fun(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g)
    best = None
    for label, Ainit in (("product+noise", product_init(real, D, seed + 99)),
                         ("random", A0)):
        t1 = time.time()
        res = minimize(fun, flatten(Ainit), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, maxcor=20, ftol=1e-15, gtol=1e-12))
        gap = res.fun - E0
        print(f"gJ={gJ} D={D} [{label:14s}] E={res.fun:+.12f}  gap={gap:.3e}  "
              f"rel={gap/abs(E0):.3e}  iters={res.nit}  {time.time()-t1:.1f}s")
        if best is None or res.fun < best[0]:
            best = (res.fun, label, int(res.nit))
    rel = (best[0] - E0) / abs(E0)
    print(f"BEST gJ={gJ} D={D}: E={best[0]:+.12f}  E0={E0:+.12f}  "
          f"rel gap={rel:.3e}  (compile {tc:.1f}s)")
    record({"mode": "opt", "gJ": gJ, "D": D, "E_opt": best[0], "E0_ED": E0,
            "rel_gap": float(rel), "variational_ok": bool(best[0] >= E0 - 1e-9),
            "best_start": best[1], "iters": best[2], "compile_s": tc})


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "consistency":
        mode_consistency()
    elif m == "adfd":
        mode_adfd(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 24)
    elif m == "opt":
        mode_opt(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
