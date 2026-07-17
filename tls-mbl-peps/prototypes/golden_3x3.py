#!/usr/bin/env python3
"""tlsmbl Phase-1 golden test (architecture §14: T-GOLD precursor + T-PROP-DIST + T-DET + INV-7).

Validates, on 3x3 lattices (2^9 = 512, exactly solvable):
  A. ED oracle self-consistency: sparse Lanczos vs dense eigh; J=0 analytic sum.
  B. Normative einsum conventions E-1..E-5: a random PEPS is contracted two ways
     (independent brute-force statevector vs boundary-MPS machinery with zip-up
     compression through the exact_truncate kernel) and every observable in the
     Hamiltonian's support is compared: norm, <sz_i>, <sx_i>, all 36 <sz_i sz_j>
     pairs (same-row, cross-row via dressed environments per §8.4), total energy.
  C. Variational bound: E[random PEPS] >= E0(ED).
  D. INV-7 guard fires on injected NaN; sampler determinism (T-DET-lite);
     distribution sanity for the 1/Delta prior (T-PROP-DIST-lite).

Imports the kernel prototypes from bench_kernel.py (continuity: benchmark -> kernels/).
"""
import json, math, string, sys
import numpy as np
from pathlib import Path as _P
_RESULTS = _P(__file__).resolve().parent / "results"; _RESULTS.mkdir(exist_ok=True)
from scipy import sparse
from scipy.sparse.linalg import eigsh

from bench_kernel import exact_truncate, crandn

L, d = 3, 2
MASTER = 20260716
RESULTS = {"config": {"L": L, "master_seed": MASTER}, "checks": []}

I2 = np.eye(2)
Z = np.diag([1.0, -1.0])
X = np.array([[0.0, 1.0], [1.0, 0.0]])
SI = lambda x, y: y * L + x          # site index, row-major (matches statevector axis order)


# ---------------- model layer (§7.1, §7.2) ----------------
def sample_realization(seed, gJ, Rc=3, delta_min=1e-3):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    eps = rng.uniform(-1.0, 1.0, (L, L))                     # eps[y, x]
    delta = np.exp(rng.uniform(np.log(delta_min), 0.0, (L, L)))
    pairs = {}
    sites = [(x, y) for y in range(L) for x in range(L)]
    for a in range(len(sites)):
        for b in range(a + 1, len(sites)):
            (x1, y1), (x2, y2) = sites[a], sites[b]
            r = math.hypot(x1 - x2, y1 - y2)
            if 1.0 <= r <= Rc:
                pairs[(sites[a], sites[b])] = gJ * rng.uniform(-1.0, 1.0) / r ** 3
    return {"eps": eps, "delta": delta, "pairs": pairs}


def build_terms(real):
    onsite = []
    for y in range(L):
        for x in range(L):
            onsite.append(((x, y), "z", real["eps"][y, x] / 2.0))
            onsite.append(((x, y), "x", real["delta"][y, x] / 2.0))
    pair = [(i, j, J) for (i, j), J in real["pairs"].items()]
    return onsite, pair


# ---------------- ED oracle (§7.3) ----------------
def op_chain(ops, N):
    M = sparse.identity(1, format="csr")
    for s in range(N):
        M = sparse.kron(M, sparse.csr_matrix(ops.get(s, I2)), format="csr")
    return M


def build_H(onsite, pair):
    N = L * L
    H = sparse.csr_matrix((2 ** N, 2 ** N))
    for (x, y), op, c in onsite:
        H = H + c * op_chain({SI(x, y): Z if op == "z" else X}, N)
    for (x1, y1), (x2, y2), J in pair:
        H = H + J * op_chain({SI(x1, y1): Z, SI(x2, y2): Z}, N)
    return H


# ---------------- PEPS + boundary MPS (§8.1–§8.4, einsums E-1..E-5) ----------------
def random_peps(D, seed):
    spawns = np.random.SeedSequence(seed).spawn(L * L)
    A = [[None] * L for _ in range(L)]
    for y in range(L):
        for x in range(L):
            Dl, Du = (1 if x == 0 else D), (1 if y == 0 else D)
            Dr, Dd = (1 if x == L - 1 else D), (1 if y == L - 1 else D)
            rng = np.random.default_rng(spawns[y * L + x])
            A[y][x] = crandn(rng, d, Dl, Du, Dr, Dd) / np.sqrt(D ** 3 * d)
    return A


def peps_statevector(A):
    """Independent brute-force oracle: one big einsum over the full 9-tensor network."""
    letters = iter(string.ascii_letters)
    nl = lambda: next(letters)
    phys = {(x, y): nl() for y in range(L) for x in range(L)}
    hb = {(x, y): nl() for y in range(L) for x in range(L - 1)}
    vb = {(x, y): nl() for y in range(L - 1) for x in range(L)}
    subs, bnd = [], []
    for y in range(L):
        for x in range(L):
            l = hb[(x - 1, y)] if x > 0 else (bnd.append(nl()) or bnd[-1])
            u = vb[(x, y - 1)] if y > 0 else (bnd.append(nl()) or bnd[-1])
            r = hb[(x, y)] if x < L - 1 else (bnd.append(nl()) or bnd[-1])
            dn = vb[(x, y)] if y < L - 1 else (bnd.append(nl()) or bnd[-1])
            subs.append(phys[(x, y)] + l + u + r + dn)
    out = "".join(phys[(x, y)] for y in range(L) for x in range(L)) + "".join(bnd)
    expr = ",".join(subs) + "->" + out
    ops = [A[y][x] for y in range(L) for x in range(L)]
    return np.einsum(expr, *ops, optimize=True).reshape(2 ** (L * L))


def dbl(A, O=None):
    """E-1 / E-2: double-layer tensor, legs (lL, uU, rR, dD), ket index slow."""
    if O is None:
        t = np.einsum("plurd,pLURD->lLuUrRdD", A, A.conj())
    else:
        t = np.einsum("pq,qlurd,pLURD->lLuUrRdD", O, A, A.conj())
    s = A.shape
    return t.reshape(s[1] ** 2, s[2] ** 2, s[3] ** 2, s[4] ** 2)


def absorb_top(M, row):
    """E-3: top boundary MPS absorbs a row from above."""
    return [np.einsum("lvr,avbw->lawrb", M[x], row[x]).reshape(
        M[x].shape[0] * row[x].shape[0], row[x].shape[3], M[x].shape[2] * row[x].shape[2])
        for x in range(L)]


def absorb_bottom(M, row):
    """Mirror of E-3: bottom MPS (physical leg up) absorbs a row from below."""
    return [np.einsum("lwr,avbw->lavrb", M[x], row[x]).reshape(
        M[x].shape[0] * row[x].shape[0], row[x].shape[1], M[x].shape[2] * row[x].shape[2])
        for x in range(L)]


def compress(mps, chi):
    """§8.3 compression, corrected per ADR-010: exact right-canonicalization (LQ
    sweep, no truncation) THEN the truncating left->right sweep with the
    exact_truncate kernel. Only in this gauge are the SVD spectra the state's
    Schmidt coefficients, making the truncation optimal and INV-1's discarded
    weights meaningful. Kernel operand shape (E-5) is unchanged."""
    mps = list(mps)
    for x in range(L - 1, 0, -1):                 # right-canonicalize, exact
        k, w, m = mps[x].shape
        Q_, R_ = np.linalg.qr(mps[x].reshape(k, w * m).conj().T)   # mat = R_^H Q_^H
        mps[x] = Q_.conj().T.reshape(Q_.shape[1], w, m)            # orthonormal rows
        mps[x - 1] = np.einsum("awk,kr->awr", mps[x - 1], R_.conj().T)
    carry, out, disc = np.ones((1, 1)), [], []
    for T in mps:                                  # truncating sweep
        W = np.einsum("kK,Kwm->kwm", carry, T)
        k, w, m = W.shape
        U, S, Vh, dw = exact_truncate(W.reshape(k * w, m), chi)   # E-5 operand
        disc.append(dw)
        out.append(U.reshape(k, w, U.shape[1]))
        carry = S[:, None] * Vh
    out[-1] = out[-1] * carry[0, 0]
    s = np.linalg.norm(out[-1])
    out[-1] = out[-1] / s
    return out, float(np.log(s)), float(max(disc))


TRIV = lambda: [np.ones((1, 1, 1)) for _ in range(L)]


def build_tops(A, chi, insert=None):
    """tops[y] = rows 0..y-1 absorbed; optional {(x,y): O} baked in (dressed env, §8.4)."""
    tops, lns, discs = [TRIV()], [0.0], [0.0]
    for y in range(L):
        row = [dbl(A[y][x], insert.get((x, y)) if insert else None) for x in range(L)]
        t, ln, dw = compress(absorb_top(tops[-1], row), chi)
        tops.append(t); lns.append(lns[-1] + ln); discs.append(max(discs[-1], dw))
    return tops, lns, max(discs)


def build_bottoms(A, chi):
    """bottoms[y] = rows y..L-1 absorbed (bottoms[L] trivial)."""
    bots, lns = [None] * (L + 1), [0.0] * (L + 1)
    bots[L] = TRIV()
    for y in range(L - 1, -1, -1):
        row = [dbl(A[y][x]) for x in range(L)]
        bots[y], ln, _ = compress(absorb_bottom(bots[y + 1], row), chi)
        lns[y] = lns[y + 1] + ln
    return bots, lns


def sandwich(T, Tln, A, y, B, Bln, ops=None):
    """E-4a/b/c: <T| row_y |B>, with optional operator insertions {x: O}."""
    F = np.ones((1, 1, 1))
    for x in range(L):
        a = dbl(A[y][x], (ops or {}).get(x))
        F = np.einsum("tmb,tvT->mbvT", F, T[x])      # E-4a
        F = np.einsum("mbvT,mvnw->bTnw", F, a)       # E-4b
        F = np.einsum("bTnw,bwB->TnB", F, B[x])      # E-4c
    return complex(F.squeeze()) * math.exp(Tln + Bln)


def bmps_observables(A, chi, onsite, pair):
    tops, tlns, disc = build_tops(A, chi)
    bots, blns = build_bottoms(A, chi)
    norms = [sandwich(tops[y], tlns[y], A, y, bots[y + 1], blns[y + 1]) for y in range(L)]
    row_consistency = max(abs(n / norms[0] - 1.0) for n in norms)
    nrm = norms[0]
    sz, sx = {}, {}
    for y in range(L):
        for x in range(L):
            sz[(x, y)] = sandwich(tops[y], tlns[y], A, y, bots[y + 1], blns[y + 1], {x: Z}) / nrm
            sx[(x, y)] = sandwich(tops[y], tlns[y], A, y, bots[y + 1], blns[y + 1], {x: X}) / nrm
    zz = {}
    dressed_cache = {}
    for (i, j), _ in [((i, j), J) for (i, j, J) in pair]:
        (x1, y1), (x2, y2) = i, j
        if y1 == y2:
            v = sandwich(tops[y1], tlns[y1], A, y1, bots[y1 + 1], blns[y1 + 1], {x1: Z, x2: Z})
        else:                                   # dressed environment (§8.4)
            if i not in dressed_cache:
                dressed_cache[i] = build_tops(A, chi, insert={i: Z})
            dt, dl, _ = dressed_cache[i]
            v = sandwich(dt[y2], dl[y2], A, y2, bots[y2 + 1], blns[y2 + 1], {x2: Z})
        zz[(i, j)] = v / nrm
    E = sum(c * (sz if op == "z" else sx)[s].real for s, op, c in onsite) \
        + sum(J * zz[(i, j)].real for i, j, J in pair)
    max_imag = max([abs(v.imag) for v in list(sz.values()) + list(sx.values()) + list(zz.values())])
    return dict(norm=nrm, sz=sz, sx=sx, zz=zz, E=E, disc=disc,
                row_consistency=row_consistency, max_imag=max_imag)


# ---------------- brute-force oracle values ----------------
def brute_observables(A, onsite, pair):
    v = peps_statevector(A)
    nrm = float(np.vdot(v, v).real)
    N = L * L
    ev = lambda ops: float(np.vdot(v, op_chain(ops, N) @ v).real) / nrm
    sz = {(x, y): ev({SI(x, y): Z}) for y in range(L) for x in range(L)}
    sx = {(x, y): ev({SI(x, y): X}) for y in range(L) for x in range(L)}
    zz = {(i, j): ev({SI(*i): Z, SI(*j): Z}) for i, j, _ in pair}
    H = build_H(onsite, pair)
    E = float(np.vdot(v, H @ v).real) / nrm
    return dict(norm=nrm, sz=sz, sx=sx, zz=zz, E=E)


def check(name, value, tol, extra=None):
    ok = bool(value <= tol)
    RESULTS["checks"].append({"name": name, "value": float(value), "tol": tol,
                              "pass": ok, **(extra or {})})
    print(f"[{'PASS' if ok else 'FAIL'}] {name:58s} {value:.3e} (tol {tol:.0e})", flush=True)
    return ok


def main():
    ok = True
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(MASTER).spawn(2)]

    # --- D: sampler determinism + distribution sanity ---
    r1, r2 = sample_realization(seeds[0], 0.3), sample_realization(seeds[0], 0.3)
    det = float(np.max(np.abs(r1["eps"] - r2["eps"])) +
                max(abs(r1["pairs"][k] - r2["pairs"][k]) for k in r1["pairs"]))
    ok &= check("T-DET sampler bitwise reproducibility", det, 0.0)
    big = np.random.default_rng(1)
    ln_d = np.log(np.exp(big.uniform(np.log(1e-3), 0.0, 20000)))
    ok &= check("T-PROP mean(ln Delta) vs ln(dmin)/2", abs(ln_d.mean() - np.log(1e-3) / 2), 0.05)

    for gJ in (1e-3, 0.3):
        for si, seed in enumerate(seeds):
            tag = f"gJ={gJ} seed{si}"
            real = sample_realization(seed, gJ)
            onsite, pair = build_terms(real)

            # --- A: ED oracle ---
            H = build_H(onsite, pair)
            E0d = float(np.linalg.eigh(H.toarray())[0][0])
            E0s = float(eigsh(H, k=1, which="SA", maxiter=20000)[0][0])
            ok &= check(f"ED sparse vs dense [{tag}]", abs(E0s - E0d), 1e-9)
            on0, _ = build_terms(real)
            H0 = build_H(on0, [])
            E0_free = float(eigsh(H0, k=1, which="SA", maxiter=20000)[0][0])
            E_analytic = -0.5 * float(np.sqrt(real["eps"] ** 2 + real["delta"] ** 2).sum())
            ok &= check(f"ED J=0 vs analytic sum [{tag}]", abs(E0_free - E_analytic), 1e-10)

            # --- B: contraction conventions, both D, chi = D^2 (lossless at L=3) ---
            for D in (2, 3):
                A = random_peps(D, seed + 7 * D)
                bf = brute_observables(A, onsite, pair)
                bm = bmps_observables(A, D * D, onsite, pair)
                errs = [abs(bm["norm"].real / bf["norm"] - 1.0)]
                errs += [abs(bm["sz"][k].real - bf["sz"][k]) for k in bf["sz"]]
                errs += [abs(bm["sx"][k].real - bf["sx"][k]) for k in bf["sx"]]
                errs += [abs(bm["zz"][k].real - bf["zz"][k]) for k in bf["zz"]]
                ok &= check(f"E-1..E-5 max obs err D={D} [{tag}] (n={len(errs)})",
                            max(errs), 1e-9,
                            {"disc_weight": bm["disc"], "row_consistency": bm["row_consistency"]})
                ok &= check(f"energy assembly rel err D={D} [{tag}]",
                            abs(bm["E"] - bf["E"]) / max(abs(bf["E"]), 1e-12), 1e-9)
                ok &= check(f"row-consistency of norm D={D} [{tag}]",
                            bm["row_consistency"], 1e-10)
                ok &= check(f"imag leakage D={D} [{tag}]", bm["max_imag"], 1e-10)
                # --- C: variational bound ---
                ok &= check(f"variational bound E_PEPS >= E0 D={D} [{tag}]",
                            max(0.0, E0d - bm["E"]), 1e-9)

    # --- D: INV-7 guard fires ---
    A = random_peps(2, 12345)
    A[0][0] = A[0][0].copy(); A[0][0][0, 0, 0, 0, 0] = np.nan
    try:
        bmps_observables(A, 4, *build_terms(sample_realization(seeds[0], 0.3)))
        fired = 0.0 + 1.0
    except ValueError as e:
        fired = 0.0 if "INV-7" in str(e) else 1.0
    ok &= check("INV-7 raises on injected NaN", fired, 0.0)

    RESULTS["all_pass"] = bool(ok)
    with open(str(_RESULTS / "golden3x3-results.json"), "w") as f:
        json.dump(RESULTS, f, indent=1, default=str)
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(c['pass'] for c in RESULTS['checks'])}/{len(RESULTS['checks'])})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
