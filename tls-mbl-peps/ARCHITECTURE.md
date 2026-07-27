# ARCHITECTURE: `tls-mbl-peps`

**Variational PEPS solver for many-body localization of TLS defects in amorphous Al₂O₃, with SDRG preconditioning, sketch-gated truncation, and certified energy reporting.**

Version 1.0 — implementation blueprint. Status of the science: the Hamiltonian (§7), boundary-MPS contraction (§8), and randomized SVD (§8.6) are established methods; the SDRG∘PEPS composition (§9) is a research-grade proposal and is architecturally quarantined behind an A/B gate so the system is correct and useful even if Stage A fails to help.

---

## §0. Instructions to the implementing agent

You are implementing this system from scratch. Follow these rules exactly:

1. **Build in phase order (§16).** Do not begin phase N+1 until every exit criterion of phase N passes in CI. The exact-diagonalization reference (Phase 1) is the correctness anchor for everything after it; it is not optional and not last.
2. **Invariants are load-bearing (§3).** Every invariant INV-k is implemented as executable code on the hot path, not as a comment or a test-only assertion. An `EnergyReport` that has not passed its gates cannot be constructed — enforce by construction (private constructor / factory that runs the gates), not by convention.
3. **Index conventions (§6) are law.** Every tensor in the codebase uses the leg order defined there. Every `einsum` string in this document is normative — copy it, do not re-derive it. If you believe a string is wrong, prove it wrong with a shape test first, then fix the document and the code together.
4. **Two backends, one interface.** Every truncation kernel exists as a reference implementation (pure PyTorch, exact SVD) and an accelerated one (sketched, later Rust). The reference is the oracle; equivalence tests (§14.7) bind them.
5. **No silent numerics.** Non-finite values raise immediately (INV-7). Every truncation logs discarded weight. Every stochastic step records its seed.
6. **When this document underspecifies something,** choose the simplest option consistent with the invariants, record the choice as a new ADR in `docs/adr/`, and continue. Do not stall.

---

## §1. Scope and non-goals

**Goal.** For disorder ensembles of the TLS Hamiltonian $H_{\mathrm{MBL}}$ (§7) on finite $L\times L$ lattices, compute variationally certified ground states via finite PEPS, extract disorder-averaged static observables (§12), and from them estimate qubit decoherence channels ($\Gamma_1$, spectral-diffusion proxies), with explicit two-tier honesty about which outputs are rigorous and which are model-dependent.

**Deliverables.**
- D1: Converged `EnergyReport` + `StateArtifact` per disorder realization, resumable, bit-reproducible per seed.
- D2: Aggregated observables with bootstrap confidence intervals across $N_{\mathrm{dis}}$ realizations.
- D3: Kernel microbenchmarks establishing the $D$-scaling law of the truncation kernel (exact vs sketched).
- D4: A/B report quantifying SDRG preconditioning benefit ($D_{\mathrm{eff}}$ vs $D$ at matched accuracy).

**Non-goals (v1).** Real-time dynamics (no TEBD/TDVP); excited-state targeting beyond SDRG cluster statistics; 3D lattices; GPU (design must not preclude it — all tensor ops go through torch — but no GPU-specific code paths); the L2/L3 distributed sharding of §15.4 (interface stubs only, activated only if the $D$-extrapolation demands $D \gtrsim 10$).

---

## §2. Notation and units

| Symbol | Meaning |
|---|---|
| $L$ | linear lattice size; $N = L^2$ sites |
| $d = 2$ | physical dimension per site |
| $D$ | PEPS virtual bond dimension (interior); boundary legs have dim 1 |
| $\chi$ | environment (boundary-MPS) bond dimension; policy $\chi = cD^2$, default $c=1$ |
| $\varepsilon_i, \tilde\Delta_i, E_i$ | asymmetry, (polaron-renormalized) tunneling, splitting $E_i=\sqrt{\varepsilon_i^2+\tilde\Delta_i^2}$ |
| $J_{ij}$ | dipolar $zz$ coupling $U_0 c_{ij}/r_{ij}^3$ |
| $W$ | disorder bandwidth; **the internal energy unit** |
| $g_J$ | dimensionless coupling scale $J(a)/W = \chi_g$ (default $10^{-3}$) |
| $R_c$ | interaction cutoff radius (lattice units) |
| $\Omega$ | SDRG running scale |
| $A[x,y]$ | PEPS site tensor; $a[x,y]$ | double-layer tensor |
| $\mathcal{T}_y, \mathcal{B}_y$ | top/bottom boundary MPS after absorbing rows $1..y$ / $y..L$ |

**Units module (`core/units.py`).** Internal: $\hbar = 1$, all energies in units of $W$, all lengths in units of the coarse-graining constant $a = (P_0\, t\, W)^{-1/2}$. Physical constants ($P_0, U_0, \gamma, \rho, v, p, \epsilon_r, t$) live only in `ModelParams.physical` and are converted **once** at ingestion into the two dimensionless numbers the solver actually consumes: $g_J = P_0 U_0$ and $W$-relative field distributions. Conversion back to GHz/seconds happens only in `observables/decoherence.py` and `io/`. No SI quantity may appear anywhere else — enforce with a `Quantity` NewType that the tensor layers do not import.

---

## §3. Invariant registry (non-bypassable)

Each invariant has: an ID, a gate location (the function that enforces it), a failure action, and a test (§14). `verify.py` in each package re-exports its gates so `tlsmbl verify <artifact>` can re-check any stored result offline.

| ID | Statement | Gate location | Failure action |
|---|---|---|---|
| INV-1 | **Environment convergence certificate.** An energy may be reported only if the boundary-MPS compression it used satisfies, on every bond, discarded weight $w_{\mathrm{disc}} = \sum_{k>\chi}\sigma_k^2 / \sum_k \sigma_k^2 \le \varepsilon_{\mathrm{env}}$, and the two-sided sweep energy difference $\lvert E^{\downarrow} - E^{\uparrow}\rvert \le \varepsilon_{\mathrm{env}}^{E}$ (contract top→bottom and bottom→top; both stored). Discarded weights are the state's true Schmidt discards only because compression right-canonicalizes before truncating (ADR-010); the gate is unsound without it. | `peps/energy.py::energy_certified` | raise `EnvironmentNotConverged`; optimizer catches, raises $\chi \to \chi + \Delta\chi$, retries (max 3) |
| INV-2 | **$\chi$-stability audit.** Strict monotonicity of $E$ in $\chi$ is not guaranteed for approximate contraction, so the enforceable certificate is stability: at the converged point, re-evaluate the energy once at $2\chi$ and require $\lvert E(\chi) - E(2\chi)\rvert \le \tau_\chi$; both values are stored in the report. | `optimize/finalize.py::chi_extrapolation_check` | mark artifact `UNCERTIFIED`, exclude from aggregation unless `--allow-uncertified` |
| INV-3 | **Sketch-error gate (two-sided, ADR-009).** The randomized backend may return a truncation only if its posterior estimate (§8.6) satisfies $\widehat{\lVert\rho - QQ^\dagger\rho\rVert}_2 \le \max\!\big(\eta\,\sigma_1^{\mathrm{est}},\; c_{\mathrm{gate}}\,\hat\sigma_{\chi+1}\big)$ — sketch quality is judged against the best any rank-$\chi$ truncation could achieve ($\hat\sigma_{\chi+1}$ is read off the sketch's own spectrum since $k=\chi{+}p>\chi$); whether rank $\chi$ itself is adequate is INV-1's job. Otherwise it falls back to the exact backend for that call and increments `fallback_count`. | `kernels/rsvd.py::truncate` | silent fallback + counter; if `fallback_rate > 20%` over a sweep, disable sketching for the realization and log |
| INV-4 | **Density-operator sanity.** Every matricized compression operand is checked: finite entries; if a norm object, Hermiticity $\lVert\rho-\rho^\dagger\rVert_F \le 10^{-10}\lVert\rho\rVert_F$ after symmetrization. | `kernels/common.py::check_operand` | raise `NumericalCorruption` |
| INV-5 | **Hartree tail bound.** Reported per-site energy carries the rigorous tail bound $\delta e_{\mathrm{tail}} = 2\pi \bar P U_0 / R_c$ (lattice form: $g_J \sum_{r>R_c} r^{-3} \approx 2\pi g_J/R_c$); require $\delta e_{\mathrm{tail}} \le \tau_{\mathrm{tail}}\cdot \lvert e \rvert$ else artifact is `UNCERTIFIED`. | `model/hartree.py::tail_bound` | mark `UNCERTIFIED` |
| INV-6 | **Reproducibility.** Every realization is keyed `(master_seed, realization_index)`; all RNG (numpy for disorder, torch for init and sketches) derived via `np.random.SeedSequence(master_seed).spawn()`. `torch.use_deterministic_algorithms(True)` in certification runs. The manifest records seeds, config hash, git SHA, package versions. | `core/rng.py`, `io/manifest.py` | build refuses to run without a seed |
| INV-7 | **No silent NaN / degenerate-spectrum discipline.** All kernel outputs pass `torch.isfinite` checks; SVD backward uses Lorentzian-broadened $F_{ij}$ (§8.5) with $\epsilon_F$ logged; any `nan` raises with full tensor provenance (site, sweep, bond). | decorator `core/guards.py::@finite` on every kernel | raise `NumericalCorruption` |
| INV-8 | **SDRG fidelity ledger.** Stage A must track the operator-norm sum of all dropped terms $B_{\mathrm{drop}} = \sum_m \lVert h_{\mathrm{dropped}}^{(m)}\rVert$ and the cluster-projection error (§9.4); if $B_{\mathrm{drop}} > \tau_{\mathrm{sdrg}} \cdot \lVert H\rVert_{\mathrm{loc}}$ the preconditioner is rejected for that realization and the pipeline runs Stage-A-off. | `sdrg/ledger.py` | automatic Stage-A bypass, logged |
| INV-9 | **Gauge-fixed complex SVD.** Every SVD (forward) fixes the phase gauge: the largest-modulus entry of each column of $U$ is rotated to be real positive (apply inverse phase to $V$). Required for AD stability of complex SVD. | `kernels/svd.py::svd_gauge_fixed` | n/a (constructive) |

---

## §4. Repository layout and dependencies

```
tls-mbl-peps/
├── pyproject.toml            # PEP 621; package name tlsmbl
├── src/tlsmbl/
│   ├── core/                 # units.py, rng.py, guards.py, types.py, config.py
│   ├── model/                # sampling.py, hamiltonian.py, hartree.py, ed_reference.py
│   ├── peps/                 # state.py, doublelayer.py, boundary.py, energy.py, correlators.py, autodiff.py
│   ├── kernels/              # interface.py, svd.py (exact), rsvd.py (sketched), zipup.py, rust_backend.py
│   ├── sdrg/                 # rules.py, circuit.py, transform.py, ledger.py
│   ├── optimize/             # init.py, simple_update.py, lbfgs_driver.py, ladder.py, finalize.py
│   ├── ensemble/             # orchestrate.py, checkpoint.py, aggregate.py
│   ├── observables/          # static.py, decoherence.py
│   ├── io/                   # store.py (zarr), manifest.py, schema.py
│   └── cli.py                # typer app: run | verify | bench | aggregate | ab-test
├── rust/                     # cargo workspace (Phase 6)
│   └── tlsmbl-kernels/       # pyo3 crate: zip-up truncation kernel
├── tests/{unit,golden,property,perf}/
├── configs/                  # benchmark.yaml, smoke.yaml, ab_sdrg.yaml
└── docs/adr/                 # ADR-001..; template included
```

**Dependencies (pin in pyproject):** `torch>=2.3` (CPU build; complex128 default), `numpy`, `scipy` (sparse Lanczos for ED), `zarr`, `numcodecs`, `pydantic>=2` (config schema + validation), `typer`, `hypothesis` (property tests), `pytest`, `pytest-benchmark`. Rust phase: `pyo3`, `numpy` (rust-numpy), `faer` for QR/SVD, `rayon`. **Do not** depend on quimb/TeNPy for the solver itself (the point is a controlled, invariant-gated implementation) — quimb MAY be used inside `tests/golden/` as an independent cross-check oracle, clearly quarantined to tests.

**Dtype policy.** `complex128` everywhere in certification runs; `complex64` allowed only in `bench` mode; the dtype is a config field threaded through `core/types.py::TensorSpec`, never hardcoded.

---

## §5. Data model (exact types)

All in `core/types.py` as frozen dataclasses / pydantic models. Shapes are part of the type docs and are asserted in `__post_init__`.

```python
class ModelParams(BaseModel):
    L: int                      # lattice linear size
    W: float = 1.0              # bandwidth in internal units (always 1.0; field kept for clarity)
    delta_min: float = 1e-3     # Δ_min/W, lower cutoff of log-uniform tunneling
    g_J: float = 1e-3           # J(a)/W = P0*U0  (dimensionless dipolar scale)
    R_c: int = 3                # interaction cutoff in lattice units (Chebyshev radius)
    polaron_kappa: float = 0.0  # ln-suppression scale: Δ̃ = Δ * exp(-kappa * xi_i), xi_i ~ U[0,1]; 0 disables
    physical: PhysicalConstants | None = None   # SI ingestion block (P0,t,U0 parts...) -> derives g_J, a
    seed_realization: int       # spawned, not chosen by hand

class DisorderRealization(BaseModel):     # produced by model/sampling.py
    params: ModelParams
    eps:   Float[Array, "L L"]            # ε_i / W ∈ [-1, 1]
    delta: Float[Array, "L L"]            # Δ̃_i / W  (already polaron-renormalized)
    J:     SparsePairs                    # {((x1,y1),(x2,y2)) -> J_ij/W}, only r ≤ R_c, r≥1
    h_mf:  Float[Array, "L L"]            # Hartree tail field, updated by outer loop (§7.4)
    rng_fingerprint: str

class HamiltonianTerms(BaseModel):        # canonical term list; the ONLY H representation
    onsite: list[tuple[Site, Op1, float]]         # (i, 'z'|'x', coeff)
    pair:   list[tuple[Site, Site, float]]        # zz only in v1: (i, j, J_ij)
    norm_local: float                              # Σ|coeff|, used by INV-8 threshold

class PEPSState:
    tensors: list[list[Tensor]]           # A[x][y], complex, shape (d, Dl, Du, Dr, Dd); boundary legs dim 1
    D: int
    def leg_dims(self, x, y) -> tuple[int,int,int,int]

class EnvBundle:                           # per-realization contraction cache
    top:    list[BoundaryMPS]              # T_y for y = 0..L   (T_0 = trivial)
    bottom: list[BoundaryMPS]              # B_y for y = L+1..1
    chi: int
    disc_weights: list[float]              # per compression, for INV-1
    updown_gap: float                      # |E↓ − E↑| certificate input

class EnergyReport(BaseModel):             # constructible ONLY via peps.energy.energy_certified
    e_total: float; e_per_site: float
    env: EnvCertificate                    # chi, max disc weight, updown gap, fallback stats
    tail_bound: float                      # INV-5 value
    chi_stability: tuple[float,float] | None   # (E(χ), E(2χ)) for INV-2
    sketch: SketchStats | None             # posterior errors, fallback_count
    grad_norm: float; n_iters: int; wall_s: float
    certified: bool                        # AND of all gates

class SDRGCircuit(BaseModel):              # §9
    ops: list[SiteRotation | BondCluster]  # ordered, m = 1..M
    dropped_norm: float                    # INV-8 ledger
    projection_error: float
    depth_estimate: int

class StateArtifact:                       # what io/store.py persists per realization
    realization: DisorderRealization
    circuit: SDRGCircuit | None
    peps: PEPSState                        # final tensors
    report: EnergyReport
    observables: StaticObservables
```

`SparsePairs` is a dict keyed by canonically ordered site pairs; iteration order is sorted (reproducibility).

**Zarr layout (`io/store.py`).**

```
run.zarr/
├── manifest.json                  # config hash, git sha, seeds, versions, invariant thresholds
├── realizations/{k:05d}/
│   ├── disorder/{eps, delta, J_indices, J_values, h_mf}
│   ├── sdrg/{ops.msgpack, ledger}
│   ├── peps/tensors                # ragged store: one array per site, chunked whole
│   ├── report.json
│   └── observables/{sz, sx, czz_r, pair_table}
└── aggregate/{...}                 # written only by ensemble/aggregate.py
```

---

## §6. Index conventions and normative einsum strings

**Lattice.** Site $(x, y)$: $x$ = column $1..L$ (left→right), $y$ = row $1..L$ (top→bottom). Distances are Euclidean in lattice units; $R_c$ cutoff uses $r_{ij} = \lVert \mathbf r_i - \mathbf r_j \rVert_2 \le R_c$, $r \ge 1$.

**PEPS site tensor.** `A[p, l, u, r, d]` — physical, left, up, right, down. Boundary legs have dimension 1 (never omitted). Interior legs have dimension $D$ (heterogeneous dims allowed by the code; the config uses uniform $D$).

**Double layer (norm).** Bra/ket fused per leg, ket index fastest:

```python
# a[(l l̄),(u ū),(r r̄),(d d̄)]
a = torch.einsum('plurd,pLURD->lLuUrRdD', A, A.conj()) \
        .reshape(Dl*Dl, Du*Du, Dr*Dr, Dd*Dd)              # E-1
```

**Double layer with one-site operator** $O \in \mathbb{C}^{d\times d}$ (acts on ket):

```python
aO = torch.einsum('pq,qlurd,pLURD->lLuUrRdD', O, A, A.conj()).reshape(...)   # E-2
```

**Boundary MPS tensor.** `M[l, v, r]` — left bond ($\le\chi$), vertical physical leg (dim $D^2$, pointing **down** for top MPS $\mathcal T$, **up** for bottom MPS $\mathcal B$), right bond.

**Row absorption (top MPS ← row $y$).** `M[l,v,r]` meets row tensor `a[α,u,β,w]` (`α`=left, `u`=up — contracts with `v`, `β`=right, `w`=down). Normative:

```python
Mp = torch.einsum('lvr,avbw->lawrb', M, a).reshape(chiL*Da, Dw, chiR*Db)   # E-3
```

Result legs: left $(l\,a)$, physical $w$ (the new down leg), right $(r\,b)$. After absorbing a full row, bond dims are $\le \chi D^2$; compression (§8.3) restores $\le\chi$. Bottom MPS is the mirror image (physical leg = up leg of the row below); implement once with a `flip` flag, test both orientations explicitly.

**Norm and expectation values.** With cached $\mathcal T_{y-1}$ (top rows $1..y{-}1$) and $\mathcal B_{y+1}$ (rows $y{+}1..L$), the norm reduces to an MPS–MPO–MPS sandwich along row $y$:

```python
# left-to-right transfer; F has legs (top_bond, mid_down?, bottom_bond):
F = ones(1,1,1)
for x in 1..L:
    F = torch.einsum('tmb,tvT->mbvT', F, T[x])           # E-4a  absorb top MPS tensor
    F = torch.einsum('mbvT,mvnw->bTnw', F, a[x])         # E-4b  absorb row double-layer (m=left,v=up,n=right,w=down)
    F = torch.einsum('bTnw,bwB->TnB', F, Bm[x])          # E-4c  absorb bottom MPS tensor
norm = F.squeeze()                                       # scalar
```

`⟨O_i⟩`: same loop with `a[x_i]` replaced by `aO` (E-2), divided by `norm`. Same-row `⟨O_i O_j⟩`: two replacements in one loop. Cross-row pairs: §8.4.

**Zip-up compression matricization.** During a left→right compression sweep of a fat MPS (bonds $\chi D^2$), at site $x$ the working tensor `W[k, w, m]` (`k` = truncated left bond $\le\chi$, `w` = physical $D^2$, `m` = fat right bond $\le\chi D^2$) is matricized **rows = (k,w), cols = m**:

```python
Wmat = W.reshape(k*Dw, m)          # E-5   ρ ∈ C^{(χD²)×(χD²)} in steady state — THE kernel operand
```

This `Wmat` is the object every truncation backend (§8) consumes. Its SVD at cost $\Theta((\chi D^2)^3)$ is the bottleneck identified in the derivation; the sketched backend reduces it to $\Theta(\chi^3 D^4)$ using the factored matvec (§8.6).

---

## §7. Model layer (`model/`)

### 7.1 Disorder sampling (`sampling.py`)

```
function sample_realization(params, seed) -> DisorderRealization:
    rng = np.default_rng(SeedSequence(seed))
    eps[x,y]   ~ U[-1, 1]                                  # ε/W
    delta[x,y] = exp( U[ln delta_min, ln 1] )              # P(Δ) ∝ 1/Δ  (log-uniform)
    if polaron_kappa > 0: delta *= exp(-polaron_kappa * U[0,1])   # quenched renormalization proxy
    for each ordered pair (i,j), 1 ≤ r_ij ≤ R_c:
        c ~ U[-1, 1];  J[i,j] = g_J * c / r_ij**3
    h_mf = zeros(L, L)
    return DisorderRealization(...)
```

Distributional tests: §14.5. The $1/\Delta$ law is the standard-tunneling-model prior; `delta_min` regularizes it and is a physics knob (sweep in the study plan).

### 7.2 Hamiltonian assembly (`hamiltonian.py`)

$$H = \sum_i \tfrac{\varepsilon_i + h^{\mathrm{MF}}_i}{2}\sigma^z_i + \tfrac{\tilde\Delta_i}{2}\sigma^x_i + \sum_{(i,j): r\le R_c} J_{ij}\, \sigma^z_i \sigma^z_j$$

emitted as `HamiltonianTerms` (the only representation any layer consumes; ED, PEPS energy, and SDRG all read this list — one source of truth).

### 7.3 ED reference (`ed_reference.py`) — the oracle

Sparse $H$ in the full $2^N$ basis via Kronecker assembly of the term list; `scipy.sparse.linalg.eigsh(k=4, which='SA')`. Mandatory support: $L \in \{3,4\}$ ($2^{16} = 65{,}536$). Optional `--big` path for $L=5$ ($2^{25}$: CSR ~ few GB, Lanczos with float64; implement matrix-free `LinearOperator` applying term list on-the-fly to avoid storing $H$). Also exports `ed_observables()` returning $\langle\sigma^z_i\rangle, \langle\sigma^x_i\rangle, \langle\sigma^z_i\sigma^z_j\rangle$ for golden comparison.

### 7.4 Hartree tail (`hartree.py`) — outer self-consistency

```
h_mf ← 0
for outer in 1..K_max (default 8):
    converge PEPS (inner problem) → measure m[x,y] = ⟨σz⟩
    h_new[i] = Σ_{j: r_ij > R_c} (g_J c_ij / r³) m[j]        # c_ij for r>R_c drawn once at sampling with same rng stream
    h_mf ← (1-α) h_mf + α h_new        (α = 0.5 damping)
    stop when max|h_mf − h_new| < τ_MF (default 1e-4)
```

Tail contributions are generated lazily from the realization's spawned RNG stream so `r > R_c` couplings are reproducible without storing $O(N^2)$ values. INV-5 bound computed here. v1 default: `K_max = 1` with `h_mf = 0` and the bound reported (the pure-truncation baseline); the loop is behind `config.model.hartree.enabled`.

---

## §8. PEPS + kernels layers (`peps/`, `kernels/`)

**ADR-001 (recorded up front).** Disorder makes every site tensor distinct, so translation-invariant CTMRG is the wrong environment algorithm here. The environment is **finite-lattice boundary-MPS contraction** (Verstraete–Cirac / Lubasch-style). The algebraic kernel is unchanged from the derivation — a truncated SVD of a $(\chi D^2)\times(\chi D^2)$ operand — so all bottleneck analysis and the sketched mitigation carry over verbatim; only the surrounding geometry differs.

### 8.1 State (`state.py`)

`PEPSState.random(L, D, dtype, generator)` — i.i.d. `CN(0, 1/sqrt(D³d))` entries; `PEPSState.from_product(spins)` — $D$-padded product state ($A[p,0,0,0,0] = v_p$, rest $10^{-2}\cdot$noise to break gauge degeneracy); `PEPSState.grow(D_new)` — zero-pad legs then add $10^{-3}$-scale noise on the new slices (the $D$-ladder operator, §10.3).

### 8.2 Environment construction (`boundary.py`)

```
function build_env(a[·,·], chi, backend) -> EnvBundle:
    T[0] = trivial MPS (all bonds 1)
    for y in 1..L:
        fatT = absorb_row(T[y-1], a[·,y])        # E-3 per column; bonds ≤ χD²
        T[y], stats = compress(fatT, chi, backend)   # §8.3; log disc weight (INV-1)
    B[L+1] = trivial;  for y in L..1: B[y] = compress(absorb_row_from_below(B[y+1], a[·,y]))
    return EnvBundle(top=T, bottom=B, ...)
```

Both sweep directions are built (needed for INV-1's up/down certificate and §8.4 caching). Memory: `EnvBundle` holds $2(L{+}1)$ MPSs of $L$ tensors, $O(L^2 \chi^2 D^2)$ — the dominant persistent allocation.

### 8.3 Compression (`kernels/zipup.py`) — the bottleneck, isolated

Single entry point used by everything:

```python
def compress(fat_mps, chi, backend: TruncationBackend, *, want_grad: bool) -> tuple[BoundaryMPS, CompressStats]
```

Algorithm (per ADR-010: exact right-canonicalization sweep, then truncating left→right sweep; optional right→left variational polish if `config.env.polish=True`):

```
for x in L..2:                                     # exact LQ sweep, no truncation
    (Q, R) = lq(fat[x])                            # fat[x] = R·Q, Q rows orthonormal
    fat[x] = Q;  fat[x-1] = fat[x-1] · R           # absorb R leftward
carry = Identity(1)
for x in 1..L:                                     # truncating sweep, now in canonical gauge
    W = einsum('kK,Kwm->kwm', carry, fat[x])      # apply carried factor;  k ≤ χ, w = D², m ≤ χD²
    Wmat = reshape(W, (k*w, m))                    # E-5
    U, S, Vh, stats = backend.truncate(Wmat, chi)  # ← THE KERNEL (INV-3, INV-4, INV-7, INV-9 inside)
    out[x] = reshape(U, (k, w, χx));  carry = diag(S) @ Vh
normalize: absorb final carry scalar into out[L]; record Σ log-norms separately (avoid overflow, INV-7)
```

**ADR-010 (from the executed 3×3 golden test).** Naive zip-up (truncating sweep alone) SVDs a gauge-dependent matrix: without the right-canonical gauge, the spectrum being truncated is *not* the state's Schmidt spectrum, so the truncation is suboptimal, "lossless" $\chi$ silently isn't (measured: $O(10^{-1})$ observable errors and row-inconsistent norms at $\chi=D^2$ on 3×3, where the Schmidt rank provably fits), and — worse — INV-1's discarded weights certify the wrong quantity. With the exact LQ pre-sweep, errors drop to $\le 3\times10^{-15}$ with identically zero discarded weight in the same setting. The pre-sweep adds one QR of the same $(\chi D^2)$-cube complexity class per site (constant-factor ≈2×); the truncation-kernel operand shape is unchanged, so the D3 benchmark and Stage-B sketching analysis are unaffected. Inside the AD graph the LQ factor is realized via full-rank SVD (ADR-011): framework QR-VJPs (torch and JAX alike) reject wide matrices, and the right-edge canonicalization operand is wide.

`TruncationBackend` protocol (`kernels/interface.py`):

```python
class TruncationBackend(Protocol):
    def truncate(self, Wmat: Tensor, chi: int) -> TruncResult:  # U(m×χ'), S(χ'), Vh(χ'×n), disc_weight, posterior_err|None
```

Implementations: `ExactSVD` (torch `linalg.svd`, gauge-fixed INV-9, reference oracle), `SketchedSVD` (§8.6), `RustZipUp` (§8.7, Phase 6 — replaces the whole per-site loop, not just the SVD).

### 8.4 Energy and correlators (`energy.py`, `correlators.py`)

Energy assembles term-by-term from `HamiltonianTerms` using cached environments — never rebuild an environment per term:

- **On-site terms and same-row pairs:** row sandwich E-4 with operator insertions. For row $y$: precompute column prefix transfers $F^{\rightarrow}_x$ and suffix $F^{\leftarrow}_x$ once ($O(L)$ E-4 steps); then any single insertion at column $x$ costs one spliced contraction $O(\chi^2 D^4)$, and any same-row pair $(x_1,x_2)$ costs the prefix at $x_1{-}1$, suffix at $x_2{+}1$, and the explicit segment between — $O((x_2{-}x_1)\,\chi^2 D^4)$, bounded by $R_c$.
- **Cross-row pairs** $(y_1 < y_2 \le y_1{+}R_c)$: contract the slab $\mathcal T_{y_1-1} \cdot (\text{rows } y_1..y_2 \text{ with insertions}) \cdot \mathcal B_{y_2+1}$ column-by-column with the same prefix/suffix trick applied to the slab transfer object (legs: top bond, $\le R_c$ vertical $D^2$ legs, bottom bond). Cost per pair $O(L\,\chi^2 D^{2(1+\Delta y)})$ is unacceptable for $\Delta y \ge 2$; instead compress the slab: absorb rows $y_1..y_2{-}1$ into $\mathcal T$ sequentially **with the $\sigma^z$ insertion at $(x_1,y_1)$ baked into that row's E-2 tensor**, yielding an operator-dressed top MPS $\mathcal T^{(z@i)}_{y_2-1}$; then all pairs $(i, \cdot)$ in row $y_2$ read off one dressed sandwich. Reuse: one dressed environment per source site serves all its $O(R_c^2)$ partners above it. Total correlator cost: $O(N \cdot R_c \cdot L\,\chi^3 D^4)$-class — subleading to optimization but nontrivial; budget in §15.
- **Norm:** one undressed sandwich per row (already needed); assert all rows agree to $10^{-10}$ relative (cheap cross-check of environment consistency; log worst-case).

`energy_certified(state, H, env_cfg) -> EnergyReport` runs INV-1/2/4/5/7 gates and is the **only** public energy API.

### 8.5 Autodiff (`autodiff.py`)

Gradient = torch autograd through the full certified-energy graph (Liao–Liu–Wang–Xiang approach):

- Real parametrization: optimizer sees `torch.view_as_real(A)` leaves; complex views reconstructed inside the graph.
- **TruncSVD backward (ADR-012, validated by executed T-AD-FD).** In the compression graph every truncation's outputs recombine *bilinearly over the truncation index* ($U$ against the carry $SV^\dagger$), so the loss depends on each truncation only through the rank-$\chi$ product. For such losses the exact adjoint — including kept↔discarded spectral coupling — is obtained by computing the **full economy SVD in forward and slicing**: cotangents on discarded columns are zero and the framework's full-SVD vjp (torch and JAX both implement the complex case) yields the exact gradient. Executed validation on the full 3×3 energy graph: FD agreement $8\times10^{-10}$ lossless and $4.7\times10^{-9}$ at discarded weight $1.9\times10^{-2}$. The projector-term formula below is **demoted to the sketched-backend fallback** (no discarded spectrum available there); its error is $O(\sigma_{\chi+1}/\sigma_\chi)$ — small exactly where INV-3 certifies operation. Degenerate-spectrum hardening still applies to the production Function: $F_{ij} = (s_j^2 - s_i^2)\,/\,\big((s_j^2-s_i^2)^2 + \epsilon_F^2\big)$ (Lorentzian broadening, $\epsilon_F$ from config, logged per call — INV-7):

  $dA = U\big[(F \circ (U^\dagger dU - dU^\dagger U))S + S(F \circ (V^\dagger dV - dV^\dagger V))\big]V^\dagger \;+\; (I - UU^\dagger)\,dU\,S^{-1}V^\dagger \;+\; U S^{-1} dV^\dagger (I - VV^\dagger)$

  plus the diagonal $dS$ term; complex case requires the INV-9 gauge fix in forward and the standard imaginary-diagonal correction term — implement against finite differences before anything depends on it (§14.4 blocks Phase 2 exit).
- Memory control: `torch.utils.checkpoint.checkpoint` around each **row absorption+compression** unit (recompute in backward). Peak activation memory then $O(L\,\chi^2 D^4)$ for the active row rather than $O(L^2 \cdot)$. Config flag `env.checkpoint_rows=True` default.
- Sketch randomness under AD: the Gaussian test matrix $G$ is a saved constant per forward call (`no_grad` buffer); gradients flow through $Q$ (QR backward) and the small SVD only.

### 8.6 Sketched backend (`kernels/rsvd.py`) — Stage B

Halko–Martinsson–Tropp with oversampling $p$ (default 8) and power iterations $q$ (default 1):

```
function truncate(Wop, chi):
    # Wop is either an explicit (m×n) matrix (v1) or a factored LinearOp (v1.1, below)
    G ~ CN(0,1)^{n×(chi+p)}                (saved buffer)
    Y = Wop @ G;  repeat q times: Y = Wop @ (Wopᴴ @ Y)     # matmuls: Θ(χ³D⁴) explicit; Θ(χ³D⁴) factored
    Q, _ = qr(Y)
    B = Qᴴ @ Wop                            # (chi+p) × n
    Ub, S, Vh = svd_gauge_fixed(B);  U = Q @ Ub;  keep leading chi
    # posterior gate (INV-3, two-sided per ADR-009): s probe vectors ω_j ~ CN(0,1)ⁿ, s = 6:
    est = 10 * sqrt(2/π) * max_j ‖(I − QQᴴ) (Wop @ ω_j)‖₂        # ≥ ‖W − QQᴴW‖₂ w.p. 1 − 10⁻ˢ
    thresh = max(η * S[0], c_gate * S[chi])                       # c_gate default 10
    if est > thresh:  return ExactSVD.truncate(...)  [fallback_count += 1]
```

**ADR-009 (from the D3 kernel benchmark).** The original fixed-$\eta$ gate conflated two failure modes: "the sketch missed the range" and "rank $\chi$ cannot achieve $\eta$ accuracy on this operand." Measured instance: $D{=}4$, $\chi{=}16$, spectrum $\sigma_k = e^{-0.5k}$ — the sketch's error equaled the *optimal* rank-16 error ($3.3546\times10^{-4}$, agreement to 14 significant digits, principal angle $9\times10^{-5}$ deg), yet the v1 gate rejected it because $\eta\sigma_1 = 10^{-6}$ demands what no rank-16 truncation can deliver. The two-sided threshold accepts optimal-quality sketches and routes rank inadequacy to INV-1's discarded-weight check and $\chi$-escalation, where it belongs. Slow-spectrum operands ($\sigma_k \sim 1/k$) still fall back conservatively (probe estimator is Frobenius-weighted, structurally pessimistic there) — acceptable, since that regime triggers $\chi$-escalation anyway.

Complexity: matmuls dominate at $\Theta(\chi^3 D^4)$ vs exact $\Theta(\chi^3 D^6)$ — the $\Theta(D^2)$ speedup. **v1 ships the explicit-matrix form** (the fat tensor is materialized by absorption anyway); **v1.1** (`kernels/factored.py`, ADR-015; config `env.factored`) compresses straight from the factored per-site pairs `(carry, M, a)`, never materializing the $\Theta(\chi^2 D^6)$ fat tensors: the ADR-010 canonical gauge is carried by right-to-left bond Grams (transferred in $D^2$ physical-leg slices, Cholesky-factored with relative jitter), and each truncation SVDs $W\!\cdot\!L_x$ — same Schmidt spectrum and flop class as the LQ-swept operand, peak memory $\Theta(\chi^2 D^4)$, a $D^2\times$ reduction. (The original v1.1 sketch — a `LinearOp` matvec on the *uncanonicalized* operand at $\Theta(\chi D^4)$ peak — predates ADR-010 and is incompatible with it; see ADR-015.) Activated when the $D$-ladder hits memory limits ($D \ge 6$ requires it), guarded by the equivalence gate `tests/unit/test_factored_compress.py`.

Justification for tightness of the gate in the target regime: localized-phase environments have rapidly decaying spectra, so $\sigma_{\chi+1} \ll \sigma_1$ precisely where this solver operates; the gate makes the method self-diagnosing outside that regime rather than silently wrong (INV-3).

### 8.7 Rust kernel (`rust/tlsmbl-kernels`) — Phase 6, conditional

Scope: the entire zip-up loop of §8.3 (absorb-matricize-RSVD-carry for one row), not a per-SVD FFI (per-call crossings would dominate). Interface (pyo3):

```rust
#[pyfunction]  // complex128, row-major; returns (tensors, S_list, disc_weights, posterior_errs)
fn zipup_row(fat_tensors: Vec<PyReadonlyArray3<Complex64>>, chi: usize, p: usize, q: usize, eta: f64) -> ZipUpOut
```

Internals: `faer` QR + SVD, `rayon` over the Gaussian block columns, BLAS-3 matmuls. AD contract: Python side wraps `zipup_row` in `TruncSVD`-style `autograd.Function`s per site using the returned $(U,S,V^\dagger)$ — backward stays in torch (formula §8.5), so Rust never needs to differentiate. Equivalence gate: §14.7 must pass at $10^{-10}$ (exact mode) and distribution-level for sketched mode (same seeds ⇒ identical $G$ ⇒ bitwise-comparable up to BLAS reduction order; assert $\le 10^{-8}$).

---

## §9. SDRG preconditioner (`sdrg/`) — Stage A, quarantined

**ADR-002.** Decimation must not destroy lattice regularity (PEPS needs it). Therefore: the circuit $U$ is accumulated, but decimated sites **remain in the lattice as pinned sites** — they keep their dominant rotated local term and their exactly-retained 2-local residual couplings. $|\Psi\rangle = U|\Phi\rangle$ with $\Phi$ a PEPS on the *original* $L\times L$ lattice for $\tilde H = \mathrm{PT}_2[U^\dagger H U]$. The benefit is not fewer sites; it is that $U$ carries the strong-coupling entanglement, so $\Phi$ reaches matched accuracy at smaller $D_{\mathrm{eff}}$ — which is the only claim the A/B harness (§16, Phase 4) tests. Rejected alternative: coarse-grained PEPS on active sites only (irregular geometry, unbounded implementation risk).

### 9.1 Scales and loop (`rules.py`)

Active scale set: site scales $E_i = \sqrt{\bar\varepsilon_i^2 + \tilde\Delta_i^2}$ (with $\bar\varepsilon = \varepsilon + h^{\mathrm{MF}}$) and bond scales $|J_{ij}|$.

```
while Ω = max(scales) > omega_stop  and  n_decimated < f_max·N:
    if Ω is site scale E_i:  site_decimate(i)
    else bond scale |J_ij|:  bond_decimate(i, j)
    update scale heap; append op to circuit; update ledger
```

### 9.2 Site decimation (exact rotation + PT₂)

1. $u_i = \exp(-i\theta_i\sigma^y_i/2)$, $\theta_i = \mathrm{atan2}(\tilde\Delta_i, \bar\varepsilon_i)$; rotated local term $\tfrac{E_i}{2}\tilde\sigma^z_i$. Conjugation of couplings is **exact and 2-local**: $\sigma^z_i \to \cos\theta_i\,\tilde\sigma^z_i - \sin\theta_i\,\tilde\sigma^x_i$.
2. Pin $\langle\tilde\sigma^z_i\rangle = -1$. First order (kept exactly if `keep_first_order`, default True — the terms are 2-local and PEPS can see them; else absorbed): field shifts $\bar\varepsilon_j \mathrel{-}= 2 J_{ij}\cos\theta_i$ for all partners $j$.
3. Second order via the $\tilde\sigma^x_i$ channel, excitation energy $E_i$ — coefficients pinned to machine precision by the executed T-SDRG-3SITE (ADR-013): for unordered partner pairs $\{j,k\}\subset\mathcal N(i)$, $j\ne k$: $J_{jk} \mathrel{+}= -\,2\,J_{ji}J_{ik}\sin^2\theta_i / E_i$; the diagonal renormalizes the scalar offset $E_0 \mathrel{+}= -E_i/2 - \sin^2\theta_i\sum_j J_{ij}^2/E_i$ (ledgered).
4. Everything beyond PT₂ (norm of third-order corrections estimated as $\sum_{j,k}|J_{ji}J_{ik}|\,|J_{jk}|/E_i^2$-class terms) → `dropped_norm` (INV-8).

### 9.3 Bond decimation (cluster projection, $d$ stays 2)

For dominant $|J_{ij}|$, alignment sign $s = -\,\mathrm{sign}(J_{ij})$ (F for $J<0$, AF for $J>0$):

1. Project onto the aligned doublet (gap $2|J_{ij}|$ to the discarded sector). The doublet is an effective spin-$\tfrac12$ hosted on site $i$; site $j$ becomes pinned-trivial (identity slice with a recorded moment map).
2. Effective couplings on the cluster — validated by T-SDRG-3SITE (ADR-013) under this spec's $(\Delta/2)\sigma^x + J\sigma^z\sigma^z$ normalization: $\tilde\Delta_{(ij)} = -\,\tilde\Delta_i\tilde\Delta_j / (2|J_{ij}|)$ (PT₂; the sign is a $\tau^x$ gauge choice, the factor 2 is not), scalar offset $E_0 \mathrel{+}= -|J_{ij}| - (\tilde\Delta_i^2+\tilde\Delta_j^2)/(8|J_{ij}|)$, $\bar\varepsilon_{(ij)} = \bar\varepsilon_i + s\,\bar\varepsilon_j$ (exact within doublet), external bonds $J_{(ij),k} = J_{ik} + s\,J_{jk}$ (exact within doublet; pure algebra on the precomputed pair list — no geometric re-derivation needed).
3. Moment map $\mu_{(ij)} = \{\,i: +1,\; j: s\,\}$ recorded so $\langle\sigma^z_{i,j}\rangle$ of the physical spins is reconstructed from the cluster spin at measurement time (`circuit.pushforward`).
4. Projection error contribution: leaked weight $\sim (\tilde\Delta_{i}^2+\tilde\Delta_j^2 + \sum_k (J_{ik}-sJ_{jk})^2)/(2|J_{ij}|)^2$-class → `projection_error` ledger (INV-8).

### 9.4 Circuit and pushforward (`circuit.py`, `transform.py`)

`SDRGCircuit.apply_dagger(H_terms) -> (H̃_terms, ledger)` builds $\tilde H$ once. `circuit.pushforward(op_terms)` conjugates any observable through the ops in reverse for measurement on $\Phi$ (site rotations exact; cluster ops exact-2-local within doublet, leakage ledgered). If `ledger.total > τ_sdrg · H.norm_local` → INV-8 bypass: pipeline reruns with `circuit=None` and records the bypass. Stage A is therefore *never* able to make a run fail — only to fail to help.

---

## §10. Optimization layer (`optimize/`)

### 10.1 Initialization (`init.py`, `simple_update.py`)

Order: (1) product state from local ground directions of $\tilde H$'s on-site terms (each site: ground state of $\tfrac{\bar\varepsilon}{2}\sigma^z + \tfrac{\tilde\Delta}{2}\sigma^x$), $D$-padded with $10^{-2}$ noise; (2) optional simple-update warmup (`config.optimize.su_steps > 0`): imaginary-time bond updates on the **NN-truncated** part of $\tilde H$ only, Trotter $\tau: 0.5 \to 0.01$ geometric, per-bond local SVD to $D$ with environment-free weights (standard SU); SU energies are never reported (uncertified by construction — the factory refuses).

### 10.2 Gradient descent driver (`lbfgs_driver.py`)

`torch.optim.LBFGS(params=view_as_real(A_list), history_size=20, line_search_fn='strong_wolfe', max_iter=cfg.inner_iters)`. The closure computes the (differentiable) energy at the *current* $\chi$ with gates armed; on `EnvironmentNotConverged` the driver — outside the closure — raises $\chi \mathrel{+}= \Delta\chi$ (default $+D^2$), rebuilds environments, and restarts LBFGS warm (INV-1 retry ≤ 3). Stopping: $\mathrm{rel}\,\Delta E < 10^{-8}$ over 5 consecutive iterations **and** $\lVert g\rVert_2 < 10^{-6}\sqrt{2N D^4 d}$, or `max_outer` reached (flag `converged=False` in report).

### 10.3 $D$-ladder (`ladder.py`)

```
for D in cfg.ladder (default [2, 3, 4, 6]):
    chi = c * D**2
    state = state.grow(D) if state else init(D)
    state = lbfgs(state, H̃, chi)
    record E(D), wall, kernel stats
finalize: INV-2 chi-stability at final D; 1/D linear extrapolation of E(D) reported with fit residual
```

### 10.4 Finalization (`finalize.py`)

Runs INV-2 (energy re-evaluated at $2\chi$), stamps `certified`, measures observables (§12) once on the final state, calls `circuit.pushforward` for physical-frame observables, writes `StateArtifact`.

---

## §11. Ensemble + IO (`ensemble/`, `io/`)

**L1 parallelism only in v1** — embarrassingly parallel over realizations. `ProcessPoolExecutor(mp_context='spawn', max_workers=cfg.workers)`; each worker pins `torch.set_num_threads(total_cores // workers)`; task = `run_realization(master_seed, k, config) -> path`. Checkpoint granularity: after sampling, after SDRG, after each ladder rung, after finalize — `run.zarr/realizations/{k}/` is self-describing and resumable (`orchestrate.resume()` scans for the last completed stage marker). Crashes lose at most one rung.

`aggregate.py`: reads all certified artifacts (uncertified excluded unless `--allow-uncertified`, and then labeled), computes disorder means with $10^4$-resample bootstrap CIs for scalars ($q_{EA}$, $e$, resonance density) and pointwise-bootstrap bands for $\overline{C_{zz}}(r)$; writes `aggregate/` + a run-level `REPORT.md` echoing every invariant statistic (fallback rates, bypass counts, uncertified count).

`manifest.py` (INV-6): config hash (canonical-JSON SHA256), git SHA + dirty flag, package versions, master seed, per-realization spawned seeds, invariant thresholds. `tlsmbl verify run.zarr` re-runs every offline-checkable gate on stored artifacts and re-derives the manifest hash.

---

## §12. Observables (`observables/`) — two honesty tiers

**Tier 1 — rigorous functionals of the certified variational state:**
- $\langle\sigma^z_i\rangle, \langle\sigma^x_i\rangle$ (physical frame via pushforward); $q_{EA} = N^{-1}\sum_i \langle\sigma^z_i\rangle^2$.
- Connected correlator $\overline{C_{zz}}(r)$ averaged over pairs binned by $r$ (all pairs $r \le \min(L/2, 2R_c)$ using §8.4 caching); localization length $\xi$ from log-linear fit with bootstrap CI over disorder.
- SDRG flow diagnostics (when Stage A ran): decimation-scale sequence $\Omega_m$, cluster-size distribution — flow toward broadening distributions is the strong-disorder (localization) signature.
- Resonance census: pair count density $n_{\mathrm{res}}(r) = \overline{\#\{(i,j): r_{ij}\in[r,r{+}1),\ |E_i - E_j| < |J_{ij}|\}}$ — the finite-size delocalization proxy tied to the dipolar pair-resonance instability; this is the observable that operationalizes the "MBL is a finite-size statement here" caveat.

**Tier 2 — model-dependent decoherence estimates (every model input echoed into the report):**
- Effective fluctuator table: per site, physical-frame splitting $E_i$ and transverse weight $(\tilde\Delta_i/E_i)$ from $\tilde H$ + local state.
- $\Gamma_1(\omega_q) = \sum_i \dfrac{2 g_i^2\gamma_i}{\gamma_i^2 + (E_i - \omega_q)^2}$, with $g_i = g_0\,(\tilde\Delta_i/E_i)$ and a declared phenomenological TLS relaxation model $\gamma_i = \gamma_0 (E_i/W)^3 \coth(E_i/2T)$ — $g_0, \gamma_0, T, \omega_q$ are **inputs**, echoed verbatim; the static solver cannot derive $\gamma_i$ and the report says so in a fixed disclaimer field.
- Spectral-diffusion proxy: variance of qubit dispersive shift over the thermally-active fluctuator ensemble (formula documented in the module docstring; Tier-2 flag).

---

## §13. Configuration schema (`core/config.py`, pydantic v2; single YAML)

```yaml
# configs/benchmark.yaml — the acceptance configuration (matches the derivation's verdict regime)
run:      { name: bench-L16, master_seed: 20260715, n_realizations: 32, workers: 8, out: runs/bench.zarr }
model:    { L: 16, delta_min: 1.0e-3, g_J: 1.0e-3, R_c: 3, polaron_kappa: 0.0,
            hartree: { enabled: false, K_max: 8, alpha: 0.5, tol: 1.0e-4 } }
sdrg:     { enabled: true, omega_stop: 0.3, f_max: 0.4, keep_first_order: true, tau_sdrg: 0.05 }
peps:     { ladder: [2, 3, 4, 6], chi_factor: 1, dtype: complex128 }
env:      { eps_env: 1.0e-8, eps_env_E: 1.0e-7, polish: true, checkpoint_rows: true, retry_max: 3, dchi: auto }
kernels:  { backend: sketched, oversample: 8, power_iters: 1, eta: 1.0e-6, c_gate: 10.0, probes: 6,
            fallback_disable_rate: 0.20, eps_F: 1.0e-12 }
optimize: { su_steps: 200, inner_iters: 20, max_outer: 400, tol_E: 1.0e-8, tol_g_scale: 1.0e-6 }
invariants: { tau_chi: 1.0e-6, tau_tail: 0.02, allow_uncertified: false }
observables: { tier2: { enabled: true, omega_q: 5.0GHz, g0: 0.1MHz, gamma0: 1kHz, T: 30mK } }
```

Every field validated with ranges; unknown keys are a hard error (`extra='forbid'`). `configs/smoke.yaml`: `L: 4, ladder: [2,3], n_realizations: 2, backend: exact` — must finish < 5 min on 4 cores (CI job).

---

## §14. Testing strategy (`tests/`; IDs referenced by §16 exit criteria)

1. **T-GOLD-ED (golden, blocking Phase 2).** For seeds {0,1,2}, $L\in\{3,4\}$, two regimes: (a) weak $g_J=10^{-3}$: require $|E_{\mathrm{PEPS}} - E_{\mathrm{ED}}| / |E_{\mathrm{ED}}| \le 10^{-8}$ at $D=3$ (state nearly product — if this fails, wiring is wrong, not physics); (b) strong $g_J=0.3$: $\le 10^{-4}$ at $D=6, \chi=64$, and site-resolved $\langle\sigma^{z,x}_i\rangle$ match ED $\le 10^{-3}$ absolute.
2. **T-GOLD-XCHECK.** Same instances contracted by quimb (tests-only dependency) agree with our environment norm to $10^{-9}$ — isolates contraction bugs from optimization bugs.
3. **T-SDRG-3SITE (blocking Phase 4).** 3-site chain, dominant center field: PT₂ rules of §9.2 reproduce the exact Schur-complement downfolded $2\times2$-sector couplings to $O(J^3/E^2)$; measured convergence order must be $\ge 3$ under $J$-scaling. Bond decimation analog: 2-site strong bond + probe site, doublet projection vs exact.
4. **T-AD-FD (blocking Phase 2).** Central finite differences vs autograd on $L=3, D=2, \chi=8$: relative agreement $10^{-6}$, run for both backends and with/without checkpointing; includes degenerate-singular-value construction to exercise $\epsilon_F$ broadening.
5. **T-PROP-DIST.** hypothesis: KS test of $\ln\Delta$ uniformity; pair-list symmetry/cutoff correctness; lazy tail stream reproducibility.
6. **T-DET.** Same `(master_seed, k)` twice ⇒ identical `report.json` and bit-identical tensors (deterministic mode).
7. **T-EQ-BACKENDS (blocking Phase 3/6).** Exact vs sketched: same operand ⇒ subspace angle between truncated column spaces $\le 10^{-6}$ when posterior gate passes, and energies through full pipeline agree $\le 10\,\tau_\chi$; torch vs Rust: $10^{-10}$ (exact mode), $10^{-8}$ (sketched, shared seeds).
8. **T-INV-*.** One test per invariant proving the failure action fires (constructed divergent environment for INV-1, injected NaN for INV-7, adversarial slow-decay operand for INV-3 fallback, oversized ledger for INV-8 bypass, etc.). `EnergyReport()` direct construction must not compile/must raise.
9. **T-PERF.** pytest-benchmark on the kernel: exact vs sketched at $D\in\{2,3,4,6,8\}$, $\chi=D^2$; CI asserts fitted exponent gap $\ge 1.6$ in $D$ (theory: 2.0) and no >15% regression run-over-run.

**Validation provenance (executed prototypes).** The D3 kernel benchmark and a 3×3 golden battery (51/51 checks) have already been executed against this spec: ED oracle verified (sparse ≡ dense to 10⁻¹⁵, J=0 analytic sum to 10⁻¹⁶); einsums E-1..E-5, zip-up compression, dressed-environment cross-row correlators (§8.4), full energy assembly, the INV-7 guard, and sampler determinism all verified against an independent brute-force statevector oracle to ≤ 3×10⁻¹⁵, with identically zero discarded weight at χ=D². Two spec amendments resulted (ADR-009, ADR-010). Phase-2 execution (JAX substrate, ADR-011): the differentiable energy graph reproduces the certified numpy engine to 3×10⁻¹⁵; T-AD-FD passes at 8×10⁻¹⁰ (lossless χ) and 4.7×10⁻⁹ (χ=2, discarded weight 1.9×10⁻²); LBFGS on 3×3 reaches relative ED gaps of 7×10⁻⁹ (g_J=10⁻³, D=2, product init, 203 iters — §10.1 init empirically validated), 2.6×10⁻⁷ (g_J=0.3, D=2) and 7.2×10⁻⁸ (g_J=0.3, D=3, monotone in D), all variational-bound-respecting, ≈7 ms/iter (D=2) post-compile. The prototype files `bench_kernel.py` and `golden_3x3.py` are normative seeds for `kernels/` and `tests/golden/` respectively; `ad_phase2.py` is the normative seed for `peps/autodiff.py` and `optimize/`. 4×4 execution (T-GOLD-ED at L=4): sparse-Lanczos ED at 2¹⁶ verified (eigen-residual 10⁻¹⁴, J=0 analytic 2×10⁻¹⁴, 98 dipolar pair terms, ~2 s); engines agree to 3×10⁻¹⁴ at the provably lossless χ=16, while at truncating χ on *random* states they disagree at √(disc) scale — flat-spectrum truncation is gauge-ambiguous, so **cross-engine agreement is not a valid certificate under truncation; INV-2 stability is** (this motivates ε_F broadening and the certificate design); T-AD-FD holds at 4×4 through real truncation (2.9×10⁻⁹ at disc 3.8×10⁻⁵); T-GOLD-ED: g_J=0.3 D=2 relative gap 8.9×10⁻⁸ (4000 iters, optimizer-floor-limited, ≈2 ms/iter), g_J=10⁻³ 6.4×10⁻¹⁰, truncated-path χ=6 run reaches 2.0×10⁻⁷ with INV-2 stability 1.4×10⁻¹¹ and discarded weight collapsing to 2×10⁻¹⁵ on the optimized state — direct empirical confirmation of the localized-phase spectral-decay premise underlying Stage B and INV-3. `phase3_4x4.py` is the normative seed for the L=4 golden tests. T-SDRG-3SITE executed: 9/9 (three random draws × site / AF-bond / F-bond decimation); rule↔Schur identities at ~2×10⁻¹⁶; local eigenvalue convergence orders 2.9–3.3 tending to 3.00; two factor-2 coefficient errors in §9's original prose discovered and corrected in place (ADR-013); `sdrg_3site.py` is the normative seed for `sdrg/rules.py` and its golden test. Converged D≥3 runs at L=4 are deferred to production CI (per-iteration cost exceeds this container's session budget — precisely the regime the sketched kernel and multicore target address).

**Production port provenance (July 2026, torch).** P0–P5 implemented in `src/tlsmbl/`
with all §16 exit gates green in the test suite (124 tests): P0 lint/mypy-strict/T-DET
config hashing/INV-6 refusal; P1 sampler bitwise-parity with the prototype oracle,
exact H-matrix identity, 12 stored ED fixtures (ADR-014); P2 T-GOLD-ED (weak 1e-7,
strong 1e-5 full-gold), T-GOLD-XCHECK (quimb, 1e-9), T-AD-FD (1.3e-9 lossless /
4.8e-9 truncating), T-INV-1/2/4/7, §8.5 hardened SVD backward validated against
native vjp + FD (measured finding: at *exact* degeneracy no backward computable from
(gU,gS,gVh) matches FD — native torch fails too; contract is exactness above ε_F,
finite bounded gradient below); P3 T-INV-3 (ADR-009 case optimal to 1e-9, slow-decay
fallback), T-EQ-BACKENDS (angle < 5e-6 rad on the prototype instance), T-PERF
exponent gap 4.16 (gate ≥ 1.6); P4 T-SDRG-3SITE via prototype Schur oracle (Tier-I
1e-12, Tier-II order ≥ 2.9; two composition defects found and fixed: pinned-term/E0
double-count, unrotated pinned-site bonds), T-INV-8 bypass, A/B harness; P5
resume-after-kill (loses ≤ 1 rung), ensemble T-DET (bitwise), INV-5 gate + audit
CLI. P6/P7 remain conditional and unopened.

---

## §15. Performance model and budgets

| Component | Time | Memory |
|---|---|---|
| Zip-up, exact SVD | $\Theta(L^2\,\chi^3 D^6) \to \Theta(L^2 D^{12})$ at $\chi=D^2$ | $\Theta(\chi^2 D^4) = \Theta(D^8)$ |
| Zip-up, sketched | $\Theta(L^2\,\chi^3 D^4) \to \Theta(L^2 D^{10})$ | $\Theta(\chi^2 D^6)$ v1 (fat tensor); $\Theta(\chi^2 D^4)$ v1.1 factored (ADR-015) |
| Energy w/ correlators | $O(N R_c L\,\chi^3 D^4)$-class (dressed-environment reuse, §8.4) | reuses `EnvBundle` |
| AD backward | $\le 2\times$ forward w/ row checkpointing | $O(L \chi^2 D^4)$ activations |
| ED oracle | $L{=}4$: seconds; $L{=}5$ matrix-free: minutes, ~3–5 GB | — |

**Acceptance regime** ($L{=}16, D{=}6, \chi{=}36$): one exact kernel call $\approx (\chi D^2)^3 = 1296^3 \approx 2.2\times10^9$ flop; $\sim 2L^2$ calls/energy, $\times$ optimizer iterations $\Rightarrow 10^{13-14}$ flop/realization — minutes-to-hours single-node, memory hundreds of MB. Sketched target: $\ge 20\times$ kernel speedup at $D=6$ (theory $\sim D^2 = 36\times$; accept BLAS-constant erosion). **Budget gates:** smoke < 5 min; bench realization < 4 h/worker at `backend: sketched`; if the $1/D$ extrapolation demands $D \ge 10$ ($\Theta(D^8)$ memory $\to$ tens of GB), *only then* open Phase 7: L2 column-strip environment sharding + L3 TSQR (interfaces stubbed in `kernels/interface.py::ShardedBackend`, compute/comm ratio $\chi D^4$ per the derivation; not implemented in v1).

---

## §16. Phased implementation plan (entry/exit gates)

| Phase | Scope | Exit criteria (all CI-enforced) |
|---|---|---|
| P0 | scaffold: core types, units, rng, guards, config, manifest, CLI skeleton | lint+mypy clean; T-DET on config hashing; INV-6 refusal test |
| P1 | model layer + ED oracle + Tier-1 observables on ED states | T-PROP-DIST; ED reproduces analytic 2-site cases; T-GOLD fixtures generated & stored |
| P2 | finite PEPS, boundary env, exact backend, AD energy, LBFGS, ladder | **T-GOLD-ED, T-GOLD-XCHECK, T-AD-FD, T-INV-1/2/4/7**; smoke.yaml green |
| P3 | sketched backend + gates + microbench | T-EQ-BACKENDS(torch), T-INV-3, T-PERF exponent gap |
| P4 | SDRG circuit + transform + ledger + **A/B harness** (`tlsmbl ab-test`: matched-accuracy $D_{\mathrm{eff}}$ vs $D$, ≥16 realizations) | T-SDRG-3SITE, T-INV-8; A/B report generated (a *negative* result is an acceptable exit — Stage A is quarantined) |
| P5 | ensemble orchestration, checkpoint/resume, aggregate, Hartree loop | resume-after-kill test; T-DET at ensemble level; INV-5 wired |
| P6 (cond.) | Rust zip-up kernel via pyo3 | T-EQ-BACKENDS(rust); ≥3× wall-clock over torch-sketched at $D{=}6$ |
| P7 (cond.) | L2/L3 sharding — **only if** $D\ge10$ required by extrapolation | design doc + fresh ADR first; out of v1 scope |

---

## §17. ADR index

ADR-001 boundary-MPS over CTMRG (disorder ⇒ inhomogeneous finite lattice) · ADR-002 SDRG pinned-full-lattice composition over irregular coarse lattice · ADR-003 AD-through-contraction over hand-derived environment gradients (fallback path documented in `autodiff.py` header) · ADR-004 complex128 certification dtype · ADR-005 zarr over HDF5 (concurrent-writer resumability) · ADR-006 zip-up + single polish sweep over full variational compression (polish flag) · ADR-007 explicit `Wmat` v1, factored v1.1 (memory trigger; realized per ADR-015) · ADR-008 Rust FFI boundary = whole row loop, AD stays in torch · ADR-009 two-sided INV-3 gate — sketch quality judged vs the achievable rank-χ floor, rank adequacy delegated to INV-1 (defect found and fixed via the executed D3 benchmark; see §8.6) · ADR-010 canonicalize-then-truncate compression — naive zip-up truncates gauge artifacts rather than Schmidt spectra, corrupting both accuracy and the INV-1 certificate (defect found and fixed via the executed 3×3 golden test; see §8.3) · ADR-011 AD-graph canonicalization is SVD-based (framework QR-VJPs lack wide-matrix support); Phase-2 prototype executed in JAX because the egress allowlist blocks torch CPU wheels and the PyPI torch wheel is CUDA-linked — validated math is framework-portable, production remains torch · ADR-012 exact TruncSVD backward = full-SVD-then-slice, exact for the bilinearly-recombining compression loss including kept↔discarded coupling (FD-validated to 5×10⁻⁹ under heavy truncation); the projector formula is the sketched-backend fallback with error O(σ_{χ+1}/σ_χ), coherent with INV-3. · ADR-013 SDRG PT₂ coefficients pinned by executed T-SDRG-3SITE — both prose rules carried factor-2 errors (pair generation and cluster tunneling); validated forms equal fixed-denominator Schur downfolding to 2×10⁻¹⁶ with measured eigenvalue convergence order → 3.00 across site, AF-bond, and F-bond decimation. · ADR-015 factored compression (ADR-007 v1.1) realized via bond-Gram canonicalization — the original LinearOp sketch predates ADR-010 and would truncate the uncanonicalized operand; the Gram-carried gauge preserves the Schmidt spectrum at the same flop class while eliminating the Θ(χ²D⁶) fat tensors (measured: D=4 row peak RSS 0.86→0.31 GB; D=6, χ=36 row feasible in 1.17 GB where v1 needs ~8 GB); see `docs/adr/adr-015-factored-gram-compression.md`.

## §18. Risk register

| Risk | Mitigation |
|---|---|
| AD memory blowup at $L{=}16, D{=}6$ | row checkpointing (default on); v1.1 factored matvec; fallback ADR-003 path |
| Complex-SVD backward instability (degenerate $\sigma$) | INV-9 gauge fix + $\epsilon_F$ broadening + T-AD-FD degenerate case |
| Disordered landscape ⇒ local minima | SDRG/product init + SU warmup + multi-start (`n_starts` config) keeping best certified energy |
| PT₂ ledger blows up at strong coupling | INV-8 auto-bypass; Stage A can only fail to help |
| Sketch gate thrashing (slow spectral decay) | INV-3 fallback + per-realization disable at 20% rate |
| Dipolar pair resonances destabilize localization at large $L$ | not a bug: report $n_{\mathrm{res}}(r)$, frame all claims as finite-size at declared $(L, R_c)$; $R_c$ and $L$ sweeps in study plan |
| Tail truncation bias | INV-5 bound + Hartree loop when enabled |

## §19. Novelty register (candidate invention disclosures)

NR-1 **Posterior-gated randomized truncation inside a differentiable tensor-network contraction** (INV-3 + §8.5/§8.6: certified fallback within the AD graph). NR-2 **SDRG-pinned-lattice PEPS composition with fidelity ledger and automatic bypass** (§9, INV-8). NR-3 **Gate-constructible certified result artifacts** for variational solvers (§3, `EnergyReport` factory pattern). NR-4 **Operator-dressed boundary-MPS caching for finite-range dipolar correlators** (§8.4). NR-5 **Reproducible lazy tail-coupling streams** for long-range disorder without $O(N^2)$ storage (§7.4). Each maps to a standalone spec in the format of the krylov-solver invention portfolio; NR-1 and NR-2 are the strongest candidates.

## §20. CLI and acceptance

```
tlsmbl run configs/benchmark.yaml            # full pipeline, resumable
tlsmbl verify runs/bench.zarr                # offline invariant re-check (INV audit)
tlsmbl bench kernels --D 2 3 4 6 8           # D-scaling microbenchmark → D3 deliverable
tlsmbl ab-test configs/ab_sdrg.yaml          # Stage-A value measurement → D4 deliverable
tlsmbl aggregate runs/bench.zarr             # D2 deliverable + REPORT.md
```

**Definition of done (v1):** P0–P5 exit criteria green; `benchmark.yaml` completes with ≥ 90% certified realizations; `verify` clean; `REPORT.md` contains $E(D)$ extrapolation, $q_{EA}$, $\xi$, $n_{\mathrm{res}}(r)$, Tier-2 $\Gamma_1$ with echoed model inputs, and the full invariant audit.
