# Theory: why Dirichlet-energy spectral metrics are *selectively* blind to the format confound

This is the theoretical spine of the tight claim. It explains — and predicts — the empirical
observation that mean-pooled activation probes fail the eval-awareness **negative control**
(deploy-vs-placebo) while Dirichlet-energy spectral metrics do not.

## Setup

Fix the (byte-identical, position-matched) task tokens. Let `X ∈ ℝ^{n×d}` be their hidden-state
matrix at a layer (`n` task tokens, `d` hidden dims). Let `W ∈ ℝ^{n×n}` be the symmetrized,
head-aggregated attention among those tokens, `D = diag(W·1)` the degree matrix, and
`L = D − W` the (combinatorial) graph Laplacian. Note `L·1 = 0` and `1ᵀL = 0`.

The **Dirichlet energy** of the signal is
`E(X) = tr(Xᵀ L X) = Σ_{k≤d} Σ_{i<j} W_ij (X_ik − X_jk)²`,
i.e. a purely **relational** quantity: it only sees *differences between tokens*. The repo's
`smoothness_index`, `spectral_entropy`, and `hfer` are functions of this operator / its spectrum.

## Theorem 1 (DC-invariance of Dirichlet energy)

For any constant vector `α ∈ ℝ^d`, `E(X + 1_n αᵀ) = E(X)`.

**Proof.** `E(X + 1αᵀ) = tr((X+1αᵀ)ᵀ L (X+1αᵀ))
= tr(XᵀLX) + 2·tr(αᵀ 1ᵀ L X) + tr(αᵀ 1ᵀ L 1 α)`.
Since `1ᵀL = 0` and `L1 = 0`, both the cross term and the quadratic term vanish. ∎

**Decomposition.** Write `X = DC(X) + AC(X)` where `DC(X) = 1_n·mean_tok(X)ᵀ` (the rank-1,
token-uniform component) and `AC(X) = X − DC(X)` (token-mean-zero, relational). Because
`DC(X)` is a constant per token, Theorem 1 gives **`E(X) = E(AC(X))` exactly**: Dirichlet-energy
metrics read *only* the AC component and are identically blind to DC.

By contrast, a **mean-pooled activation probe** uses `mean_tok(X) = DC` (up to the `1_n` lift),
so it is a function of *only* the DC component — the exact complement of what the spectral
metrics see.

## Theorem 2 (spectral complementarity & frequency-graded suppression) — the real one

Theorem 1 is the `k=0` corner of a stronger statement. Let `L = Φ Λ Φᵀ` with eigenpairs
`(λ_k, φ_k)`, `0 = λ_0 ≤ λ_1 ≤ …`, and graph-Fourier transform `X̂ = Φᵀ X` (rows `x̂_k ∈ ℝ^d`).

**(i) Mean-pool is the DC read-out.** For the combinatorial `L`, `φ_0 ∝ 1`, so the mean-pool
statistic `m(X) = (1/n)1ᵀX ∝ x̂_0`. A linear probe on `m(X)` reads *only* the DC mode `x̂_0`.

**(ii) Dirichlet energy is DC-annihilating and frequency-graded.**
`E(X) = tr(XᵀLX) = Σ_k λ_k ‖x̂_k‖²`. Thus `∂E/∂x̂_0 = 0`, and a perturbation `Δ` whose graph-spectral
support lies in `{k : λ_k ≤ ε}` moves the energy by at most `|E(X+Δ)−E(X)| ≤ 2ε‖X‖‖Δ‖ + ε‖Δ‖²` —
**suppressed in proportion to its graph frequency**, not merely at DC.

**(iii) Complementarity.** Mean-pool (ideal low-pass at DC) and Dirichlet energy (the quadratic
high-pass `Σ λ_k`) are *orthogonal spectral filters*: every signal component is registered by at
most one, graded by frequency. A mean-pooled probe is maximally sensitive to exactly the low
frequencies the energy metric suppresses.

*Proof.* (i) `φ_0 ∝ 1` because `L1=0`; `Φ` orthonormal ⇒ `x̂_0 = φ_0ᵀX ∝ 1ᵀX`. (ii) `XᵀLX =
XᵀΦΛΦᵀX = X̂ᵀΛX̂ = Σ_k λ_k x̂_k x̂_kᵀ`; take the trace. The bound is Cauchy–Schwarz on
`E(X+Δ)−E(X) = 2⟨X,LΔ⟩ + ⟨Δ,LΔ⟩` with `‖L|_{span φ_{k≤K}}‖ ≤ ε`. ∎

### The confound model and the falsifiable prediction (from Theorem 2)

**Sharp prediction (frequency separation).** A frame is a *global, slowly-varying* contextual
influence on the task tokens, so its effect should concentrate at **low graph frequency** (small
`λ`). Decompose the shift `Δ = X_a − X_b` in the (common, deploy-graph) Fourier basis and define its
**graph-spectral centroid** `c(Δ) = Σ_k λ_k ‖Δ̂_k‖² / Σ_k ‖Δ̂_k‖²` (mean graph frequency of the
shift) and its **DC mass fraction** `f_DC = ‖Δ̂_0‖² / Σ_k ‖Δ̂_k‖²`. If the *format/wording* confound
is graph-smoother than genuine *evaluativeness* (which reshapes token-relational structure), then:

> **(P4)** `c(eval·deploy) > c(deploy·placebo)` and `f_DC(eval·deploy) < f_DC(deploy·placebo)`:
> the evaluativeness shift is *higher graph-frequency* than the pure-confound shift.

P4 is a single-number, falsifiable consequence of Theorem 2 that does **not** reduce to the DC
special case — it predicts a *graded* spectrum, and can fail (if evaluativeness is also low-frequency,
`c` would be equal and the account is wrong). The DC/AC crossover (below) is the coarse `f_DC` version.

The coarse form: model the shift as `Δ = b · 1_n uᵀ + R`, a **token-uniform bias** `b·1uᵀ` (the
low-frequency format/context component) plus a **relational residual** `R` (`mean_tok(R) ≈ 0`).

- A mean-pooled / DC probe reads `b` (the uniform bias) and is blind to `R`.
- A Dirichlet-energy / AC / spectral probe reads `R` and is blind to `b` (Theorem 1).

**Prediction P.** If the *non-evaluative* confound (what the negative control deploy-vs-placebo
measures: wording / length / filler / position) is **DC-dominated**, while genuine
*evaluativeness* carries substantial **AC** (relational) content, then:

- **(P1)** DC probes are *confounded*: selectivity `S = AUC(eval·deploy) − AUC(deploy·placebo) ≈ 0`
  (they separate the negative control as well as the real contrast).
- **(P2)** AC and spectral probes are *selective*: `S > 0`.
- **(P3)** Spectral selectivity ≈ activation-**AC** selectivity (both capture `R`), and both
  exceed DC selectivity — pinning the effect to the DC/AC split rather than to "spectral magic".

This is falsifiable: if the confound is *not* DC-dominated, DC probes would be selective too and
`S(DC) ≈ S(AC)`, refuting the account.

## What this buys the paper (one tight claim)

> **The residual-stream eval-vs-deploy shift on fixed task tokens splits into a token-uniform
> component that carries the format confound and a token-relational component that is
> evaluativeness-selective. Dirichlet-energy spectral metrics are provably invariant to the
> former (Theorem 1), which is exactly why they are more selective than mean-pooled activation
> probes.** We verify P1–P3 across models.

Companion (optional second claim): the relational (AC / spectral) selectivity **scales** with model
size (the field's power-law), tested on the within-family ladders in `configs/models_scaling.json`.

## Theorem 2 prediction P4 is FALSIFIED (Llama-3.2-3B, N=15, symmetric basis)

Theorem-first, then validated — and it failed. The graph-frequency spectrum of the shift:

| quantity | eval-shift | confound-shift | predicted | result |
|---|---|---|---|---|
| centroid `c` | 0.485 | 0.505 | eval > | eval **<** (Wilcoxon p=0.99, 2/15 support) |
| DC fraction `f_DC` | 0.270 | 0.251 | eval < | eval **>** (p=0.98) |

The evaluativeness and pure-confound shifts have **nearly identical graph-frequency spectra**;
if anything evaluativeness is *slightly more* DC-concentrated — the opposite of P4. Verified under
the symmetric normalized Laplacian (orthonormal eigenbasis, valid Parseval); an earlier `rw` run
(non-orthonormal basis, invalid) gave the same sign.

**Consequences (honest):**
- The identities in Theorem 2 (i–iii) hold, but the **mechanistic premise is false**: a
  signal-frequency-separation account does **not** explain any spectral selectivity, because the two
  shifts are spectrally indistinguishable.
- Together with the single-model, non-replicating crossover below, the defensible conclusion is the
  **negative/methodological** one: *after position-matching + selectivity controls, the eval-awareness
  probe signal is format-confound-dominated; the residual relational signal is weak, scale-gated, and
  carries no distinctive graph-frequency signature.* This directly answers the field's open question
  ("Is evaluation awareness just format sensitivity?") — largely yes; spectral geometry does not rescue it.
- Remaining untested locus (a hypothesis, **not** a claim): the eval-selective information, if any, may
  reside in the **attention graph** `W` itself (which spectral metrics see and mean-pool does not),
  not in the signal `X`. This would need a graph-change test; do not assert it without one.

## Validation status (small-scale, N=15 tasks, position-matched) — HONEST

The crossover prediction is **confirmed clearly in one model and not yet replicated**:

| model | S(DC) | S(AC) | S(spectral) | spectral eval p |
|---|---|---|---|---|
| Qwen2.5-0.5B | +0.03 | −0.04 | +0.02 | 0.42 (no eval signal at all) |
| Llama-3.2-1B | **−0.16** | −0.13 | **+0.09** | 0.26 (weak, right direction) |
| Gemma-2-2B | 0.00 | −0.02 | **−0.10** | 0.17 (counterexample: spectral anti-selective) |
| **Llama-3.2-3B** | −0.07 | +0.13 | **+0.15** | **0.007** ✓ clean P1–P3 |

Theorem 1 is unconditional (math); what is **not yet established** is the empirical premise
(confound = DC, evaluativeness = AC) beyond Llama-3.2-3B. Small models (0.5–2B) show no
significant eval signal at N=15 (consistent with the eval-awareness power-law), so there is
nothing to decompose; only the 3B model clears significance. **The claim currently rests on a
single model** and must be established at scale.

**Falsifiable targets for the scale-up (H100):** (T1) the crossover holds at ≥7B with N≥60 tasks,
p<0.05 per model; (T2) selectivity S(spectral) increases monotonically with model size (below-
threshold → clear), tying the small-model nulls to the power-law rather than to failure; (T3)
Gemma is not a systematic counterexample (check Gemma-2-9B). If T1–T3 fail, the honest paper is the
*negative/methodological* one: "probe-based eval-awareness is DC-format-confounded; the relational
signal is weak and scale-gated."

## Measurables (see `decomposition.py`)

- DC/AC **shift magnitudes**: `‖DC(Δ)‖`, `‖AC(Δ)‖` for eval-vs-deploy and for the negative control.
  Prediction: the negative-control shift is *more* DC-dominated than the eval shift.
- The **probe crossover** P1–P3: `S` for DC-probe (mean_norm), AC-probe (token-variance,
  participation ratio — both provably DC-invariant), and spectral (smoothness+hfer).
