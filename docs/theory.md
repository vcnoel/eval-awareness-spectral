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

## The confound model and the falsifiable prediction

Model the frame's causal effect on the fixed task tokens, `Δ = X_eval − X_deploy`, as
`Δ = b · 1_n uᵀ + R`, a **token-uniform bias** `b·1uᵀ` (all task tokens shift alike because they
all attend to the *same* surrounding frame — a "format/context DC bias") plus a **relational
residual** `R` with `mean_tok(R) ≈ 0`.

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

## Measurables (see `decomposition.py`)

- DC/AC **shift magnitudes**: `‖DC(Δ)‖`, `‖AC(Δ)‖` for eval-vs-deploy and for the negative control.
  Prediction: the negative-control shift is *more* DC-dominated than the eval shift.
- The **probe crossover** P1–P3: `S` for DC-probe (mean_norm), AC-probe (token-variance,
  participation ratio — both provably DC-invariant), and spectral (smoothness+hfer).
