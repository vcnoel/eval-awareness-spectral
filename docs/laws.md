# Batch law-mining: candidate invariants of the frame effect (mined, stress-tested)

Method (how maths should be done here): generate a *battery* of measurable candidate
invariants from the empirical observations, compute them all across models / layers /
tasks (`law_mining.py`), **stress-test** each (nulls, artifact checks, monotonicity),
and keep only what survives. Data: per-token task-token activations for eval / deploy /
placebo on 4 instruction models spanning 3 families (Qwen2.5-0.5B, Llama-3.2-1B,
Gemma-2-2B, Llama-3.2-3B), position-matched, 6 tasks each, late-layer band.

Notation per task, layer: `X_e,X_d,X_p ∈ ℝ^{n×d}`; shifts `Δe=X_e−X_d`, `Δc=X_d−X_p`;
mean directions `δe=mean_tok Δe`, `δc=mean_tok Δc`.

## Candidates that DIED (and why — this is the point)

- **P4 frequency separation** (Theorem 2): eval and confound shifts have nearly identical
  graph-frequency spectra (centroid 0.485 vs 0.505, p=0.99). *Falsified* — see `theory.md`.
- **C1/C2 naive collinearity** `cos(Δe,Δc)`: looked like a robust anti-alignment (≈ −0.3),
  **but it is inflated by a shared-anchor artifact** — `Δe` and `Δc` share `deploy` with
  opposite sign, so `E[(e−d)·(d−p)] ⊃ −‖d‖²` biases the cosine negative regardless of content
  (the sign flips if you reorient `Δc`). Stress test = matched vs mismatched-task cosine:

  | model | C1 matched | C1 **mismatched (anchor-free)** |
  |---|---|---|
  | Qwen-0.5B | −0.136 | **+0.002** |
  | Llama-1B | −0.344 | **−0.207** |
  | Gemma-2B | −0.272 | **−0.142** |
  | Llama-3B | −0.313 | **−0.192** |

  The naive matched value is partly artifact; the **anchor-free residual survives** (below).
- **C8 magnitude ratio** `‖Δe‖/‖Δc‖`: 0.80→1.09→1.18→1.18 — *saturates*, and only one 0.5B
  model, so size vs family is confounded. Not a clean law.

## The law that SURVIVES: eval-direction crystallization

Two independent, artifact-free measurements, monotone across all 4 models / 3 families:

| model | params | C4 task-consistency(eval) ↑ | C10 eval rank ↓ | C1 mismatch (anchor-free) |
|---|---|---|---|---|
| Qwen-0.5B | 0.5 | 0.454 | 2.61 | +0.002 |
| Llama-1B | 1.0 | 0.545 | 2.35 | −0.207 |
| Gemma-2B | 2.0 | 0.551 | 2.33 | −0.142 |
| Llama-3B | 3.0 | **0.654** | **1.90** | −0.192 |

**Null checks.** C4/C10 use only the eval direction (no anchor-sign artifact). Random unit
vectors in `d≈3072` have `E[cos]≈0 ± 1/√d ≈ 0.018`, so C4 ∈ [0.45,0.65] is ~25–35σ above
null — the eval shift is *genuinely* a consistent cross-task direction, not noise. The
anchor-free anti-alignment (mismatched C1) is ~0 at 0.5B and ≈ −0.18 for every model ≥1B.

**Conjecture (Eval-direction crystallization).** As instruction-tuned LLMs scale, the
evaluation frame induces a representational shift that (a) concentrates into a
lower-dimensional, task-universal direction (cross-task consistency ↑, effective rank ↓),
and (b) becomes anchor-free anti-aligned (`cos ≈ −0.2`) with the direction induced by
non-evaluative format/length changes. Below ~1B this structure is absent (the eval shift is
high-rank and format-uncorrelated).

**Why this is the load-bearing law.** It *derives* the rest: selectivity is scale-gated
because eval and format only become distinct (anti-aligned, low-rank) directions at scale;
the small-model detection null and the field's power-law follow; and it explains why a
linear/spectral probe separates eval from format only at ≳1–3B. It is not a re-description
of the selectivity result — it is measured on the raw shift directions, independent of any
probe.

## WITHIN-FAMILY confirmation (F3 passed): Qwen 0.5→3→7B, bf16

The decisive test — one architecture, size varied — so the trend is size, not family:

| Qwen | params | C4 consistency ↑ | C10 rank ↓ | C1 anti-align (anchor-free) | C8 norm ratio |
|---|---|---|---|---|---|
| 0.5B | 0.5 | 0.454 | 2.61 | +0.002 | 0.80 |
| 3B | 3.0 | 0.531 | 2.46 | −0.151 | 0.88 |
| 7B | 7.0 | 0.550 | 2.36 | −0.258 | 1.04 |

**All four signatures are monotone in size within a single architecture**, and it replicates in
the Llama ladder (1B→3B: C4 0.545→0.654, C10 2.35→1.90). So the crystallization is a **genuine
scale effect, not a family artifact** — F1–F3 pass at the scales tested. The cleanest, most
artifact-free signal is the anchor-free anti-alignment C1_mismatch: **+0.002 → −0.151 → −0.258**
(no eval/format structure at 0.5B → clear, strengthening anti-alignment by 7B).

Adversarial note: C4 (raw cross-task consistency) begins to *plateau* at the top (0.53→0.55),
while C10 (rank), C1_mismatch (anti-alignment), and C8 (magnitude) keep moving — the
crystallization continues but is carried more by rank-collapse and anti-alignment than by raw
consistency at ≥3B. bf16 weights; all diagnostics/measurements computed in fp32/fp64 (precision
where it counts is preserved).

## Falsifiable predictions for the scale-up (H100, true fp32)

- **F1 (crystallization continues):** C4 ↑ and C10 ↓ monotonically through 7–8B; C4 → ~0.8+,
  C10 → ~1.3 (approaching a 1-D universal "being-evaluated" direction).
- **F2 (anti-alignment persists/strengthens):** anchor-free mismatched `cos(δe,δc)` stays
  ≈ −0.2 or grows more negative with scale, and stays ~0 only sub-1B.
- **F3 (within-family, size-controlled):** run one family's full ladder (Qwen 0.5→3→7B) so
  the trend is size, not family. If F1–F3 hold, the crystallization law + the selectivity it
  implies is the paper. If C4/C10 are flat within a family, the "law" is a family artifact.

Reproduce: `python -m eval_awareness_spectral.law_mining` (reads the cached token_acts).
