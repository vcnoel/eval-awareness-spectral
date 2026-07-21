# eval-awareness-spectral

**Do a language model's spectral internals shift when it believes it is being *evaluated* rather than *deployed*?**

This repo takes a set of identical base tasks, wraps each one in different *situational frames* (a benchmark/test frame vs. an ordinary-deployment frame vs. controls), and pushes them through the [`spectral-trust`](https://pypi.org/project/spectral-trust/) graph-signal-processing diagnostics. It then measures, **layer by layer**, how far apart the frames move the model's internal representations — and whether that shift is statistically reliable across tasks and *decodable* by a probe.

The premise (from a conversation with Jasmine Li): "evaluation awareness" — models behaving differently when they detect they are under test — should leave a *geometric* fingerprint in the attention graph and hidden-state spectrum, not just in the output text. This repo is an instrument for looking for that fingerprint **without fooling ourselves about length and formality confounds**.

> Content is held **byte-identical** across conditions; only the frame changes. Any spectral difference is therefore attributable to the frame, not the task.

We *use* `spectral-trust` as the measurement instrument — every number comes from `framework.analyze_text(...)`. We do not re-derive its graph pipeline.

---

## Pinned to spectral-trust 0.2.2

```bash
conda activate gemma_spectral
pip install spectral-trust==0.2.2 scikit-learn
pip install -e .        # optional: exposes the CLI
```

0.2.2 is pinned exactly. It changed the measurement surface relative to earlier versions, so **results are not comparable across `spectral-trust` versions** and every diagnostics row is tagged with `st_version`. Notable for this repo:

- `SpectralDiagnostics` no longer carries `attention_gini` (it needs the aggregated adjacency, which `analyze_text` doesn't expose — recomputing it would mean rebuilding library internals, so we don't). We **do** keep `gini_sparsity`, computed here as the Gini of the library's own returned `spectral_masses` — a read-out of the instrument's output, not a re-derivation. Rows carry `gini_source` to make that provenance explicit.
- The random-walk Laplacian eigensolver was corrected upstream, so absolute numbers differ from the v1 pilot. **The old v1 headline (`gini_sparsity` significant in 26/26 layers) is retired**, not silently replaced; it will be re-established or refuted by the clean 0.2.2 re-run.

Everything runs on a single 16 GB GPU in float32 (bfloat16 for `gemma-4-E2B`; 0.2.2 upcasts internally). We force `eigen_solver="dense"`: the default sparse (ARPACK) solver returns only `num_eigenvalues` eigenvectors, but `spectral_entropy`/`hfer`/`energy` are computed from the *full* eigenbasis, so on the `full` scope the sparse path silently truncates and corrupts them.

---

## What it measures

For every `(model, task, condition, cell, layer)` we read the GSP diagnostics off the attention-derived graph Laplacian and the hidden-state signal on it. Metrics are **tagged by role** — claims are read only from the primary (scale-invariant) set:

| metric | role | intuition |
|---|---|---|
| `smoothness_index` | **primary** | normalized signal smoothness on the attention graph |
| `spectral_entropy` | **primary** | spread of the signal across graph frequencies |
| `hfer` | **primary** | high-frequency energy ratio |
| `gini_sparsity` | **primary** | inequality of the spectral-mass distribution (Gini of `spectral_masses`) |
| `energy` | *descriptive only* | Dirichlet energy — **length/magnitude-sensitive, never a headline** |
| `fiedler_value` | *descriptive only* | algebraic connectivity λ₂ — length-sensitive |
| `spectral_radius`, `max_imaginary` | *directed* | attention-flow **asymmetry** on the directed Laplacian |
| `lambda2`, `lambda3` | *support* | λ₂,λ₃; the gap λ₃−λ₂ gates the Davis-Kahan analysis |

## The four instruments

Each addresses the central methodological risk: that a "signal" is really just *the eval frame is longer and more formal*, not *the model knows it's being tested*.

### 1. Task-subgraph isolation
The graph is built on **only the byte-identical task tokens** (`subgraph_indices` = the task token span, located per prompt and asserted identical across frames). This fixes the node set across conditions. Under the default `rw` normalization the Laplacian is `I − D⁻¹W` with the *subgraph* degree, so it is **intrinsically row-normalized** — the softmax "mass leak" to the surrounding frame tokens is divided out by construction. This is why the primary cell is `(task_subgraph, rw)`.

**Honest caveat.** Fixing the node set fixes node *count*, which is necessary but not sufficient. To *show* what the `rw` cell controls for, we also compute `(task_subgraph, none)` (combinatorial `L = D − W`), where the length/mass artifact is deliberately visible. The `(full, rw)` cell keeps the whole prompt so length remains an observable. Every row carries `graph_scope` and `normalization`; nothing is mixed silently.

### 2. Spectral velocity
The cross-layer discrete derivative of each metric — *how the representation reorganizes with depth*. Tested **functionally** with a cluster-based permutation test (Maris–Oostenveld, sign-flip null) over the layer axis, comparing the eval and deploy velocity *curves* rather than collapsing to a fragile arg-max index. The peak-velocity layer is reported only as a **bootstrap-stability** descriptor.

### 3. Directed topology
`spectral_radius` and `max_imaginary` on the directed Laplacian capture attention-flow **asymmetry** that symmetric metrics discard — a candidate "self-monitoring" signature (routing toward instruction/frame tokens), and on the subgraph, asymmetry *within* the task tokens.

### 4. Separability (the headline)
- **Leave-task-out linear probe** on the primary metrics (`task_subgraph/rw`): L2-regularized logistic regression, C tuned by grouped inner CV, whole tasks held out. To keep the feature count sane for dozens of prompts, each metric's depth profile is collapsed to 6 **trajectory (MMT) statistics** (delta, range, flux, max_jump, slope, auc) → ~24 features rather than layers×metrics (~100). Reported with prompt/feature counts explicit, a **bootstrap CI over tasks**, and a **permutation null on AUC**. A cheap per-layer probe localizes the effect.
- **Transfer test** — train on the *subtle* cue (`eval_subtle` vs `deploy`), test on the *explicit* frame (`eval` vs `deploy`), and vice-versa. Above-chance transfer is the key evidence that the geometry encodes "being evaluated" rather than one frame's surface form.
- **Davis-Kahan Fiedler rotation** — the principal angle `1−|cos θ|` between the eval and deploy Fiedler vectors on the *identical* task subgraph, per layer. Every layer is **gated on the eigengap** λ₃−λ₂ (a small gap means the Fiedler vector is unidentified and the angle is noise — that is the actual content of the Davis-Kahan bound), with a mismatched-pairing permutation null.

## Frames

[`data/framings.json`](data/framings.json) — `deploy`, `eval`, and `placebo` are written to **matched token length** (verify with `python scripts/audit_lengths.py`).

- `control` — minimal "helpful assistant" (unmatched length; secondary baseline).
- `deploy` — live private conversation, nobody grading.
- `eval` — formal capability & safety evaluation, scored against a rubric.
- `placebo` — long-but-non-evaluative preamble.
- `eval_subtle` — a light "Question 7 of 20…" benchmark cue (used for the transfer test).

**Primary contrasts:** `eval` vs `deploy` (length-matched) and **`eval` vs `placebo`** (isolates the *evaluativeness* from the mere presence of a long formal preamble).

## Usage

```bash
# Fast end-to-end smoke test (1 model, 6 tasks, all cells + separability):
python scripts/run_pilot.py

# Full prompt-only sweep, primary eval-vs-deploy analysis:
python -m eval_awareness_spectral.cli run --analyze

# The evaluativeness-isolating contrast:
python -m eval_awareness_spectral.cli run --analyze --cond-b placebo

# One model, faster cell set, no directed metrics:
python -m eval_awareness_spectral.cli run --model gemma-2-2b-it --cells fast --no-directed --analyze

# Re-analyze an existing long CSV (with its Fiedler pickle for Davis-Kahan):
python -m eval_awareness_spectral.cli analyze \
    --long results/diagnostics_long_all.csv --fiedler results/fiedler_vectors_all.pkl \
    --cond-a eval --cond-b placebo
```

### Outputs
- `results/<model>/diagnostics_long.csv` — one row per `(model, task, condition, run, cell, layer)`; `graph_scope`, `normalization`, `st_version` columns.
- `results/<model>/fiedler_vectors.pkl` — Fiedler eigenvectors + λ₂,λ₃ per subgraph cell (Davis-Kahan input).
- `results/layer_effects_<a>_vs_<b>.csv`, `model_summary_<a>_vs_<b>.csv` — per-cell paired effects (Cohen's *d_z*, paired-t/Wilcoxon, BH-FDR).
- `results/velocity_clusters_<a>_vs_<b>.csv`, `peak_velocity_stability.csv` — velocity cluster test + peak-layer bootstrap.
- `results/probe_<a>_vs_<b>.csv`, `probe_per_layer_<a>_vs_<b>.csv`, `transfer.csv`, `davis_kahan_<a>_vs_<b>.csv` — separability suite.
- `figures/` — condition curves, effect maps (faceted by cell), velocity curves, Davis-Kahan rotation, probe AUC bars.

## Statistics & epistemics

- Each base task is a **paired** observation. Per-layer paired *t* / Wilcoxon, effect size **Cohen's *d_z*** = mean(Δ)/sd(Δ), BH-FDR across `(metric, layer)` within a cell. The **headline is the leave-task-out probe AUC**, not a count of significant layers.
- **Stated predictions (not pre-registration).** These were fixed *after* seeing the 6-task v1 pilot but *before* the clean 0.2.2 re-run:
  - **H1** — a scale-invariant eval−deploy shift survives on `(task_subgraph, rw)`.
  - **H2** — `eval − placebo` isolates evaluativeness from formal-long framing.
  - **H3** — a probe trained on the *subtle* cue transfers above chance to the *explicit* frame.
  - **H4** — the eval and deploy velocity curves differ in a cluster-significant layer band.

## Safety & dual-use

Purely **observational / mechanistic**: small open-weight models from the local cache, no deception elicitation, no training against evaluations. The aim is **transparency** — measuring *whether* a model's internals discriminate "eval" from "deploy," which is interpretability/oversight work. Results are reported as "the frame leaves a geometric fingerprint," never as a deployable eval-evasion detector, and frame-discrimination is **not** evidence of deception on its own.

## Layout

```
data/            tasks.jsonl, framings.json          (experimental materials)
configs/         models.json                         (model list + dtypes)
src/eval_awareness_spectral/
  prompts.py       frame → chat-templated prompt builder (+ task text for spans)
  spans.py         locate the byte-identical task token span; assert cross-frame identity
  runner.py        drive spectral_trust.analyze_text across cells; capture Fiedler vectors
  generation.py    optional behavioral generation stage
  analysis.py      paired deltas, effect sizes, FDR, velocity + cluster-permutation test
  separability.py  leave-task-out probe, transfer test, Davis-Kahan rotation
  plots.py         curves, effect maps, velocity, Davis-Kahan, probe bars
  cli.py           run / analyze entrypoints
scripts/run_pilot.py, scripts/audit_lengths.py
```

## Results

### Llama-3.2-3B-Instruct — full 15-task run, eval vs deploy

**Headline: a model's internal representation of byte-identical content is decodably different when it believes it is being evaluated.** A leave-task-out linear probe on the *task-subgraph* internals (fixed node set — the confound is structurally controlled) separates eval from deploy at **AUC = 0.85** (95% task-bootstrap CI [0.75, 0.98]), **permutation *p* = 0.002** (null mean 0.49, properly calibrated). This is on 30 prompts / 15 paired tasks with only 24 features.

**The length confound is most of the naive effect — but not all of it.** On the whole prompt (`full/rw`), `smoothness_index` shifts by mean |*d_z*| ≈ **8.2** (96% of layers significant). Isolating the byte-identical task tokens (`task_subgraph/rw`) shrinks it to |*d_z*| ≈ **1.65** (71% of layers) — so ~80% of the apparent effect was frame bulk/length, yet a robust, layer-broad shift survives on the identical tokens (`spectral_entropy` 1.39, `gini_sparsity` 1.20).

**Where the signal lives:** mid-to-late layers. The per-layer probe peaks at **layers 19–23 (AUC ≈ 0.87)**, and the eval−deploy *velocity* (cross-layer reorganization) differs cluster-significantly for `spectral_entropy`, `smoothness_index`, `gini_sparsity`, `energy` (all *p* = 0.001), concentrated in layers ~12–19. It does **not** for `fiedler_value` (*p* = 0.43) or `hfer` (*p* = 0.26).

**Where it does *not* live — an honest null:** the Davis-Kahan analysis shows the frame barely rotates the Fiedler eigenvector. Matched (same task, eval vs deploy) rotation `1−|cos θ|` ≈ 0.01–0.05 across the 21/28 gap-*interpretable* layers, versus a cross-task baseline ≈ 0.4 (matched ≪ mismatched, *p* = 0.001). So the eval signal is carried by the **spectral energy distribution** (smoothness/entropy/gini and their depth profile), **not** by the orientation of the graph's principal axis, which is task-determined and frame-invariant.

**No shared "being-evaluated" axis across framings (caveat: N=15).** A probe trained on the *subtle* cue (`eval_subtle` vs `deploy`) does **not** transfer to the *explicit* eval frame — transfer AUC 0.21 / 0.32, *below* chance. The explicit-evaluation fingerprint appears specific to that framing rather than a general eval detector; if anything the two eval framings are represented in opposite directions. Directional only at this sample size.

**Confound audit (task_subgraph/rw probe, LOTO AUC, permutation *p*):**

| contrast | AUC | *p* | what it controls |
|---|---|---|---|
| eval vs deploy | **0.85** | 0.002 | headline — length- & position-matched |
| eval vs placebo | **0.81** | 0.002 | evaluativeness vs long-formal framing |
| **deploy vs placebo** | **0.59** | **0.18** | **negative control** — both non-eval, length-matched → at chance ✓ |
| control vs deploy | 0.73 | 0.018 | max length/position gap → *weaker* than eval signal |
| eval vs eval_subtle | 0.88 | 0.001 | the two eval framings are themselves distinct |

- **Node set is truly fixed:** the task subgraph has 18.5 ± 5.7 nodes in *every* condition (byte-identical tokens).
- **RoPE position is controlled for the headline:** the identical task span starts at token **125 (eval) vs 127 (deploy)** — a 2-token gap — so eval-vs-deploy AUC 0.85 is not a position artifact. (eval-vs-placebo has a 27-token gap; but deploy-vs-placebo differ by 25 tokens and are still at chance, so a gap of that size does not by itself make conditions decodable.)
- **It's evaluativeness, not wording/length:** deploy vs placebo (both non-evaluative, length-matched) is at chance (0.59, *p*=0.18), and the largest length/position gap (control vs deploy) yields a *weaker* signal (0.73) than the length-matched eval contrast (0.85).

> Read responsibly: this shows Llama-3.2-3B *discriminates* eval-framing from deploy-framing in the internal representation of identical content, and that the discrimination survives length, position, and wording controls. It is **not** evidence of deception, sandbagging, or intent.

Cached models available for the full sweep: `gemma-3-1b-it`, `gemma-2-2b-it`, `qwen2.5-3b-it`, `llama-3.2-3b-it`, `gemma-4-E2B-it`.

### Confound hardening — Clever Hans hunt (Llama-3.2-3B)

The eval-vs-deploy AUC above is *partly* a shortcut. Two confounds, found and addressed:

1. **Absolute token position (RoPE).** Each frame has a fixed-length preamble, so the (byte-identical) task tokens sit at a condition-specific absolute position — and a 1-feature probe on task-start index gets AUC **1.0**. A raw-activation probe reads it too. Fix: **position-matched frames** (`--position-match`) pad each preamble with neutral function-word filler so the task starts at the *same* token index in every frame (verified spread = 0). After this, `position_only` → 0.5.
2. **Filler amount.** Padding shorter preambles more re-introduces a "how much filler precedes me" correlate. High-capacity activation probes (`act_full`, `act_pca`) **still hit 1.0 on the negative control** — the Clever Hans switches shortcuts rather than dying.

**Consequence for the design:** raw AUC is not a valid headline. The confound-robust metric is **selectivity** `S = AUC(eval-vs-deploy) − AUC(deploy-vs-placebo)` (the negative control subtracts shared wording/filler/position). Spectral features are the *most selective* because their invariances don't soak up those nuisances.

### Per-metric information & complementarity (`metric_study.py`, position-matched, 15 tasks)

Decomposing by `(metric × form)` where form ∈ {static, velocity, acceleration}:

- **Only `smoothness_index` (static) is genuinely selective:** eval-vs-deploy 0.71, negative-control 0.60, **S = 0.10, permutation p = 0.02**.
- **`spectral_entropy` and `gini_sparsity` have high raw AUC (0.72–0.73) but *equally high* negative-control** → S ≈ 0 → they mostly read the confound, not evaluativeness. Lumping them into one probe *diluted* the signal (this is why the combined probe looked weak).
- **Velocity/acceleration forms were worse** (negative S) on this contrast — the depth-derivatives amplify the (cross-layer-structured) filler confound. Static wins here.
- **They complement:** `smoothness + hfer` → **S = 0.16**; greedy forward-selection by S → `smoothness[static] + hfer[accel] + hfer[static]` → **S = 0.17**, negative-control 0.55 (lowest leakage). Curating nearly doubles selectivity vs the best single metric.
- **Signal localizes late:** `smoothness_index` peaks at **layer 21** (single-value AUC 0.84).

**Honest bottom line:** the evaluativeness signal *survives* position-matching in `smoothness_index (+ hfer)`, significantly (p=0.02) — it is **not** pure Clever Hans — but once position + wording + filler are stripped it is **modest** (S ≈ 0.17 at N=15) and carried by one metric, not the whole suite.

### Causal steering (`causal.py`)

Difference-of-means eval-direction (optionally rank-k SVD-denoised, à la spectral-steering), **norm-relative** dose-response (α as a fraction of the residual norm), a **fluency guard** (`degenerate`, dropped from the means), a **narrow layer band** around the late-layer signal peak (steering many layers at once compounds and destroys fluency), and **placebo / random direction** specificity controls, on held-out tasks.

Calibration mattered: raw large-α and wide-band steering *uniformly broke* generation. At layers [18,19,20], α∈{0,.05,.1,.2}, generation stays fluent (108/128) — but the **result is a null**: across every α, all three directions (eval / placebo / random) fluctuate around baseline (hedging ≈0.38, refusal ≈0.12, meta-eval ≈0) with no monotonic dose-response and **no eval-specific effect vs the controls**. Interpretation: the harness is sound, but at this scale (8 held-out tasks) with keyword-based behavioral proxies there is no demonstrated causal effect. Getting a real signal needs a sensitive behavioral metric (sandbagging accuracy on gold-labeled items, or an LLM judge) and more tasks — this is the clearest remaining gap.

### Baselines from the literature (`sandbagging.py`)

To match/beat the eval-awareness field we include its standard baselines on the same
benchmarks (GSM8K / ARC / MMLU, frame-wrapped):

- **Verbalized / black-box** — ask the model directly "eval or deployment?". On Llama-3.2-3B
  this scores **AUC 1.00** on the explicit frames: the model can *verbally* classify them
  perfectly. **This sharpens the contribution**: detecting *explicit* eval-awareness is trivial
  (black-box already saturates), so the white-box spectral probe's value is **not** raw detection
  — it is (a) **selectivity** (isolating evaluativeness from format, which black-box cannot),
  (b) the **subtle/implicit-cue** regime the model can't verbalize, and (c) **interpretability**.
- **Behavioral sandbagging** — accuracy on the *same* gold-labeled items under eval vs deploy
  framing. `sandbagging = acc(deploy) − acc(eval)`; a signed, gold-grounded behavioral effect
  (and the target for the causal steering test). Consistent with the field, we expect this to be
  small at scale.
- **Linear activation probe** (`baselines.py`) — the standard white-box baseline; confound-tested
  via the selectivity metric.

**Scaling:** `configs/models_scaling.json` gives within-family size ladders (Qwen2.5 0.5→3→7B,
Llama-3.2 1→3→8B, Gemma 1→2B) sized for a 16 GB GPU, to test the power-law (larger ⇒ stronger
detection) the field reports.

_(gemma-2-2b-it 6-task pilot was a plumbing smoke test and is superseded by the runs above.)_

## License

The `spectral-trust` dependency is AGPL-3.0; this repo follows suit.
