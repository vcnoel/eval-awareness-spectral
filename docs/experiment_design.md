# Experimental design: eval-awareness spectral fingerprint (ICLR-track target)

## Positioning against the eval-awareness literature

The field has converged on findings we can build on and, in places, beat:

- **Eval-vs-deploy is linearly decodable from residual-stream activations (~0.83 AUC)** and
  detection **scales with model size** (power law) — SAD (Laine et al.), *Evaluation Awareness
  Scales Predictably* (arXiv 2509.13333). → our **activation-probe baseline** + **scaling ladder**
  (`configs/models_scaling.json`) replicate this.
- **Probe evidence may be *format / structure sensitivity*, not genuine awareness** — *Is
  Evaluation Awareness Just Format Sensitivity?* (arXiv 2603.19426), which uses a 2×2 controlled
  design over MMLU/ARC/GSM8K. → **our core contribution goes further**: full **position-matching**
  (task at identical token index) + a **selectivity metric** with a negative control, and we show
  activation probes fail the negative control (format-confounded) while the **spectral** probe is
  more selective and interpretable.
- **Eval-awareness has limited behavioral effect** (arXiv 2605.05835) and **is not one capability**
  (arXiv 2606.23583). → we **replicate** both (causal behavioral null with proper controls; subtle→
  explicit non-transfer) — replication of the field's own findings strengthens credibility.

**Standard baselines we include** (to match/beat the field): (1) **verbalized / black-box** — ask
the model "eval or deployment?" (SAD *Stages Oversight*; `sandbagging.run_verbalized_baseline`);
(2) **linear activation probe** per layer (`baselines.py`); (3) **behavioral sandbagging** on
gold-labeled GSM8K/ARC/MMLU (`sandbagging.run_sandbagging`). Datasets = the same benchmarks the
field uses, frame-wrapped byte-identically.

**One-line thesis:** *eval-awareness is decodable but recent work shows probe evidence may be format
sensitivity; we give a confound-robust, interpretable spectral-geometric probe that isolates genuine
evaluativeness (position-matched + selectivity), quantify which spectral quantities carry it, and
replicate the field's behavioral-null and no-transfer results.*

---


Adversarial design. Every experiment below is written to survive the specific
reviewer objection stated next to it. The guiding strategic decision:

> **Do not run a raw-AUC race against an activations probe — you will lose it.**
> Eval-awareness is near-linearly decodable from the residual stream, so a linear
> probe on task-token activations will sit near ceiling. Concede that axis openly
> and win on the four axes where spectral features are actually superior:
> **(A) selectivity, (B) dimensionality, (C) cross-architecture transfer,
> (D) interpretability + causal grounding.**

Everything is on **Llama-3.2-3B-Instruct** first (light, same model), designed so
the only remaining step is scaling to more models/tasks.

---

## 0. Shared protocol (applied identically to every feature family)

To make all comparisons apples-to-apples and pre-empt "unfair comparison":

- **Unit of analysis:** the byte-identical **task-token span** only (same node set /
  same token set the spectral probe uses). Never the frame tokens.
- **Split:** leave-one-task-out (whole task held out — content never leaks).
- **Significance:** permutation null (shuffle condition label *within task*, ≥1000
  perms) → empirical *p*. **This is the arbiter, not the point AUC.**
- **Uncertainty:** task-bootstrap 95% CI.
- **Classifier held fixed** across families: L2-logistic regression, C by grouped
  inner CV, `StandardScaler`. Same estimator for spectral and activations, so any
  difference is the *features*, not the model.
- **Every family is run on the full contrast battery** (below), not just the headline.

### The contrast battery (the confound controls, run for *every* feature family)
| contrast | role | pass criterion |
|---|---|---|
| eval vs deploy | headline | AUC ≫ 0.5, *p*<0.05 |
| eval vs placebo | evaluativeness vs long-formal | AUC ≫ 0.5 |
| **deploy vs placebo** | **negative control** (both non-eval, len-matched) | **AUC ≈ 0.5 (want NULL)** |
| control vs deploy | length/position sensitivity | reference |
| eval vs eval_subtle | framing specificity | reference |

**Selectivity index** `S = AUC(eval-vs-deploy) − AUC(deploy-vs-placebo)`. A feature
family that scores high eval-vs-deploy but *fails* the negative control (also
decodes deploy-vs-placebo) is measuring wording/length, not evaluativeness. **S is
the headline metric of the whole paper**, reported per family. This is where
spectral can beat activations even at lower raw AUC.

---

## 1. Activations probe baseline (light, runnable now, confound-free)

**Reviewer objection pre-empted:** *"Your 24 spectral features beat nothing —
you never compared to the obvious activation probe, at matched capacity, with the
same controls."*

### Feature extraction (one extra forward pass per prompt, no eigendecomposition)
For each prompt, one forward pass; at each layer `ℓ`, take the residual stream over
the **task-token span** and **mean-pool** → `a_ℓ ∈ R^{d}` (d=3072, 28 layers).
Save `A ∈ R^{28×3072}` per prompt. (~75 prompts; trivial to cache.)

### The four feature families, all at controlled capacity
1. **Spectral (ours)** — 24 features (4 scale-invariant metrics × 6 trajectory stats).
2. **Activation-moments (THE fair low-dim control)** — per layer, a *graph-free*
   low-dim summary of the same task-token activations: `‖mean‖₂`, total variance,
   and participation ratio (effective rank) → 3 scalars/layer, trajectory-summarized
   to ≈24 features. **This is the decisive control**: it isolates "does the GSP
   *graph* structure add anything" from "does any cheap summary of activations work."
   If spectral ≈ activation-moments, the graph is decoration; if spectral > moments
   at matched dim (especially on selectivity S), the graph earns its place.
3. **Activations-PCA** — mean-pooled activations reduced by PCA to k ∈ {8,16,24,48}
   (PCA fit on training fold only — no leakage). Capacity-matched sweep.
4. **Activations-full (upper bound)** — all 28×3072 dims, strong L2. Establishes the
   ceiling; expected to win raw AUC. We *report this and concede it.*

### Deliverables / plots
- **AUC vs feature budget** for all families (capacity-controlled curve).
- **Selectivity S per family** (bar chart). The claim to establish: spectral's S ≥
  activations', i.e. spectral is *more selective* for evaluativeness.
- **Data-efficiency curve:** AUC vs #training tasks (2…N). Claim: spectral needs
  fewer examples (low-dim ⇒ sample-efficient), relevant because eval sets are small.
- **Layer localization:** per-layer AUC for spectral vs activation-moments (do they
  peak in the same layers? mechanistic coherence).

### What a clean win looks like (and the honest fallback)
- **Best case:** activations-full wins raw AUC, but spectral ≥ activation-moments AND
  spectral has the highest selectivity S and best data-efficiency. → "GSP is a
  compact, selective, interpretable lens on eval-awareness."
- **Fallback (must be stated if it happens):** if activation-moments ≈ spectral on
  everything, the paper's contribution is the *phenomenon + confound methodology +
  causal result*, not GSP specifically. Decide framing from the numbers, not before.

---

## 2. Full baseline ladder (stronger baselines — completes the design)

Ordered floor → ceiling, each answering a specific "but isn't it just…?":

| # | baseline | kills the objection | expected |
|---|---|---|---|
| 0 | majority / random | floor | 0.5 |
| 1 | **n_tokens only** (1 feat) | "it's just length" | on subgraph: ≈0.5 (const) |
| 2 | **task-position only** (start index) | "it's just RoPE position" | ≈0.5 for eval-vs-deploy (matched) |
| 3 | **lexical BoW on frame text** | "internals are irrelevant, it's in the text" | ~1.0 — establishes the trivial text upper bound & motivates probing internals |
| 4 | **attention-graph only** (no hidden states: attn entropy/gini) | "it's the hidden signal, not the graph" | isolates attention contribution |
| 5 | **activation-moments** (§1.2) | "any activation summary works" | the key control |
| 6 | **activations-PCA / full** (§1.3–4) | "just probe activations" | ceiling |
| 7 | **spectral (ours)** | — | — |

Plus the **unique-advantage experiment** (needs a 2nd model, designed now):

- **Cross-architecture probe transfer.** Train the probe on Llama features, test on
  Qwen2.5-3B features. **Spectral features are model-dimension-agnostic** (same 24
  graph statistics regardless of hidden dim / head count), so a spectral probe *can*
  transfer across architectures; an activation probe **cannot** (3072-dim Llama vs
  different-dim Qwen — undefined). This is a genuine, un-rebuttable spectral win even
  if within-model AUC is lower. Report transfer AUC spectral (finite) vs activations
  (not applicable). This alone can carry the "why spectral" argument.

### Benchmarks / datasets (the scaling path, designed to be clean)
- **Frame-wrapping over a standard benchmark.** Replace the 15 hand-written tasks
  with N items drawn from established sets (MMLU, GSM8K, TruthfulQA, a safety set),
  each wrapped byte-identically in the 5 frames. Gives hundreds of paired tasks,
  standardized, and lets us analyze by task type. Removes "cherry-picked prompts."
- **Frame-paraphrase generalization (kills "you memorized the wording").** ≥5
  paraphrases per condition (5 eval wordings, 5 deploy wordings, …). Train the probe
  on a subset of wordings, test on **held-out wordings**. Above-chance ⇒ the signal
  is *evaluativeness*, not one specific phrasing. This is the strongest anti-confound
  control and should be a headline result.
- **Position-matched frames (closes the residual gap).** Pad every frame's preamble
  to an identical token length so the task span starts at the **same absolute token
  index** in all conditions (eval-vs-placebo currently offsets by 27 tokens). Re-run
  the whole battery on position-matched frames; the effect must survive.

---

## 3. Causal experiment (good-faith, adversarial)

**Reviewer objection pre-empted:** *"This is correlational — internals differing
under eval says nothing about behavior or mechanism."*

Goal: show the detected eval-representation is **causally load-bearing** for behavior,
with specificity controls that rule out "steering just perturbs the model."

### Direction & intervention
- **Eval direction** `v_eval^ℓ = mean(act_eval) − mean(act_deploy)` over the
  task-token span, per layer (difference-of-means steering vector; derived on
  **train** tasks only).
- **Intervention:** at inference, add `α·v̂_eval^ℓ` to the residual stream at layer(s)
  ℓ over the generated positions. Bidirectional:
  - deploy prompt **+** `α v_eval` → predict shift **toward** eval-behavior.
  - eval prompt **−** `α v_eval` → predict shift **toward** deploy-behavior.
- **Dose–response:** sweep α; predict monotonic behavioral shift (not a single point).

### Specificity controls (the adversarial core)
| control direction | predicted effect | rules out |
|---|---|---|
| **random** vector, matched norm | no eval-specific behavioral shift | "steering = noise" |
| **placebo direction** `mean(act_placebo) − mean(act_deploy)` | no eval-specific shift | "it's a generic long-context/formality effect, not evaluativeness" |
| shuffled-label direction | no shift | leakage / artifact |

The **placebo-direction control is the crux**: it has the same "long formal preamble"
content as eval but is non-evaluative. If steering along `v_eval` shifts behavior but
`v_placebo` does not, causal specificity to *evaluativeness* is established.

### Behavioral read-outs (pre-specify before running)
On **held-out tasks and held-out frame wordings**:
- **Sandbagging:** accuracy on capability items (GSM8K/MMLU) under +eval-steer vs
  baseline — predict a *drop* if the model down-modulates effort when "tested," or a
  *rise* (trying harder). Either directional, pre-registered.
- **Caution/refusal shift:** refusal + hedging rate on dual-use/safety items.
- **Explicit meta-awareness:** rate of "as this is a test / evaluation…" style tokens.
Report effect sizes with the random/placebo controls on the same axis.

### Mediation (ties it back to the detector)
Show the intervention that changes behavior also moves the **spectral read-out** in
the predicted direction (the spectral probe's score shifts under +eval-steer). This
closes the loop: *detected spectral signal ↔ activation direction ↔ behavior*.

### Necessity variant (optional, stronger)
Activation **patching**: replace the eval-prompt task-token residual with its
deploy-prompt counterpart along `v_eval` and test whether eval-behavior reverts →
a necessity (not just sufficiency) claim.

---

## 4. What "only scaling remains" means after this

Once §1–3 are run on Llama-3.2-3B and one 2nd model:
- Confound-free detection with the full baseline ladder + selectivity S. ✓
- Frame-paraphrase & position-matched generalization. ✓
- Cross-architecture transfer (spectral's structural advantage). ✓
- Causal steering with random+placebo controls + mediation. ✓

Remaining = **breadth only**: more models (families × scales), more benchmark items,
more frame paraphrases. No new experiment type — just N. That is the definition of a
complete design.

### Residual risks to disclose (don't let a reviewer find them first)
- N tasks still modest until benchmark-wrapping lands → power caveat.
- Steering magnitude α can degrade fluency; report a fluency metric alongside.
- Difference-of-means direction is linear; if eval-representation is nonlinear, the
  causal effect underestimates — state it.
- Single language, instruction-tuned models only → scope claim accordingly.
