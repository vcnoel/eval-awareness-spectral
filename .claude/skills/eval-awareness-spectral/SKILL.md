---
name: eval-awareness-spectral
description: Run and reason about the eval-awareness spectral experiments in this repo — detecting whether an LLM's internals shift under an evaluation frame vs deployment, on spectral-trust 0.2.2. Use when running the pipeline (spectral sweep, activation baselines, per-metric study, causal steering), interpreting results, or avoiding the known confounds (position/RoPE, filler-amount) and pitfalls.
---

# eval-awareness-spectral

Instrument: byte-identical tasks wrapped in situational frames (eval / deploy / placebo /
eval_subtle / control), pushed through `spectral-trust` GSP diagnostics on the **task-token
subgraph**. Question: does the model represent the *same content* differently when it thinks
it is being evaluated? Env: conda `gemma_spectral`. Always `PYTHONPATH=src`.

## The one thing to internalize: raw AUC is NOT the headline — selectivity is

Every "eval is decodable" number is partly a shortcut (Clever Hans). Report
**selectivity** `S = AUC(eval-vs-deploy) − AUC(deploy-vs-placebo)`. The negative control
(deploy-vs-placebo, both non-evaluative) subtracts shared wording/filler/position. A family
with high eval-AUC but equally high negative-control AUC is reading a confound, not
evaluativeness.

## Known confounds (in order discovered) — do not regress

1. **Absolute position / RoPE.** Fixed-length preambles put the task at a condition-specific
   token index; a 1-feature position probe scores AUC 1.0. Fix: `--position-match` (pad
   preambles with neutral function-word filler so task-start spread = 0). Verify with the
   `position_only` baseline → must be ~0.5.
2. **Filler amount.** Position-matching pads shorter preambles more → high-capacity activation
   probes still hit 1.0 on the negative control. This is why activations are **not** a clean
   baseline; use selectivity, and rely on the length-matched **eval-vs-deploy** contrast
   (both get ~equal filler) as the cleanest.
3. **Only `smoothness_index` is genuinely selective** (S≈0.10, p≈0.02); entropy/gini are mostly
   confound; velocity/accel forms hurt selectivity. Curate `smoothness + hfer` (S≈0.16–0.17).

## Library gotchas (spectral-trust 0.2.2)

- Force `eigen_solver="dense"` — the sparse solver truncates the eigenbasis and corrupts
  entropy/hfer/energy on graphs >50 nodes.
- 0.2.2 dropped gini; we recompute `gini_sparsity` from the returned `spectral_masses`.
- Directed metrics are computed on the FULL adjacency even under `subgraph_indices` — record
  them only on the `full` scope.
- Under `--position-match`, the `full`-scope graph is filler-dominated and its degenerate
  Laplacian STALLS the eigensolver → use the `task_subgraph` cell only (the driver does this).

## Pitfalls that already bit us

- **Windows cp1252 console** crashes on unicode (−, ×, ↔). Keep prints ASCII / reconfigure stdout.
- **Task-span reconcile:** position-matching perturbs the task's boundary token in the bare
  `control` template → keep the majority tokenization, drop only outlier frames (`reconcile_task_spans`).
- **Causal steering magnitude:** a raw diff-of-means α breaks generation uniformly. Steer
  **norm-relative** (`α · ‖h‖ · unit_dir`), dose-response α∈{0,.25,.5,1,2}, fluency guard, and
  placebo+random direction controls. Optionally rank-k SVD-denoise the direction.

## Commands

```bash
conda activate gemma_spectral; export PYTHONPATH=src   # (PowerShell: $env:PYTHONPATH="src")

# Core spectral sweep + analysis (eval vs deploy):
python -m eval_awareness_spectral.cli run --model llama-3.2-3b-it --analyze
python -m eval_awareness_spectral.cli run --analyze --cond-b placebo   # evaluativeness contrast

# Full experiment driver (spectral + activation baselines + per-metric + causal):
python scripts/run_experiments.py --tasks 15 --position-match --fresh   # confound-hardened
python scripts/run_experiments.py --tasks 15 --position-match --skip-battery  # skip slow act_full perms

# Per-metric information & complementarity study:
python -m eval_awareness_spectral.metric_study results/<run>/<model> 500
```

`--skip-battery` avoids the slow `act_full` (89k-dim) permutation nulls, which are the usual
run-time sink. Results/figures/pkls are gitignored.

## Modules

`prompts.py` frames + position-matching · `spans.py` task-span location + reconcile ·
`runner.py` spectral sweep (cells = scope×normalization) · `activations.py` residual pooling +
moments · `baselines.py` feature-family ladder + selectivity · `metric_study.py` per-metric ×
form decomposition · `separability.py` probe / transfer / Davis-Kahan · `causal.py` norm-relative
steering · `plots.py` · `cli.py` · `scripts/run_experiments.py` driver. Design rationale in
`docs/experiment_design.md`.
