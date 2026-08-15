# A Probe Direction Is a Property of Its Prompt

Analysis code, data manifests and results for the paper *A Probe Direction Is a Property of
Its Prompt* ([arXiv:2608.13329](https://arxiv.org/abs/2608.13329)).

The manuscript itself is not in this repository. arXiv publishes the LaTeX source of every
submission, so a second copy here would only be something that can disagree with it. What is
here is the part arXiv does not carry: the code that produced the numbers, and the numbers.

Linear probes for "the model believes it is being evaluated" are built by contrasting activations on
prompts that announce an evaluation against prompts that do not. The resulting AUROC is reported per
model, compared across models, and correlated with scale. **We measure what that scalar is actually a
property of, and the answer is mostly the contrastive prompt the experimenter happened to write.**

Building 36 directions from a 6 × 6 crossing of evaluation against deployment framings over
byte-identical task text:

| | |
|---|---|
| Correlation of probe score with log₁₀ parameters | **−0.988 to +0.939** on one model family |
| Same range on every ladder measured | three families, two vendors |
| Cosine between directions for *one* concept, different wrappers | **0.3749**, against **0.9995** for two halves of one wrapper |
| Variance attributable to the model | **0.8%**, against **38.3%** for the wrapper × model interaction |
| Generalizability coefficient of a single-wrapper design | **Eρ² = 0.012 to 0.263**, against a conventional 0.80 |
| Wrappers needed for a defensible between-model comparison | **≈ 9**, where practice uses 1 |

Both published results that disagree in sign about how evaluation awareness scales are reproducible
from this one design, by wrapper choice alone.

The construction is not specific to this concept: the same contrastive difference-of-means, with the
same unfixed prompt choice, is how directions are built for refusal, truthfulness and steering. What
follows is about the estimator; evaluation awareness is where it carries a scaling claim.

## What is here

```
scripts/        the analysis pipeline, one script per result
results/        every result as JSON, tracked
src/            eval_awareness_spectral, the library the pipeline imports
configs/        model ladders pinned by revision, protocol schema, claim ledger
data/           framings, task pools, and manifests (no benchmark text; see below)
```

**No number in the paper is typed by hand.** Every printed quantity is a LaTeX macro resolved from a
file in `results/` at build time, so a number can change only by a run changing, and a missing or
stale input fails the build rather than silently printing a previous value. The generator ships with
the arXiv source rather than here.

## Reproducing the results

The tracked `results/*.json` are the paper's numbers. Regenerating them needs artifacts we do not
redistribute, because they belong to their original authors:

```bash
pip install -e .
export EAS_REANALYSIS_ROOT=/path/to/released-artifacts   # Manek 2026; Arditi et al. 2024
export EAS_SAD_ARCHIVE=/path/to/sad_archive.zip          # the SAD benchmark
export EAS_SAD_PASSWORD=...                              # from the SAD project, not shipped here
make all                                                 # or run scripts/<name>.py individually
python scripts/backfill_conventions.py --write
```

Expected layout under `EAS_REANALYSIS_ROOT`:

```
manek/notebooks/...                        released notebooks
manek/datasets/contrastive_dataset.json    released contrastive set
arditi/dataset/splits/                     refusal splits
```

Running a script with the variable unset prints what to fetch rather than a path from somebody
else's laptop.

> **The `backfill_conventions.py` step is required, not optional.** Most analysis scripts still
> write their JSON directly rather than through `scripts/result_io.py`, so re-running one strips the
> conventions block and the next consumer will refuse it until it is restamped. This is a known
> wart, recorded here rather than papered over; `result_io.write_result` is the intended path.

## What the checks guard

Each was added after an error it would have caught.

| check | refuses |
|---|---|
| `result_io.py` | a results file written without a complete conventions record |
| `backfill_conventions.py` | a results file whose producer changed since it was written |

Every check that reports success is fault-injected before it is trusted. One was written with an
exemption that was a substring of the violation it guarded, so it passed while whitelisting exactly
what it existed to catch; injection found that, reading it did not.

## Models

Ten instruction-tuned checkpoints carry the primary results: `Qwen2.5` at 0.5B/1.5B/3B/7B,
`gemma-2-2b-it`, `Llama-3.2` at 1B/3B, and `SmolLM2` at 135M/360M/1.7B. Two further ladders were
extracted on a rented A100 — `Qwen3` at six scales under both chat-template renderings, and `OLMo-2`
at four, a second vendor. Every checkpoint is pinned by revision in `configs/ladders.json`.

## Data we do not redistribute

SAD benchmark question text is licence-restricted and is never committed here; `data/rendered/`
carries only a manifest of hashes, counts and template ids, so a prompt set can be verified from the
repo without the question bodies. The reanalysed studies' released files stay with their authors.
Activation caches (`results/**/*.npz`) are large and regenerable and are not tracked; the result
JSONs are neither, and are.

## The protocol package

`src/eval_awareness_spectral` is more than a helper library for the paper. It carries the causal
protocol this project was built around, which keeps three things separate that "evaluation
awareness" routinely collapses:

- **Environment cue (E)** — an externally assigned prompt or runtime factor.
- **Recognition (R)** — a model-internal or independently labelled judgment about the environment.
  It must be measured with a dedicated endpoint; it is not implied by E.
- **Behavioural propensity (P)** — a distribution over task behaviour under a defined scoring
  endpoint. It requires generation or another outcome-bearing intervention; it is not an activation
  label.

The paper measures a probe statistic, which is an **R**-side instrument, and its result is that the
instrument is mostly a property of **E**. The protocol tooling validates configurations, prompt and
result manifests, and claim ledgers, and runs offline:

```bash
eval-awareness-spectral validate-config configs/protocol.example.json
eval-awareness-spectral claim-audit --ledger <ledger.json> README.md
```

Design notes, the pre-registered analysis rules and the future-experiment protocol ship with the
arXiv source rather than in this repository.

## Citation and licence

Please cite [arXiv:2608.13329](https://arxiv.org/abs/2608.13329). Licensed under AGPL-3.0
(`LICENSE`).

The two reanalysed studies released per-layer score files, direction vectors and contrastive sets,
which most work of this kind does not. Every quantity we recompute exists because they did, and that
asymmetry should count in their favour rather than against them.
