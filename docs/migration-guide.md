# Migration guide

The current release makes the CPU-safe protocol and provenance interface primary. Historical empirical modules remain temporarily available for compatibility, but they are deprecated and are not evidence that a historical run is valid.

## Installation

| Previous usage | Current usage |
|---|---|
| Install model and plotting dependencies with the package | `pip install -e .` for CPU-safe protocol validation |
| Assume `spectral-trust` and a model runtime are mandatory | Install `.[legacy]` only for historical execution; optional external runtimes are caller-managed |
| Depend on a named local environment | Use any Python environment satisfying `pyproject.toml`; record its lock or inventory in the run manifest |

## Command line

| Previous interface | Current interface |
|---|---|
| `python -m eval_awareness_spectral.cli run ...` | `eval-awareness-spectral legacy run ...` during migration |
| `python -m eval_awareness_spectral.cli analyze ...` | `eval-awareness-spectral legacy analyze ...` during migration |
| Direct `run` or `analyze` aliases | Still accepted but deprecated; do not use in new automation |
| Ad hoc output inspection | `validate-results RESULT_MANIFEST.json` |
| Ad hoc prompt files | `validate-prompts PROMPT_MANIFEST.jsonl` |
| Unhashed settings | `validate-config PROTOCOL.json` and retain the returned config hash |
| Prose copied from output tables | `claim-audit --ledger CLAIM_LEDGER.json FILE...` |
| Discover a result format from old CSVs | `schema` to print the packaged result-manifest schema |

Validation commands do not infer or download. New inference orchestration belongs outside the core CLI and must consume validated local manifests.

## Module mapping

| Historical module or concept | Current location |
|---|---|
| prompt construction and span bookkeeping | `eval_awareness_spectral.data` prompt records and manifest validation |
| raw residual extraction | external harvester plus `integrations.standalone.adapt_numpy_activations` |
| Goodfire-like activation rows | `integrations.goodfire.adapt_activation_container` |
| spectral calculations embedded in runner code | `representations.attention_graph` and `representations.spectral` |
| pooled activation baselines | `representations.residual` |
| probe and selectivity utilities | `evaluation.probes` and `evaluation.selectivity` |
| transfer analysis | `evaluation.transfer` |
| scaling analysis | `evaluation.scaling` |
| paired and permutation statistics | `evaluation.statistics` |
| steering, patching, mediation | `interventions.steering`, `interventions.patching`, `interventions.mediation` |
| causalab/SGLang/lm-eval calls | callback protocols in `integrations.goodfire` |
| scalar-EOS generation and ad hoc completion checks | `generation_contract` terminal normalization, validity, span, and alignment records |
| CSV/pickle output conventions | checksummed result manifests plus `provenance` utilities |
| empirical “law mining” | candidate-only workflow documented in [laws](laws.md) |

Historical modules such as `runner`, `analysis`, `baselines`, `separability`, `causal`, `law_mining`, and plotting helpers remain for compatibility. Do not build a new pipeline on them. Migrate one boundary at a time: manifests, representation functions, evaluation, then interventions.

## Changed defaults and rejected behavior

- **Operator:** basis-producing analysis now requires `symmetric-normalized`. Historical random-walk basis defaults are not reproduced.
- **Random walk:** categorical random-walk basis analysis fails instead of returning coefficients that may be treated as Euclidean orthogonal. Graph-only legacy quantities must be labeled as such.
- **Data access:** core validation uses local files. It does not download datasets, models, or tokenizers.
- **Inference:** no inference occurs from config, prompt, result, schema, or claim validation.
- **Generation validity:** preserve every terminal ID declared by the model generation configuration. Capped, repetitive, empty-final, or missing-close traces remain auditable but are ineligible for scoring.
- **Online stopping:** repetition stopping is opt-in because it changes output. Post-hoc classification can be added without changing decoding.
- **Historical outputs:** files without terminal and repetition provenance cannot be certified retroactively. Audit or regenerate them rather than assuming an ordinary completion.
- **Splits:** source-task and template-family isolation is explicit and validated; row-random splitting is insufficient.
- **Fitting:** preprocessing and selection belong inside grouped nested fitting.
- **Claims:** a numerical empirical claim requires a validated claim-ledger entry linked to a checksummed result manifest.
- **Outputs:** reserve a run directory by config hash; do not overwrite an output created by another configuration.

## Minimal migration sequence

1. Convert prompts into the local JSONL schema, retaining stable task identity, template family, factor vector, hashes, tokenizer revision, token span, and split.
2. Validate the prompt and split manifests before any inference.
3. Adapt existing NumPy activations with explicit sequence, layer, and token axes. Do not discard provenance during pooling.
4. Replace basis-dependent random-walk calculations with symmetric-normalized decomposition. Recompute affected quantities; do not relabel old values.
5. Declare graph-only, fixed-reference-signal, or joint graph-and-signal mode for each comparison.
6. Move preprocessing and model selection into grouped nested folds; rerun complete-pipeline permutations.
7. Emit checksummed metric and endpoint artifacts and a result manifest.
8. Add claim-ledger entries only after offline validation succeeds.

Legacy results cannot be promoted merely by converting their file format. If their operator, split, fitting, endpoint, or provenance does not satisfy the current protocol, mark them `reported-unverified` or rerun them. No legacy numerical results are asserted by this repository.
