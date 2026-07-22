# eval-awareness-spectral

This repository separates three variables that are often collapsed under “evaluation awareness”:

- **Environment cue (E):** an externally assigned prompt or runtime factor, such as evaluation wording, deployment wording, oversight, or time pressure.
- **Recognition (R):** a model-internal or independently labeled judgment about the environment. R must be measured with a dedicated recognition endpoint; it is not implied by E.
- **Behavioral propensity (P):** a distribution over task behavior under a defined scoring endpoint. P requires repeated generation or another outcome-bearing intervention; it is not an activation label.

The research question is whether controlled changes in E alter R and P, which representations mediate those changes, and which simpler nuisance explanations survive. The package supplies CPU-safe protocol, representation, evaluation, provenance, and release utilities for that question.

> **Pre-results status:** no validated empirical artifacts are bundled with this repository. There are no tracked model outputs, activation caches, result manifests, or validated numerical claims. The documents describe definitions, conjectures, and a protocol to run later.

## Spectral convention

Basis-dependent graph Fourier analysis defaults to the symmetric-normalized Laplacian

`L_sym = I - D^{-1/2} W D^{-1/2}`

for a symmetric, nonnegative adjacency matrix. This operator has a real Euclidean-orthonormal eigenbasis and null vector `D^{1/2}1` (for non-isolated connected nodes). That null mode is degree weighted; it is not generally the ordinary token mean.

The categorical random-walk operator `I - D^{-1}W` is rejected for basis-dependent diagnostics. It is generally nonsymmetric, so feeding its right eigenvectors to symmetric-eigensolver or Parseval-style code does not produce a valid Euclidean-orthonormal graph Fourier basis. Random-walk quantities may be used only as explicitly graph-only legacy diagnostics, never silently substituted into spectral signal accounting. The fixed combinatorial Laplacian remains a valid comparison operator where ordinary DC invariance is the intended property. See [mathematical notes](docs/mathematical-notes.md).

## Install and CPU-only validation

Python 3.11 or newer is required. The core install has no model-runtime dependency:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
eval-awareness-spectral schema
```

Validation commands are local, CPU-safe, and do not import model, tokenizer, dataset-client, or GPU stacks:

```bash
eval-awareness-spectral validate-config path/to/protocol.json
eval-awareness-spectral validate-prompts path/to/prompts.jsonl
eval-awareness-spectral validate-results path/to/results_manifest.json
eval-awareness-spectral claim-audit --ledger path/to/claim_ledger.json README.md docs/*.md
```

The core accepts local JSON/JSONL manifests and NumPy-like activation arrays. Prompt manifests preserve source-task identity, template family, factor vector, condition valence, content and rendered-text hashes, task-token hash, tokenizer revision, token span, split, optional R label, and optional P outcome. Split validation keeps source-task and template groups out of confirmation data used for final claims.

## Optional integrations

External systems are protocol-based seams, not core dependencies:

- Goodfire-style activation containers can be adapted by structural fields.
- causalab-style interchange can be supplied as a callback with explicit donor, recipient, target, and endpoint.
- SGLang-style steering can be supplied as a callback with an explicit target and endpoint.
- lm-eval or a gold scorer can be supplied through the evaluation callback.

These packages are imported lazily only when requested. A caller may instead provide a duck-typed callback or local container. Installation, authentication, model serving, and inference remain the caller’s responsibility. The repository contains no host addresses or vendor-specific compute instructions.

## Protocol and provenance

The executable design is specified in [future experiment protocol](docs/future-experiment-protocol.md); [experiment design](docs/experiment_design.md) is its short entry point. A run must bind:

1. a hashed protocol configuration;
2. a local prompt and split manifest;
3. exact model, tokenizer, dataset, and adapter revisions;
4. seeds and software/environment metadata;
5. checksummed inputs, intermediate artifacts, metrics, and endpoint outputs;
6. a result manifest and claim-ledger entry for every released empirical claim.

Run directories are reserved by configuration hash and writes are atomic. Validation rejects missing artifacts, checksum mismatches, incomplete crossed cells, split leakage, and empirical prose without a validated manifest-backed claim marker.

### No-network and no-inference release checks

Release validation must run with network access disabled and without model inference. It validates schemas, hashes, split isolation, artifact completeness, and claim provenance only. If a required prompt, split, result, endpoint, or claim manifest is absent, stop before inference and before manuscript promotion. A clean local validation is necessary but does not turn a conjecture into a result.

## Legacy compatibility

Historical modules remain importable for migration, and the old empirical interface is available only through `eval-awareness-spectral legacy ...`; direct `run` and `analyze` aliases are deprecated. Install `.[legacy]` only to inspect or migrate that path. Legacy defaults that used random-walk basis analysis are not preserved: current basis-producing runs use symmetric normalization, and categorical random-walk basis requests fail. See the [migration guide](docs/migration-guide.md).

## Documentation

- [Research intuitions](docs/research-intuitions.md): accepted design intuitions and their falsifiers.
- [Mathematical notes](docs/mathematical-notes.md): proved identities, non-theorems, and comparison modes.
- [Future experiment protocol](docs/future-experiment-protocol.md): the not-yet-run executable protocol.
- [Candidate observations and conjectures](docs/laws.md): claim-safe research targets.
- [Theory index](docs/theory.md): map from mathematical facts to empirical questions.
- [Migration guide](docs/migration-guide.md): old-to-current interfaces and defaults.

## License

This project is licensed under the GNU Affero General Public License, version 3.0. See [LICENSE](LICENSE). Network use of a modified version carries the AGPL source-availability obligations; optional integrations do not change this repository’s license.
