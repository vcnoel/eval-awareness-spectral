# Verification and provenance report

## Provenance

- Target repository: `vcnoel/eval-awareness-spectral`
- Starting revision: `4bfd36c39494b5425276de0b893aea969503a49b`
- Protected export branch: `agent/export-10`
- Inspected public dependency: `vcnoel/spectral-trust` at `f80028c4f39fcc0a54f4976897c1cca76a2578dc`
- Dependency source was inspected only to verify its public operator/eigensolver API and the random-walk mathematical defect. No source was copied or vendored.
- Machine-readable outcomes: [`verification-results.json`](verification-results.json)

## Scientific corrections checked

- The basis-producing default is the symmetric normalized Laplacian.
- Random-walk basis analysis fails before eigendecomposition, including on regular graphs.
- Accepted decompositions check symmetry, finite values, eigenvalue order, Euclidean orthogonality, reconstruction, and eigenpair residual.
- Fixed-combinatorial Dirichlet energy is invariant to ordinary token-uniform shifts in synthetic tests.
- The degree-weighted normalized null mode and lifted DC/AC Frobenius accounting are tested.
- Source-task and template-family leakage into confirmation are rejected.
- Scaling, principal components, feature selection, tuning, probe fitting, and thresholds remain inside grouped training folds.
- Permutations rebuild the complete selected pipeline.
- Transfer requires template-family metadata and rejects shared rows, source tasks, template families, negative examples, and optionally model families.
- Untagged historical diagnostics fail instead of being relabeled as symmetric; affected legacy outputs must be recomputed.
- Causal tests target explicit layers/tokens/subspaces and require behavioral sign of life, bidirectional directions, and matched controls.

## Commands and outcomes

### Frozen environment

```bash
uv lock --check
```

Pass. The committed lock resolves the single `pyproject.toml` dependency declaration. NumPy is bounded below the incompatible typing-stub release.

### Python 3.11

```bash
UV_PYTHON_INSTALL_DIR="$HOME/.cache/uv/python" uv python install 3.11
UV_PYTHON_INSTALL_DIR="$HOME/.cache/uv/python" \
  uv run --isolated --python 3.11 --extra dev \
  pytest --cov=eval_awareness_spectral --cov-report=term-missing -q
UV_PYTHON_INSTALL_DIR="$HOME/.cache/uv/python" \
  uv run --isolated --python 3.11 --extra dev mypy src/eval_awareness_spectral
```

Pass: 69 tests, zero failures, and strict type checking over 28 source files. Statement coverage is 97.26% on the accepted plan's exact correctness surface: `claims.py`, `provenance.py`, `data/splits.py`, `evaluation/statistics.py`, `evaluation/transfer.py`, `representations/attention_graph.py`, and `representations/spectral.py`.

### Python 3.12

```bash
uv run --isolated --python 3.12 --extra dev ruff check \
  src tests paper/generate_results_table.py
uv run --isolated --python 3.12 --extra dev mypy src/eval_awareness_spectral
uv run --isolated --python 3.12 --extra dev \
  pytest --cov=eval_awareness_spectral --cov-report=term-missing -q
```

Pass: Ruff clean; strict mypy clean; 69 tests and zero failures. The same seven-file registered correctness surface reaches 97.26% statement coverage against a 90% threshold; the percentage is not presented as whole-package coverage.

### Claim and configuration validation

```bash
uv run --isolated --python 3.12 --extra dev \
  eval-awareness-spectral validate-config configs/protocol.example.json
uv run --isolated --python 3.12 --extra dev \
  eval-awareness-spectral claim-audit --ledger paper/claim_ledger.json \
  README.md docs/*.md main.tex
```

Pass. The example configuration hash is `sha256:daeeb7f418ab8c35bdda61d94586c8e24e01c227122ea84de3040b89c402c881`. Ten prose files passed with no unmanifested empirical number.

### Clean wheel installation

```bash
uv build --wheel
uv venv --python 3.12 "$TEMP/venv"
uv pip install --python "$TEMP/venv/bin/python" \
  dist/eval_awareness_spectral-0.2.0-py3-none-any.whl
cd "$TEMP"
"$TEMP/venv/bin/eval-awareness-spectral" --help
"$TEMP/venv/bin/eval-awareness-spectral" schema
"$TEMP/venv/bin/eval-awareness-spectral" validate-config \
  "$CHECKOUT/configs/protocol.example.json"
```

Pass outside the source tree. The reviewed wheel is 128,088 bytes; package data resolves; CLI help, schema, and config validation work. Import inspection confirmed that `torch`, `transformers`, `datasets`, `spectral_trust`, `goodfire_core`, `causalab`, `sglang`, and `lm_eval` were not imported.

### Manuscript

```bash
tectonic -X compile main.tex --outdir /tmp/eas-paper-build \
  --keep-logs --keep-intermediates
```

Pass with Tectonic 0.15.0: six-page PDF, nine resolved bibliography entries, no undefined citation or reference. The repository CI independently compiles with `latexmk` in a CPU-only GitHub Action.

### Repository and execution scan

The final diff against `origin/main` contains no result/model artifact extension, no Lab-local experiment or subagent path, no direct optional/private-runtime import, and no hard-coded host instruction. No model or tokenizer was loaded; no dataset was downloaded; no inference, activation harvest, generation, steering, GPU allocation, or empirical experiment was run. Package and TeX dependencies were the only network downloads.

## Residual risks

- No artifact-backed empirical study exists. The repository is intentionally pre-results, and publishable conclusions require a separately approved run.
- The deprecated empirical runner was not executed. Historical quantities affected by the random-walk defect must be recomputed rather than migrated by relabeling.
- Optional Goodfire, causalab, SGLang, and evaluation-harness seams were validated with structural CPU mocks, not live private runtimes.
- The workflow file parses locally and all equivalent commands pass. Hosted GitHub Actions can run only after the draft PR is opened.
