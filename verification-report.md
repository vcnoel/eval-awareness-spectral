# Verification and provenance report

## Provenance

- Target repository: `vcnoel/eval-awareness-spectral`
- Base revision: `27cf2420bececeff902ba91df58ac4e4fa3a854a` (current `main` after merged PR #1)
- Protected export branch: `agent/export-12`
- Source experiment: `exp_01ky5kxdbgfvt8rhn7d03hs4ca`
- Regression source: `artifact://devoteam-069866/experiments/exp_01ky5kxdbgfvt8rhn7d03hs4ca/qwen3-0.6b/adaptive-shards-v1/short/stages_v1/cap4096_v1/rollouts.jsonl`, SHA256 `a2f10a9b51e1bc6123f3d8e3623aeb9bd71417630b16a9709ca42b754306ccfb`
- Machine-readable outcomes: [`verification-results.json`](verification-results.json) and [`reproduction-results.json`](reproduction-results.json)

The source experiment was read only. The export rewrites the termination logic as a typed dependency-free API; it does not copy the experiment runner, model glue, scheduler code, or activation implementation.

## Generation contract checked

- Model generation configuration accepts scalar, list, tuple, or absent terminal IDs; every declared ID is preserved in order.
- Tokenizer EOS is appended only as a compatible fallback.
- Primary EOS, secondary EOS, repetition loop, max-token cap, and error are mutually exclusive structured reasons.
- A repetition loop is always incomplete, degenerate, and ineligible for scoring, including after a reasoning close marker.
- Final spans distinguish EOS before close, close with empty final, cap after partial final, loop after close, missing close, and valid final.
- Nonterminal generated tokens stay one-to-one with selected activation indices for every termination reason.
- Online detection is deterministic over one unpadded row and rejects multi-row batches rather than globally stopping unrelated rows.
- Legacy decoding arguments are unchanged by default. Online stopping is available only through the explicit `--stop-repetition` flag; causal and benchmark paths classify post hoc. The historical eight-token verbalized-judge cap is configurable through `judge_max_new_tokens` and warns when it yields no eligible rows.
- Historical files without termination metadata are not retroactively reinterpreted.

## Reproduction check

The compact fixture preserves the actual failure's eight-token repeated unit, source positions, rollout ID, artifact reference, and source checksum. The detector catches six copies at exclusive generated index 329, before the recorded 4,096-token cap.

False positives are zero across five registered controls: a short refrain, a four-copy short chorus, an ordinary enumeration, SQL/code-like output, and punctuation-rich output. The ordered Qwen terminal set `[151645, 151643]`, batch-padding invariance, per-row terminals, close-marker variants, scorer eligibility, and activation alignment all pass.

The accounting reproduction also passes. Eleven Qwen ledger rows sum exactly to 14,427 H100-seconds, eight prior roots retain null usage fields rather than zero, and 229.4 RU divided by 4.0075 H100-hours yields the displayed 57.2427 RU/H100-hour over-attribution ceiling within `0.00005`. The report never labels that ceiling a billing rate.

## Commands and outcomes

### Python 3.11

```bash
uv run --isolated --python 3.11 --extra dev ruff check \
  src tests paper/generate_results_table.py scripts/build_resource_accounting.py
uv run --isolated --python 3.11 --extra dev mypy src/eval_awareness_spectral
uv run --isolated --python 3.11 --extra dev \
  pytest --cov=eval_awareness_spectral --cov-report=term-missing -q
```

Pass after the approved review edits: Ruff clean, strict mypy clean across 29 source files, 98 tests and zero failures in 11.82 seconds. Registered statement coverage is 96.42% against a 90% gate.

### Python 3.12

```bash
uv run --isolated --python 3.12 --extra dev ruff check \
  src tests paper/generate_results_table.py scripts/build_resource_accounting.py
uv run --isolated --python 3.12 --extra dev mypy src/eval_awareness_spectral
uv run --isolated --python 3.12 --extra dev \
  pytest --cov=eval_awareness_spectral --cov-report=term-missing -q
```

Pass after the approved review edits: Ruff clean, strict mypy clean across 29 source files, 98 tests and zero failures in 13.85 seconds. The same registered surface reaches 96.42% statement coverage.

### Arithmetic, claims, and lock

```bash
uv lock --check
uv run --no-sync eval-awareness-spectral validate-config configs/protocol.example.json
uv run --no-sync eval-awareness-spectral claim-audit \
  --ledger paper/claim_ledger.json README.md docs/*.md docs/upstream-issues/*.md main.tex
uv run --no-sync python scripts/build_resource_accounting.py --check
```

Pass: the 90-package lock is current; the configuration hash remains `sha256:daeeb7f418ab8c35bdda61d94586c8e24e01c227122ea84de3040b89c402c881`; 16 prose files pass the scientific claim audit; generated Markdown, HTML, and CSV accounting files exactly match the machine-readable input.

### Clean wheel and public API

```bash
uv build --wheel
python -m venv "$TEMP/venv"
"$TEMP/venv/bin/pip" install dist/eval_awareness_spectral-0.3.0-py3-none-any.whl
cd "$TEMP"
"$TEMP/venv/bin/eval-awareness-spectral" --help
"$TEMP/venv/bin/eval-awareness-spectral" schema
"$TEMP/venv/bin/eval-awareness-spectral" validate-config "$CHECKOUT/configs/protocol.example.json"
```

Pass outside the source tree. The wheel is 135,568 bytes. The public termination API imports and classifies secondary EOS correctly. Import inspection confirms that `torch`, `transformers`, `datasets`, `spectral_trust`, `goodfire_core`, `causalab`, `sglang`, and `lm_eval` are not loaded.

### Report, delivery, and repository scans

The standalone resource HTML parses with one `h1`, two complete tables, and no root-absolute local asset links. The three upstream issue packets cite exact lines at `f80028c4f39f`, `f75f834a1cd2`, and `9d852fae1a91`. Their historical-impact statements request audits and do not declare published results invalid.

The tracked patch contains no Lab-local path, secret, model artifact, scheduler command, GPU request, model/tokenizer load, dataset download, or inference call. The current export used zero scheduler jobs and zero GPU-seconds.

The export environment has git push authorization for the approved target but no authenticated GitHub issue API. No upstream issue was opened; complete bodies and direct manual-creation links are in [`docs/upstream-issues/`](docs/upstream-issues/). Draft-PR delivery remains pending until the reviewed branch is pushed.

## Residual risks

- The detector is deliberately conservative. It can miss irregular semantic loops below the repeated-mass or rolling-coverage thresholds.
- The online stopping adapter was tested through its dependency-free callable contract, not a live Transformers model, because model and tokenizer loading were prohibited.
- The verbalized baseline's historical eight-token default can yield no eligible rows on models that do not terminate promptly; new runs must raise and record `judge_max_new_tokens` rather than scoring capped text.
- Historical outputs lacking termination metadata still require repository-specific cap and repetition audits.
- The three upstream issues require manual creation in an authenticated GitHub session.
- The actual RU/H100-hour conversion remains unidentified; the report supplies only an explicit over-attribution ceiling.
