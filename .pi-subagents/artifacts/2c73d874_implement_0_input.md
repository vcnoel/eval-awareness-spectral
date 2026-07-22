# Task for implement

Implement only the data protocol files under src/eval_awareness_spectral/data/ and their focused tests under tests/unit/test_data_*.py. Do not edit any other files. Use the existing schemas.py/provenance.py. Requirements: (1) factors.py must define independently toggled environment factors, keep recognition_label and behavioral_outcome distinct, expose a license-compliant EvalAwareBench-style adapter that accepts a caller-supplied local JSONL/CSV path only (no network), and reject legacy aliasing; (2) manifests.py must build PromptRecord values with stable source_task_id, template_family, factor vector, condition valence, content/rendered/task-token SHA-256 hashes, tokenizer revision, task span, and split; serialize JSONL atomically and validate hashes/required fields; (3) splits.py must deterministically assign grouped rows by source_task_id, prohibit generated variants of one task from crossing train/validation/test/confirmation, require template-family-disjoint confirmation, expose nested group-development folds, and raise ValidationError on leakage; (4) __init__.py re-exports public APIs. Tests must be CPU-only synthetic, deterministic, and prove source-task leakage, template-family confirmation leakage, repeated variants, and adapter path-only behavior are rejected. Use strict typing and current Ruff style. Return a concise summary and tests added.

[silico] If you write an ```acceptance-report``` fenced block: every commandsRun[].result and criteriaSatisfied[].status must be EXACTLY one of "passed", "failed", or "not-run" — any other wording (e.g. "failed validation") makes the parser reject the whole report.

## Acceptance Contract
Acceptance level: attested
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Return concrete findings with file paths and severity when applicable

Required evidence: review-findings, residual-risks

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```