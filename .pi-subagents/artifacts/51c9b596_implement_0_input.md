# Task for implement

Implement only src/eval_awareness_spectral/evaluation/{__init__.py,probes.py,statistics.py,selectivity.py,transfer.py,scaling.py} and focused tests tests/unit/test_evaluation_*.py. Do not edit any other files. Use NumPy/scikit-learn and existing ValidationError. Requirements: probes.py provides a leakage-safe sklearn Pipeline where StandardScaler, optional PCA, optional SelectKBest, logistic probe, and threshold/tuning all fit inside each training fold; provide nested group CV for development and a separate fit-on-development/evaluate-on-confirmation function that never fits confirmation rows. statistics.py provides orientation-invariant decodability (max(AUC,1-AUC) and above-chance magnitude), task-bootstrap CIs, and finite-input checks. selectivity.py exposes a direct paired interaction/selectivity estimate comparing target and negative-control decodability by source task with task-bootstrap CI; do not implement signed raw-AUC subtraction. Add a full-pipeline permutation test that reruns estimator construction, tuning, and feature selection every permutation. transfer.py defines typed transfer identities and rejects any shared row ID, source_task_id, or negative-example identity; optionally requires held-out model family; provide dimension-agnostic activation summary features and an explicit aligned-feature path. scaling.py provides hierarchical task/family effect estimation with cluster bootstrap uncertainty and an evidence gate that refuses 'law' wording below configured task-family and model-family minimums. Tests must be CPU synthetic and prove: reversed scores give same decodability; transforms fit only training rows; confirmation rows never influence PCA/scaler/tuning; permutation rebuilds the full pipeline; target/control interaction is paired by task; shared negatives and source tasks are rejected; width-independent summaries work; aligned activation transfer is allowed; insufficient family evidence raises; adequate synthetic hierarchy returns uncertainty. Strict typing/Ruff. Return concise summary and passing test count.

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