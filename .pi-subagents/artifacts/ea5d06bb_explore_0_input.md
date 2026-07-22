# Task for explore

Read-only audit of the current target repository in the current working directory, focusing on existing spectral mathematics in src/eval_awareness_spectral/decomposition.py, freq_decomp.py, analysis.py, separability.py, spectral_transfer.py, law_mining.py, causal.py, metric_study.py, plus the read-only reference checkout /srv/silico-state/_shared/silico/workspaces/ws_01kxp35gerep4b83c9jeg5gnjh/resource-checkouts/19504421/repo at commit f80028c4f39f. Identify exact public APIs and the mathematical defect around random-walk vs symmetric-normalized Laplacians. Do not edit. Return a concise file/function map, corrected equations/invariants, and compatibility concerns. Do not copy source text beyond tiny public signatures.

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