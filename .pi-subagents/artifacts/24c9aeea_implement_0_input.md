# Task for implement

Implement only src/eval_awareness_spectral/integrations/{__init__.py,standalone.py,goodfire.py} and focused tests tests/unit/test_integrations_*.py. Do not edit any other files. Public core must not import optional packages at module import time. standalone.py: functions to load/validate local prompt and results manifests and adapt plain NumPy activation arrays into the package's representation APIs. goodfire.py: protocol-based documented optional seams for (a) Goodfire ActivationDataset/LocalSequenceDataset-like containers, (b) causalab interchange runners, (c) SGLang steering callbacks, and (d) Goodfire lm-evaluation-harness/gold endpoint scoring. Do not copy, import at module import time, or reimplement any private library. Use lazy optional-import helpers with actionable OptionalIntegrationUnavailable errors, and accept duck-typed callbacks/containers so the public core installs without private access. Include a status function that inspects package availability without importing them. Tests prove importing eval_awareness_spectral.integrations imports no optional module, missing integrations fail actionably, local NumPy path works, duck-typed activation rows retain sequence IDs and layer/token provenance, and causal/steering/eval callbacks preserve explicit targets/endpoints. Strict typing/Ruff, CPU only. Return concise summary and passing test count.

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