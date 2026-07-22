# Task for implement

Implement only src/eval_awareness_spectral/interventions/{__init__.py,patching.py,steering.py,mediation.py} and focused tests tests/unit/test_interventions_*.py. Do not edit any other files. NumPy synthetic only. Requirements: define typed target specs for layers plus token roles/explicit indices or spans and subspace bases; validate shape/rank/finite values. patching.py must provide token-aligned residual patching and subspace interchange that modify only specified layers/tokens/subspaces, plus equal-rank random, shuffled-label, placebo-cue, and full-residual control descriptors/builders. steering.py must apply residual steering only at specified layers/tokens; support both deployment-to-evaluation sufficiency and evaluation-to-deployment necessity direction labels without silently steering generated/unselected tokens. mediation.py must define behavior endpoint interfaces: gold scoring or a separately validated judge is scientific; keyword scoring is explicitly debug-only and rejected for causal claims. Add a behavioral sign-of-life gate that returns status instrument_not_testable when validated condition-linked outcome variation is below a configured threshold. Mediation must first verify the internal score moved in the intended direction and only then report the behavioral endpoint; projection movement alone must return no_internal_movement or no_behavioral_mediation, never mediation. Provide typed bidirectional sufficiency/necessity result validation and require both directions plus control results. Tests prove exact layer/token targeting, rank-limited patching, reverse direction sign, equal-rank controls, shuffled/placebo/full controls are identified, untestable behavior gate, debug scorer rejection, projection-only result not called mediation, and complete bidirectional control requirements. Strict typing/Ruff. Return concise summary and passing test count.

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