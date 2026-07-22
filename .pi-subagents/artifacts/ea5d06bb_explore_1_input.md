# Task for explore

Read-only architecture audit of vcnoel/eval-awareness-spectral in the current working directory at commit 4bfd36c39494. Inspect all tracked Python modules, scripts, pyproject, configs/data packaging assumptions, and any tests (likely none). Map old public APIs/CLI to the accepted target architecture: schemas/provenance, data factors/manifests/splits, representations, evaluation, interventions, optional integrations. Identify likely dependency, packaging, cache, atomic-write, completeness, logging, typing, and migration issues. Do not edit. Return concrete recommendations with filenames and preserved wrappers.

[silico] If you write an ```acceptance-report``` fenced block: every commandsRun[].result and criteriaSatisfied[].status must be EXACTLY one of "passed", "failed", or "not-run" — any other wording (e.g. "failed validation") makes the parser reject the whole report.

---
**Output:**
Write your findings to exactly this path: /srv/silico-state/_shared/silico/workspaces/ws_01kxp35gerep4b83c9jeg5gnjh/worktrees/issue-10/.pi-subagents/artifacts/outputs/ea5d06bb/explore-2.md
This path is authoritative for this run.
Ignore any other output filename or output path mentioned elsewhere, including output destinations in the base agent prompt, system prompt, or task instructions.

## Acceptance Contract
Acceptance level: reviewed
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope
- criterion-2: Return evidence sufficient for an independent acceptance review

Required evidence: changed-files, tests-added, commands-run, validation-output, residual-risks, no-staged-files

Review gate: required by reviewer.

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