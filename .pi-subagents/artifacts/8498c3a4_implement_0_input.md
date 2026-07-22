# Task for implement

Rewrite documentation only. Allowed files: README.md; docs/research-intuitions.md; docs/mathematical-notes.md; docs/future-experiment-protocol.md; docs/migration-guide.md; docs/experiment_design.md; docs/laws.md; docs/theory.md; CHANGELOG.md; .claude/skills/eval-awareness-spectral/SKILL.md; delete .claude/skills/runpod-connect/SKILL.md and create .claude/skills/research-release/SKILL.md. Do not edit code, tests, pyproject, paper/, or main.tex. This repo is pre-results: no empirical artifacts are tracked and no historical numerical result may be asserted. Requirements: README leads with environment cue E vs recognition R vs behavioral propensity P; states no validated empirical artifacts are bundled; explains symmetric-normalized default and categorical random-walk-basis rejection; standalone install/CPU validation, local manifests, optional protocol-based Goodfire/causalab/SGLang/lm-eval integrations, output provenance, no-network/no-inference release checks, legacy compatibility/migration, AGPL. Research intuitions must include five accepted intuitions, each with simplest alternative and falsification test: fixed wrappers create low-rank nuisance; prompt-only activations measure cue sensitivity not stochastic recognition; attention graph summaries need value/output controls; recognition and propensity weakly coupled/training-dependent; every conjecture has alternatives. Mathematical notes must prove fixed-combinatorial DC invariance, degree-weighted symmetric-normalized null mode, lifted DC/AC Frobenius accounting with token-count factor, non-invariance/non-theorems, and three comparison modes. Future protocol encodes factor schema, group/template confirmation split, nested fitting, complete-pipeline permutations, orientation-invariant paired selectivity, disjoint transfer, hierarchical scaling evidence, causal interventions, controls, sign-of-life gate, gold/validated judge endpoint, stop gates, and artifact manifests—without running anything. laws.md must become candidate observations/conjectures only; theory.md must become corrected theory index; experiment_design.md point to executable protocol. Migration maps old CLI/modules/defaults to new. Skills are methodological/release contracts with stop-before-inference when manifests absent; research-release replaces vendor host instructions with config/provenance/analysis/manuscript/claim gates. Remove all stale result numbers, hard-coded hosts, vendor-specific pod instructions, unsupported law language, and belief claims. Use direct technical prose and valid relative links. Return summary.

[silico] If you write an ```acceptance-report``` fenced block: every commandsRun[].result and criteriaSatisfied[].status must be EXACTLY one of "passed", "failed", or "not-run" — any other wording (e.g. "failed validation") makes the parser reject the whole report.

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