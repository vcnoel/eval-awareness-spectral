# Task for implement

Implement only src/eval_awareness_spectral/representations/{__init__.py,attention_graph.py,residual.py,spectral.py} and focused tests tests/unit/test_representations_*.py. Do not edit any other files. Read current schemas.py for ValidationError and SpectralComparisonMode. Requirements: define GraphOperator as StrEnum with symmetric-normalized (default basis path), combinatorial, and random-walk-legacy. Build Laplacians from finite nonnegative square adjacency. Basis-dependent analysis must reject random-walk-legacy categorically, even on a regular graph. Validate symmetry, finite outputs, sorted eigenvalues, Euclidean orthogonality, reconstruction/eigen residual, and shape; use numpy.linalg.eigh only after checks. Expose an immutable SpectralDecomposition and graph Fourier metrics (entropy, HFER, spectral-mass Gini, Fiedler) that only accept a validated basis. Separate graph-only change, signal change on a fixed reference graph, and joint graph-plus-signal change in typed outputs. residual.py must implement fixed-combinatorial-graph Dirichlet energy, DC/AC decomposition with the lifted DC component compared in the same Frobenius norm (token-count factor), symmetric-normalized degree-weighted null mode, and document that normalized metrics are not ordinary-DC-invariant. attention_graph.py should validate/symmetrize only by an explicit function (never silently). Tests: irregular legacy random-walk matrix is non-symmetric and basis call fails loudly; symmetric-normalized decomposition passes orthogonality/residuals; combinatorial raw energy is invariant to ordinary token-uniform shifts; at least one normalized metric changes under an ordinary uniform shift on an irregular graph; DC and AC squared norms sum to total and lifted DC norm includes sqrt(n); comparison modes stay distinct; nonfinite/asymmetric inputs fail. CPU NumPy only, strict typed Ruff style. Return concise summary and test count.

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