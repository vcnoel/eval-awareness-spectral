---
name: eval-awareness-spectral
description: Design, validate, or interpret evaluation-cue research while keeping environment cue E, recognition R, and behavioral propensity P separate. Use for local manifest validation, spectral comparison design, grouped evaluation, causal follow-up, and claim-safe interpretation.
---

# Evaluation-cue spectral research contract

This repository is pre-results. Do not quote historical metrics, infer outcomes from old module names, or describe a conjecture as observed. No validated empirical artifacts are bundled.

## Start with E, R, and P

- **E** is an assigned environment-factor vector.
- **R** is an independently measured recognition endpoint.
- **P** is a behavioral propensity estimated from outcome-bearing repeated samples.

Prompt-only activations can measure sensitivity to E. They do not establish R or P. Avoid belief, awareness, deception, or intent language unless a validated endpoint directly operationalizes the term.

## Stop-before-inference gate

Before any model call, require and locally validate:

1. hashed protocol configuration;
2. prompt manifest with stable source-task IDs, template families, factor vectors, hashes, tokenizer revision, task spans, and splits;
3. group-and-template-isolated confirmation split;
4. exact model/tokenizer and adapter revisions;
5. gold endpoint or a frozen judge-validation plan for R/P;
6. artifact destination reserved by configuration hash;
7. preregistered contrasts, controls, fitting procedure, statistics, sign-of-life threshold, and stop rules.

If any item is absent, stop before inference. Do not repair missing provenance by guessing after a run.

## Spectral method

Use the symmetric-normalized Laplacian for basis-dependent analysis. Its Euclidean-orthonormal null mode is degree weighted. Reject categorical random-walk basis requests; random-walk right eigenvectors are not generally an orthonormal Fourier basis. The combinatorial operator is permitted for a labeled fixed-graph ordinary-DC comparison.

Declare every analysis as graph-only, fixed-reference-signal, or joint graph-and-signal. Do not compare unaligned coefficients from condition-specific bases. Use orientation-invariant paired measures for eigenvectors and learned directions.

## Evaluation method

Group by source task and template family. Fit normalization, dimension reduction, layer/feature selection, direction orientation, and hyperparameters inside nested training folds. Complete-pipeline permutations must rerun those choices. Report both target and nuisance-control performance alongside paired selectivity.

Required alternatives include position, length, filler, tokenizer boundaries, lexical template, condition valence, task difficulty, residual norm, ordinary DC/AC, degree-weighted null/perpendicular content, graph-only effects, and value/output controls. Transfer claims require genuinely disjoint task, template, model-family, or endpoint groups.

Scaling language requires repeated sizes within multiple families, multiple task families, hierarchical estimation, and leave-one-family-out sensitivity. A checkpoint trend is a candidate observation only.

## Causal follow-up

Intervene only after an observational validation sign of life and endpoint validation. Derive targets on training data. Include random, placebo, off-target, matched-norm, dose, quality, and scorer controls. State whether evidence addresses sufficiency, necessity, or mediation; none establishes intent.

## Outputs and claims

Write atomic, checksummed artifacts and a result manifest recording config, splits, revisions, seeds, environment, raw endpoints, fold fits, metrics, failures, and interventions. Run release validation offline and without inference. Promote an empirical sentence only when its claim-ledger entry links to a validated manifest. Otherwise label it planned, conjecture, falsified, or reported-unverified.

Primary references: [README](../../../README.md), [protocol](../../../docs/future-experiment-protocol.md), [mathematics](../../../docs/mathematical-notes.md), and [candidate conjectures](../../../docs/laws.md).
