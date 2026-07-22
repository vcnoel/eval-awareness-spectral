---
name: research-release
description: Prepare an evaluation-cue research release using configuration, provenance, analysis, manuscript, and claim gates. Use when packaging, auditing, or publishing protocol, results, or prose; never for host access or vendor-specific compute setup.
---

# Research release contract

A release is a provenance and claim-validation operation, not an inference run. It must complete offline and without importing model runtimes.

## Gate 1: configuration

Require a validated, hashed protocol configuration. It must freeze the factor schema, grouped splits, operator, comparison modes, endpoints, controls, nested fitting, complete-pipeline permutation, multiplicity handling, sign-of-life threshold, stop rules, and retry limit. If absent, stop before inference and do not stage a release.

## Gate 2: provenance

Require local prompt and split manifests; exact dataset, model, tokenizer, adapter, scorer, and software revisions; seeds; environment inventory; commands; failure ledger; and checksums for every referenced artifact. Confirmation groups must be isolated by source task and template family. Validate all checksums with network access disabled. Missing or mutable inputs block release.

## Gate 3: analysis

Require the frozen pipeline’s outputs, outer-fold fit metadata, complete-pipeline null, paired target and nuisance-control estimates, uncertainty, comparison-mode labels, and endpoint validation. Scaling claims need hierarchical multi-family evidence. Causal claims need intervention specificity and quality controls. Analysis that fails a stop gate may be released as a null or failed protocol stage, not promoted by retuning confirmation data.

## Gate 4: manuscript

Audit README, documentation, manuscript, captions, tables, and supplements against the claim ledger. Distinguish E, R, and P. Keep algebraic theorems separate from empirical mechanisms. Describe decodability as correlational and intervention claims by their actual estimand. Remove hard-coded hosts, credentials, local absolute paths, vendor instructions, unsupported law language, and unmanifested empirical numbers.

## Gate 5: claims

Each empirical claim must have a unique claim ID, allowed status, validated result manifest, and nearby prose marker where required by the audit tool. A result manifest must resolve every dataset, split, metric, endpoint, and intervention artifact and pass checksum validation. Claims without this chain remain planned, conjecture, falsified, or reported-unverified.

## Offline release check

Run only CPU-safe local validation:

```bash
eval-awareness-spectral validate-config path/to/protocol.json
eval-awareness-spectral validate-prompts path/to/prompts.jsonl
eval-awareness-spectral validate-results path/to/results_manifest.json
eval-awareness-spectral claim-audit --ledger path/to/claim_ledger.json README.md docs/*.md
```

Do not download, serve, infer, connect to a remote host, or regenerate results during this check. Archive or publish only the declared artifacts; never include secrets, caches, model weights, or unreviewed raw data. If any gate fails, stop the release, preserve the failure output, and return to the owning stage.

See [README](../../../README.md), [future protocol](../../../docs/future-experiment-protocol.md), and [migration guide](../../../docs/migration-guide.md).
