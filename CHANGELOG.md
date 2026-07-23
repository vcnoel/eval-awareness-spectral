# Changelog

## Unreleased

### Generation validity

- Added a dependency-free, typed generation contract that preserves scalar or sequence-valued model terminal IDs and tokenizer fallbacks.
- Added structured termination, conservative online and post-hoc repetition detection, reasoning/final span parsing, scoring eligibility, and generated-token activation alignment.
- Added an explicit opt-in online stopper for the sampled legacy path; causal and benchmark paths keep decoding unchanged and classify validity post hoc.
- Added a checksum-addressed compact regression fixture derived from the Qwen3-0.6B 4,096-token loop and negative controls for legitimate repetitive output.
- Historical outputs without termination provenance remain uncertified and are not reinterpreted.

### Resource accounting

- Added machine-readable experiment availability, a checksummed Qwen GPU ledger snapshot, verified burn arithmetic, and standalone Markdown/HTML findings.
- Missing accelerator and RU records are represented as unavailable rather than zero; the documented RU/H100-hour ratio is labeled an over-attribution ceiling, not a billing rate.

### Documentation and release contract

- Reframed the project around environment cue E, recognition R, and behavioral propensity P.
- Marked the repository as pre-results: no validated empirical artifacts or historical numerical claims are bundled.
- Documented symmetric-normalized spectral analysis as the default and categorical rejection of random-walk basis diagnostics.
- Added a future experiment protocol covering grouped confirmation, nested fitting, full-pipeline permutations, controls, transfer, scaling, interventions, endpoints, stop gates, and provenance.
- Replaced empirical law language with candidate observations and falsifiable conjectures.
- Added a corrected theory index, mathematical notes, research intuitions, and a legacy migration guide.
- Replaced vendor-host operational instructions with a research-release skill based on configuration, provenance, analysis, manuscript, and claim gates.
- Defined offline, no-inference validation and stop-before-inference behavior when required manifests are absent.

### Compatibility

- The core command line is CPU-safe and validation-first.
- Historical `run` and `analyze` behavior is deprecated behind the explicit `legacy` command and optional legacy dependencies.
- Basis-producing random-walk defaults are not retained; legacy runs must migrate to symmetric normalization before their spectral quantities can support current claims.
