Implemented the representations package and focused tests. All 10 tests pass.

Key decisions:
- `random-walk-legacy` is categorically rejected for basis-dependent analysis, including regular graphs.
- Symmetrization is explicit; graph construction never silently changes adjacency.
- Spectral arrays are deeply read-only, and metrics revalidate the basis before use.
- Comparison modes use distinct frozen output types.
- Normalized residual metrics explicitly document their lack of ordinary-DC invariance.

Residual risk: HFER defaults to the upper half of sorted spectral modes; callers needing a domain-specific cutoff should pass `high_frequency_start`.