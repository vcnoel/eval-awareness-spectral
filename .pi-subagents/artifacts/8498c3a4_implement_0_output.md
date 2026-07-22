Rewrote the documentation around E/R/P separation and pre-results epistemics. Added the mathematical proofs, executable future protocol, migration guide, changelog, and release skill; removed vendor-host instructions. No code, tests, project configuration, paper files, or staged files were changed by this task.

Key decisions:
- Symmetric-normalized Laplacian is the basis-producing default; categorical random-walk basis analysis is rejected.
- Historical findings are no longer asserted; laws are candidate observations or conjectures.
- Inference stops when required manifests, splits, endpoints, or provenance are absent.
- Release checks are offline, CPU-safe, no-network, and no-inference.
- Legacy behavior remains documented only as a migration path.

The shared worktree contains unrelated unstaged code/config changes from other workers; integration should select only the documentation paths listed below.