## File/function map (target repo `src/eval_awareness_spectral/`)

- **decomposition.py** — `_moment_traj`, `_spectral_traj(df, metrics, scope="task_subgraph", norm="rw")`, `_S`, `shift_magnitude`, `run(results_model_dir, n_perm=1000)`. Tests Theorem 1 (DC/AC probe crossover); spectral probe pulled from the `(task_subgraph, rw)` cell.
- **freq_decomp.py** — `_shift_spectrum(X_a, X_b, Phi, lam)`, `run(rc, tasks_n=15, layer_lo_frac=0.4)`, `summarize(df)`, `main()`. Tests Theorem 2 (P4, graph-frequency of the shift). Explicitly sets `cfg.normalization = "sym"` with an inline comment that `rw`'s eigenvectors are non-orthonormal and would make `Phi.T @ Δ` energies meaningless.
- **analysis.py** — `load_long`, `cells_in`, `layer_effects`, `model_summary`, `condition_means`, `compute_velocity`, `velocity_cluster_effects`, `peak_velocity_stability`, `run_analysis(long_csv, out_dir, ...)`. `PRIMARY_METRICS = [smoothness_index, spectral_entropy, hfer, gini_sparsity]`; defaults to whatever cell exists in the CSV, primary cell = `(task_subgraph, rw)`.
- **separability.py** — `build_features`, `probe_separability`, `per_layer_probe`, `transfer_test`, `davis_kahan_rotation`, `run(...)`, all default `normalization: str = "rw"`.
- **spectral_transfer.py** — `model_features(df, scope="task_subgraph", norm="rw")`, default `norm="rw"`.
- **metric_study.py** — `run(results_model_dir, scope="task_subgraph", norm="rw", n_perm=1000)`, greedy/pairwise metric selection, default `norm="rw"`.
- **law_mining.py**, **causal.py** — no direct normalization selection; consume the long CSV cells produced upstream (inherit whatever normalization the runner used, default `"rw"` per `RunConfig`/CLI).

Reference library (`spectral_trust`, commit `f80028c4f39f`):
- `graph.py: GraphBuilder.construct_laplacian(adjacency)` — builds `rw`: `L = I − D⁻¹W`, `sym`: `L = I − D^{-1/2}WD^{-1/2}`, `none`: `L = D − W`.
- `spectral.py: SpectralAnalyzer.compute_eigendecomposition(laplacian)` — dense path calls `scipy.linalg.eigh(laplacian)` unconditionally, regardless of `config.normalization`. Sparse path uses `eigsh` (also symmetric-only).
- `spectral.py: analyze_layer(...)` — computes `energy`/`smoothness_index` directly via `Tr(XᵀLX)` (no eigendecomposition needed), but `spectral_entropy`, `hfer`, `spectral_masses`, `fiedler_value`, `spectral_radius` all require the eigenpairs from `compute_eigendecomposition`.

## The mathematical defect

`L_rw = I − D⁻¹W` is generally **non-symmetric** (asymmetric unless the graph is regular, i.e. constant degree). `eigh`/`eigsh` assume a symmetric/Hermitian matrix and silently read only one triangular half of the input — they do not raise on asymmetric input. So when `normalization="rw"` (the pipeline-wide default across `analysis.py`, `separability.py`, `spectral_transfer.py`, `metric_study.py`, `decomposition.py`'s `_spectral_traj`), `compute_eigendecomposition` returns eigenpairs of an implicitly-symmetrized surrogate matrix, not of `L_rw` itself. Concretely:

- Correct relationship: `L_rw` is similar to `L_sym` via `L_rw = D^{-1/2} L_sym D^{1/2}`; both share the same real, non-negative eigenvalue spectrum, but `L_rw`'s eigenvectors are `D^{-1/2}v` (orthogonal only under the `D`-weighted inner product, not Euclidean), while `L_sym`'s eigenvectors are Euclidean-orthonormal.
- `eigh(L_rw)` does not recover either of these correctly — it discards the asymmetric part and returns eigenpairs of a matrix that isn't `L_rw` at all.
- Consequences: `spectral_entropy`, `hfer`, `gini_sparsity` (via `spectral_masses`), `fiedler_value`, and `spectral_radius` computed on the `(task_subgraph, rw)` cell — the pipeline's declared **primary** cell in `analysis.py` `PRIMARY_METRICS` — rest on a mathematically invalid eigendecomposition. `energy`/`smoothness_index` are unaffected since they never call `compute_eigendecomposition`.
- `freq_decomp.py` already documents and works around exactly this defect for its graph-Fourier transform by forcing `normalization="sym"`, but that fix is local to `freq_decomp.py` and is not propagated to the rest of the codebase, which still defaults to `norm="rw"`.

## Corrected invariant

For any downstream use of eigenvectors as an orthonormal Fourier basis (frequency-domain projection, entropy over spectral mass, HFER, Fiedler vector, sparsity of spectral mass), the Laplacian must be symmetric so `Φ` is orthonormal (`ΦᵀΦ = I`) and `Σ_k ‖Δ̂_k‖² = ‖Δ‖²` (Parseval) holds. Use `normalization="sym"` (or `"none"`, the combinatorial Laplacian, also symmetric when the adjacency is symmetric) for any metric requiring `eigh`. If `rw`-normalized spectral quantities are wanted, the eigendecomposition must go through the generalized eigenproblem `L_comb v = λ D v` (`scipy.linalg.eigh(L_comb, D)`), which correctly gives `D`-orthogonal eigenvectors — never plain `eigh(L_rw)`.

## Compatibility concerns

- Fixing this by switching defaults to `norm="sym"` will change every "primary cell" result computed to date under `(task_subgraph, rw)` in `analysis.py`, `separability.py`, `metric_study.py`, `decomposition.py`, `spectral_transfer.py`, and any long CSVs already on disk with `normalization=="rw"` rows for `spectral_entropy`/`hfer`/`fiedler_value`/`gini_sparsity` are silently unreliable and should be recomputed, not trusted as-is.
- `smoothness_index` and `energy` (direct quadratic-form metrics) are not affected by this defect and remain valid under `rw`.
- Any comparison across cells (`rw` vs `sym`) currently treats them as alternative normalization choices rather than "one is mathematically broken, one is not" — reports/tables that juxtapose `rw`- and `sym`-cell eigen-based metrics as equally valid readings need a caveat or correction.
- `freq_decomp.py`'s override to `sym` is correct but should be treated as the template fix, not a one-off; the other modules' `norm="rw"` defaults for eigen-based metrics are the actual defect surface.