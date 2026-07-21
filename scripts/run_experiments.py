"""Smoke test for the full experimental design on Llama-3.2-3B.

Runs — small and fast — the whole chain end to end, printing a confound check at
each step so we can see it is clean before buying H100 hours:

  1. spectral sweep      (task_subgraph + full cells)      -> diagnostics_long.csv
  2. activation extract  (mean-pooled task-token residual) -> activations.npz
  3. baseline battery    (9 feature families × 5 contrasts) -> selectivity table
  4. causal steering     (eval / placebo / random × α)      -> behavioral summary

Usage:
    python scripts/run_experiments.py            # 5-task smoke
    python scripts/run_experiments.py --tasks 15 # fuller (slower)
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from eval_awareness_spectral.runner import RunConfig, run_model  # noqa: E402
from eval_awareness_spectral.activations import extract_activations  # noqa: E402
from eval_awareness_spectral.baselines import run_baseline_battery, _selectivity_table  # noqa: E402
from eval_awareness_spectral.causal import run_causal  # noqa: E402
from eval_awareness_spectral.prompts import load_tasks  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("experiments")


def make_subset(n: int, out: Path) -> str:
    lines = [l for l in (ROOT / "data" / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines[:n]) + "\n", encoding="utf-8")
    return str(out)


def main():
    # never let a stray non-ASCII char in a print crash the run on a cp1252 console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--model", default="llama-3.2-3b-it")
    ap.add_argument("--hf-id", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "results" / "experiments_smoke"))
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--skip-causal", action="store_true")
    ap.add_argument("--position-match", action="store_true",
                    help="pad preambles so the task starts at the same token index in every frame")
    ap.add_argument("--fresh", action="store_true", help="recompute steps 1-2 even if cached")
    args = ap.parse_args()

    out = Path(args.out)
    tasks_path = make_subset(args.tasks, out / "_tasks_subset.jsonl")
    conditions = ["control", "deploy", "eval", "eval_subtle", "placebo"]

    # Under position-matching the preamble is mostly filler, so the FULL-scope graph
    # is meaningless (and its huge, filler-heavy Laplacian is slow/degenerate). Use
    # only the task_subgraph cell in that mode; keep both cells otherwise.
    cells = ([("task_subgraph", "rw")] if args.position_match
             else [("full", "rw"), ("task_subgraph", "rw")])
    rc = RunConfig(
        name=args.model, hf_id=args.hf_id, dtype="float32", device="cuda",
        conditions=conditions, cells=cells,
        directed=False,  # not needed for the probe comparison; keep smoke fast
        position_match=args.position_match,
        output_dir=str(out), tasks_path=tasks_path,
    )
    if args.position_match:
        log.info("POSITION-MATCHED frames enabled (task starts at same token index); "
                 "using task_subgraph cell only")

    long_csv = out / rc.name / "diagnostics_long.csv"
    npz_path = out / rc.name / "activations.npz"

    # ---- 1. spectral (reuse if cached) ----
    if long_csv.exists() and not args.fresh:
        log.info("STEP 1/4 spectral sweep — reusing %s", long_csv)
        long_df = pd.read_csv(long_csv)
    else:
        log.info("STEP 1/4 spectral sweep")
        long_df = run_model(rc)

    # ---- 2. activations (reuse if cached) ----
    if npz_path.exists() and not args.fresh:
        log.info("STEP 2/4 activation extraction — reusing %s", npz_path)
        import numpy as _np
        d = _np.load(npz_path, allow_pickle=True)
        npz = {k: d[k] for k in d.files}
        npz["model"] = str(npz["model"])
    else:
        log.info("STEP 2/4 activation extraction")
        npz = extract_activations(rc, out_npz=str(npz_path))

    if long_df.empty or "graph_scope" not in long_df.columns:
        log.error("spectral step produced no rows (all subgraphs skipped?) — aborting. "
                  "Check span warnings above.")
        return

    # confound check A: identical node set + position gap for the headline contrast
    _confound_report(long_df, npz)

    # ---- 3. baselines ----
    log.info("STEP 3/4 baseline battery")
    res = run_baseline_battery(long_df, npz, out, n_boot=args.n_boot, n_perm=args.n_perm)
    if not res.empty:
        print("\n=== SELECTIVITY  S = AUC(eval-vs-deploy) - AUC(deploy-vs-placebo) ===")
        print(_selectivity_table(res).round(3).to_string(index=False))
        print("\n=== NEGATIVE CONTROL (deploy vs placebo): every family should be ~0.5 ===")
        neg = res[res.contrast == "deploy_vs_placebo"][["family", "auc", "perm_p"]]
        print(neg.round(3).to_string(index=False))
        print("\n=== HEADLINE (eval vs deploy) by family ===")
        head = res[res.contrast == "eval_vs_deploy"][["family", "auc", "auc_ci_lo",
                                                       "auc_ci_hi", "perm_p", "n_features"]]
        print(head.round(3).sort_values("auc", ascending=False).to_string(index=False))

    # ---- 4. causal ----
    if not args.skip_causal:
        log.info("STEP 4/4 causal steering")
        cdf = run_causal(rc, npz, out, alphas=(0.0, 4.0, 8.0))
        summ_path = out / "causal_summary.csv"
        if summ_path.exists():
            print("\n=== CAUSAL: behavioral readouts by direction x alpha (deploy prompts) ===")
            print("    v_eval should shift meta_eval/hedging up with alpha;")
            print("    v_placebo & v_random should NOT (specificity controls).")
            print(pd.read_csv(summ_path).round(3).to_string(index=False))

    print(f"\nAll outputs under: {out}")


def _confound_report(long_df, npz):
    print("\n=== CONFOUND CHECK ===")
    sub = long_df[(long_df.graph_scope == "task_subgraph") & (long_df.normalization == "rw")]
    g = sub.groupby("condition").agg(n_tokens=("n_tokens", "mean"),
                                     n_nodes=("n_nodes", "mean"),
                                     n_nodes_std=("n_nodes", "std"))
    print("length (n_tokens) & node set (n_nodes) per condition:")
    print(g.round(2).to_string())
    import numpy as np
    df = pd.DataFrame({"condition": npz["condition"], "task_start": npz["task_start"]})
    pos = df.groupby("condition")["task_start"].mean()
    print("\ntask-span start position per condition:")
    print(pos.round(1).to_string())
    if "eval" in pos and "deploy" in pos:
        print(f">>> eval-deploy position gap: {pos['eval'] - pos['deploy']:+.1f} tokens "
              f"(want ~0 so the headline isn't a RoPE artifact)")


if __name__ == "__main__":
    main()
