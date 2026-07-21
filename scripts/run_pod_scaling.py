"""One-shot scaling run for a cloud GPU (A100 80GB): the Qwen within-family ladder.

Downloads each Qwen2.5-Instruct model, extracts position-matched task-subgraph
spectral diagnostics + per-token activations, then runs the crystallization
law-mining and the cross-architecture spectral transfer, printing the tables that
decide the falsifiable targets F1-F3 (docs/laws.md).

Qwen is fully open (no gating). On an 80GB A100, bf16 fits through 32B; 72B needs
2 GPUs (omitted). Analysis is float32/float64 regardless of weight dtype.

    python scripts/run_pod_scaling.py                 # 0.5,1.5,3,7,14,32B
    python scripts/run_pod_scaling.py --max-b 14      # stop at 14B
    python scripts/run_pod_scaling.py --tasks 15
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch  # noqa: E402
from eval_awareness_spectral.runner import RunConfig, run_model  # noqa: E402
from eval_awareness_spectral.activations import extract_activations  # noqa: E402
from eval_awareness_spectral import law_mining, spectral_transfer, decomposition  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pod")

# (name, hf_id, params_b, hidden_dim) — Qwen2.5 within-family ladder, all open-weights
LADDER = [
    ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", 0.5, 896),
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", 1.5, 1536),
    ("qwen2.5-3b",   "Qwen/Qwen2.5-3B-Instruct",   3.0, 2048),
    ("qwen2.5-7b",   "Qwen/Qwen2.5-7B-Instruct",   7.0, 3584),
    ("qwen2.5-14b",  "Qwen/Qwen2.5-14B-Instruct",  14.0, 5120),
    ("qwen2.5-32b",  "Qwen/Qwen2.5-32B-Instruct",  32.0, 5120),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=15)
    ap.add_argument("--max-b", type=float, default=32.0, help="skip models larger than this")
    ap.add_argument("--out", default=str(ROOT / "results" / "pod_scaling"))
    args = ap.parse_args()

    out = Path(args.out)
    subset = out / "_tasks.jsonl"
    subset.parent.mkdir(parents=True, exist_ok=True)
    lines = [l for l in (ROOT / "data" / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    subset.write_text("\n".join(lines[: args.tasks]) + "\n", encoding="utf-8")
    conds = ["control", "deploy", "eval", "eval_subtle", "placebo"]

    ladder = [m for m in LADDER if m[2] <= args.max_b]
    done = []
    for name, hf_id, pb, dim in ladder:
        log.info("========== %s (%.1fB, bf16) ==========", name, pb)
        try:
            rc = RunConfig(name=name, hf_id=hf_id, dtype="bfloat16", device="cuda",
                           conditions=conds, cells=[("task_subgraph", "rw")], directed=False,
                           position_match=True, output_dir=str(out), tasks_path=str(subset),
                           local_files_only=False)  # allow download on the pod
            run_model(rc)  # spectral diagnostics
            extract_activations(rc, out_npz=str(out / name / "activations.npz"),
                                save_token_acts=True, token_acts_max_tasks=args.tasks)
            done.append((name, hf_id, pb, dim))
        except Exception as e:
            log.error("model %s FAILED (skipping): %s", name, e)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not done:
        log.error("no models completed"); return

    rel = lambda name: f"pod_scaling/{name}"
    law_models = [(rel(n), n, pb, "qwen") for n, _, pb, _ in done]
    transfer_models = [(rel(n), n, dim) for n, _, _, dim in done]

    print("\n\n##################  CRYSTALLIZATION (Qwen within-family ladder)  ##################")
    law_mining.run(models=law_models)
    print("\n\n##################  CROSS-ARCHITECTURE SPECTRAL TRANSFER  ##################")
    spectral_transfer.run(models=transfer_models)
    print("\n\n##################  SELECTIVITY CROSSOVER per model  ##################")
    for n, _, _, _ in done:
        try:
            decomposition.run(str(out / n), n_perm=500)
        except Exception as e:
            log.error("decomposition %s failed: %s", n, e)

    print("\nAll scaling outputs under:", out)
    print("Decision: F1 = C4 up / C10 down through 14-32B; F2 = mismatch <= -0.2;",
          "F3 = transfer AUC climbs with scale.")


if __name__ == "__main__":
    main()
