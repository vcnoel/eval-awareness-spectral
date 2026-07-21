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

# (name, hf_id, params_b, hidden_dim, family, gated). Multi-family, <=14B.
# Open (gated=False) need no HF token; gated=True need HF_TOKEN + accepted license.
LADDER = [
    # Qwen2.5 within-family ladder (open)
    ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", 0.5, 896,  "qwen2.5", False),
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", 1.5, 1536, "qwen2.5", False),
    ("qwen2.5-3b",   "Qwen/Qwen2.5-3B-Instruct",   3.0, 2048, "qwen2.5", False),
    ("qwen2.5-7b",   "Qwen/Qwen2.5-7B-Instruct",   7.0, 3584, "qwen2.5", False),
    ("qwen2.5-14b",  "Qwen/Qwen2.5-14B-Instruct",  14.0, 5120, "qwen2.5", False),
    ("qwen2.5-32b",  "Qwen/Qwen2.5-32B-Instruct",  32.0, 5120, "qwen2.5", False),
    # SmolLM2 within-family ladder (open)
    ("smollm2-135m", "HuggingFaceTB/SmolLM2-135M-Instruct", 0.135, 576,  "smollm2", False),
    ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M-Instruct", 0.36, 960,   "smollm2", False),
    ("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct", 1.7, 2048,   "smollm2", False),
    # Phi within-family (open, MIT)
    ("phi3.5-mini",  "microsoft/Phi-3.5-mini-instruct", 3.8, 3072, "phi", False),
    ("phi4-mini",    "microsoft/Phi-4-mini-instruct",   3.8, 3072, "phi", False),
    ("phi4",         "microsoft/phi-4",                 14.7, 5120, "phi", False),
    # architecture-diversity single points (open)
    ("mistral-7b",   "mistralai/Mistral-7B-Instruct-v0.3", 7.0, 4096, "mistral", False),
    # gated (need HF_TOKEN + license) — auto-skip if unavailable
    ("gemma-2-2b",   "google/gemma-2-2b-it", 2.0, 2304, "gemma2", True),
    ("gemma-2-9b",   "google/gemma-2-9b-it", 9.0, 3584, "gemma2", True),
    ("llama-3.2-1b", "meta-llama/Llama-3.2-1B-Instruct", 1.0, 2048, "llama3", True),
    ("llama-3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", 3.0, 3072, "llama3", True),
    ("llama-3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", 8.0, 4096, "llama3", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=15)
    ap.add_argument("--max-b", type=float, default=9.0,
                    help="skip models larger than this (9 fits a ~30GB volume with --purge-weights)")
    ap.add_argument("--purge-weights", action="store_true",
                    help="delete each model's weights after extraction (small volume disk)")
    ap.add_argument("--out", default=str(ROOT / "results" / "pod_scaling"))
    args = ap.parse_args()

    out = Path(args.out)
    subset = out / "_tasks.jsonl"
    subset.parent.mkdir(parents=True, exist_ok=True)
    lines = [l for l in (ROOT / "data" / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    subset.write_text("\n".join(lines[: args.tasks]) + "\n", encoding="utf-8")
    conds = ["control", "deploy", "eval", "eval_subtle", "placebo"]

    import os
    has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    hf_cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"

    ladder = [m for m in LADDER if m[2] <= args.max_b]
    for name, hf_id, pb, dim, family, gated in ladder:
        if (out / name / "token_acts.pkl").exists():
            log.info("========== %s already done, skipping ==========", name)
            continue
        if gated and not has_token:
            log.info("========== %s gated, no HF_TOKEN — skipping ==========", name)
            continue
        log.info("========== %s (%s, %.2fB, bf16) ==========", name, family, pb)
        try:
            rc = RunConfig(name=name, hf_id=hf_id, dtype="bfloat16", device="cuda",
                           conditions=conds, cells=[("task_subgraph", "rw")], directed=False,
                           position_match=True, output_dir=str(out), tasks_path=str(subset),
                           local_files_only=False)  # allow download on the pod
            run_model(rc)  # spectral diagnostics
            extract_activations(rc, out_npz=str(out / name / "activations.npz"),
                                save_token_acts=True, token_acts_max_tasks=args.tasks)
        except Exception as e:
            log.error("model %s FAILED (skipping): %s", name, e)
        finally:
            if args.purge_weights:  # free the volume before the next model
                import shutil
                shutil.rmtree(hf_cache / f"models--{hf_id.replace('/', '--')}", ignore_errors=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Aggregate over ALL rungs whose results exist (so a resumed run re-uses prior ones).
    rel = lambda name: f"pod_scaling/{name}"
    present = [(n, pb, dim, fam) for n, _, pb, dim, fam, _ in LADDER
               if (out / n / "token_acts.pkl").exists()]
    if not present:
        log.error("no models completed"); return
    log.info("aggregating rungs: %s", [n for n, _, _, _ in present])
    law_models = [(rel(n), n, pb, fam) for n, pb, _, fam in present]
    transfer_models = [(rel(n), n, dim) for n, _, dim, _ in present]

    print("\n\n##################  CRYSTALLIZATION (per-family scaling)  ##################")
    law_mining.run(models=law_models)
    print("\n\n##################  CROSS-ARCHITECTURE SPECTRAL TRANSFER  ##################")
    spectral_transfer.run(models=transfer_models)
    print("\n\n##################  SELECTIVITY CROSSOVER per model  ##################")
    for n, _, _, _ in present:
        try:
            decomposition.run(str(out / n), n_perm=500)
        except Exception as e:
            log.error("decomposition %s failed: %s", n, e)

    print("\nAll scaling outputs under:", out)
    print("Decision: F1 = C4 up / C10 down through 14-32B; F2 = mismatch <= -0.2;",
          "F3 = transfer AUC climbs with scale.")


if __name__ == "__main__":
    main()
