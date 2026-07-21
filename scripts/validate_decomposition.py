"""Validate the DC/AC selectivity crossover (Theorem 1, docs/theory.md) across models.

For each small cached model: position-matched spectral sweep (task_subgraph cell) +
activation extraction (with per-token dumps) + the DC/AC/spectral selectivity crossover.
The tight claim holds iff, across models: S(DC) <= 0 < S(AC) ~ S(spectral).

    python scripts/validate_decomposition.py            # default small ladder
    python scripts/validate_decomposition.py --tasks 15
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd  # noqa: E402
from eval_awareness_spectral.runner import RunConfig, run_model  # noqa: E402
from eval_awareness_spectral.activations import extract_activations  # noqa: E402
from eval_awareness_spectral import decomposition  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("validate")

# small, fast, cached instruct models spanning families
SMALL = [
    ("qwen2.5-0.5b-it", "Qwen/Qwen2.5-0.5B-Instruct", "float32"),
    ("llama-3.2-1b-it", "meta-llama/Llama-3.2-1B-Instruct", "float32"),
    ("gemma-2-2b-it",   "google/gemma-2-2b-it", "bfloat16"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=15)
    ap.add_argument("--out", default=str(ROOT / "results" / "decomp_validation"))
    ap.add_argument("--n-perm", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out)
    subset = out / "_tasks.jsonl"
    subset.parent.mkdir(parents=True, exist_ok=True)
    lines = [l for l in (ROOT / "data" / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    subset.write_text("\n".join(lines[: args.tasks]) + "\n", encoding="utf-8")

    conds = ["control", "deploy", "eval", "eval_subtle", "placebo"]
    all_rows = []
    for name, hf_id, dtype in SMALL:
        log.info("=== %s ===", name)
        rc = RunConfig(name=name, hf_id=hf_id, dtype=dtype, device="cuda",
                       conditions=conds, cells=[("task_subgraph", "rw")], directed=False,
                       position_match=True, output_dir=str(out), tasks_path=str(subset))
        mdir = out / name
        try:
            run_model(rc)
            extract_activations(rc, out_npz=str(mdir / "activations.npz"),
                                save_token_acts=True, token_acts_max_tasks=6)
            res = decomposition.run(str(mdir), n_perm=args.n_perm)
            if res.get("crossover") is not None:
                t = res["crossover"].copy(); t.insert(0, "model", name); all_rows.append(t)
        except Exception as e:
            log.error("model %s failed: %s", name, e)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(out / "crossover_all_models.csv", index=False)
        print("\n############ DC/AC CROSSOVER ACROSS MODELS ############")
        print("want per model: S(DC) <= 0 < S(AC) ~ S(spectral)")
        print(combined.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
