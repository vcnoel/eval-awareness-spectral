"""Report per-condition prompt token counts, to verify frames are length-matched.

    python scripts/audit_lengths.py [hf_id]

The eval/deploy/placebo frames must sit within a few tokens of each other or the
Dirichlet-energy comparison is confounded by length. Prints mean/min/max tokens
per condition across all base tasks, and the max pairwise gap among the
length-matched trio.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from eval_awareness_spectral.prompts import (  # noqa: E402
    build_prompts,
    load_framings,
    load_tasks,
)

MATCHED = ["deploy", "eval", "placebo"]


def main(hf_id: str = "google/gemma-2-2b-it"):
    tok = AutoTokenizer.from_pretrained(hf_id, local_files_only=True)
    tasks = load_tasks()
    conditions = load_framings()
    prompts = build_prompts(tasks, conditions, tok)

    by_cond = {}
    for p in prompts:
        n = len(tok(p.text)["input_ids"])
        by_cond.setdefault(p.condition, []).append(n)

    print(f"\nToken counts per condition ({hf_id}, N={len(tasks)} tasks)")
    print(f"{'condition':<14} {'mean':>7} {'min':>6} {'max':>6}")
    print("-" * 36)
    means = {}
    for cond in conditions:
        counts = by_cond.get(cond, [])
        if not counts:
            continue
        means[cond] = sum(counts) / len(counts)
        print(f"{cond:<14} {means[cond]:>7.1f} {min(counts):>6d} {max(counts):>6d}")

    matched = [means[c] for c in MATCHED if c in means]
    if len(matched) == len(MATCHED):
        gap = max(matched) - min(matched)
        print(f"\nMax mean-length gap among {MATCHED}: {gap:.1f} tokens", end="  ")
        print("[OK]" if gap <= 5 else "[TUNE FRAMES: gap > 5 tokens]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "google/gemma-2-2b-it")
