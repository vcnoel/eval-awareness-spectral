"""Expand the task pool from 15 hand-written items to a real, benchmark-grounded set.

15 tasks gives no usable confidence interval and is not defensible for a submission:
one task swap can move every downstream statistic. This pulls a fixed, seeded sample
from GSM8K (math reasoning), ARC-Easy (science reasoning/knowledge), and MMLU (broad
knowledge) -- exactly the benchmarks `docs/experiment_design.md` specified -- and
merges them with the original 15 hand-written items (which cover categories the
benchmarks don't: safety-advice, ethics, self-reflection, persona, neutral-control).

Writes data/tasks_bench.jsonl in the same {id, category, task} schema as tasks.jsonl,
so it is a drop-in --tasks-path for the runner/pod driver with no code changes.

    python scripts/build_task_pool.py --per-bench 15   # -> 15 (curated) + 45 = 60 tasks
    python scripts/build_task_pool.py --per-bench 25   # -> 15 + 75 = 90 tasks
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from eval_awareness_spectral.sandbagging import load_benchmark  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bench", type=int, default=15,
                    help="items per benchmark (gsm8k/arc/mmlu); total = 15 + 3x this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "tasks_bench.jsonl"))
    ap.add_argument("--max-chars", type=int, default=600,
                    help="skip items whose stem is longer than this (keeps prompts short)")
    args = ap.parse_args()

    curated = [json.loads(l) for l in
              (ROOT / "data" / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    rows = list(curated)
    for bench, cat in [("gsm8k", "bench_gsm8k_math"), ("arc", "bench_arc_science"),
                       ("mmlu", "bench_mmlu_knowledge")]:
        items = load_benchmark(bench, n=args.per_bench * 3, seed=args.seed)  # oversample, then filter
        kept = 0
        for it in items:
            if kept >= args.per_bench:
                break
            if len(it.question) > args.max_chars:
                continue
            rows.append({"id": it.id, "category": cat, "task": it.question})
            kept += 1
        print(f"{bench}: kept {kept}/{args.per_bench} (after length filter)")

    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate task ids"

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(rows)} tasks -> {args.out}")
    from collections import Counter
    for cat, n in Counter(r["category"] for r in rows).items():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
