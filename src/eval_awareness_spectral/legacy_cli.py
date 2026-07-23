"""Command-line entrypoint.

Examples
--------
Prompt-only sweep over all cached models, then analyze eval-vs-deploy:
    python -m eval_awareness_spectral.cli run --analyze

Quick pilot on a single model with a subset of tasks:
    python -m eval_awareness_spectral.cli run --model gemma-2-2b-it --max-tasks 6 --analyze

Add the behavioral generation stage (slower, sampled):
    python -m eval_awareness_spectral.cli run --generate --runs 3 --analyze

Analyze an existing long CSV only:
    python -m eval_awareness_spectral.cli analyze --long results/diagnostics_long_all.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

from .runner import DEFAULT_CELLS, RunConfig, run_model
from .generation import run_model_with_generation
from .analysis import run_analysis
from .separability import run_separability
from .plots import make_all_figures

# Named cell presets: which (graph_scope, normalization) combinations to compute.
CELL_PRESETS = {
    "all": list(DEFAULT_CELLS),
    "primary": [("task_subgraph", "sym"), ("full", "sym")],
    "fast": [("task_subgraph", "sym")],
    "full": [("full", "sym")],
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("eval_awareness_spectral")

_ROOT = Path(__file__).resolve().parents[2]


def _load_model_configs(models_path: Path, only: Optional[List[str]]) -> List[dict]:
    with open(models_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    models = spec["models"]
    if only:
        wanted = set(only)
        models = [m for m in models if m["name"] in wanted]
        missing = wanted - {m["name"] for m in models}
        if missing:
            raise SystemExit(f"Unknown model name(s): {sorted(missing)}")
    return models


def _make_run_configs(args) -> List[RunConfig]:
    models = _load_model_configs(Path(args.models), args.model)
    conditions = args.conditions.split(",") if args.conditions else None
    cells = CELL_PRESETS[args.cells]
    rcs = []
    for m in models:
        rcs.append(
            RunConfig(
                name=m["name"],
                hf_id=m["hf_id"],
                dtype=m.get("dtype", "float32"),
                device=args.device,
                conditions=conditions,
                runs=args.runs,
                directed=args.directed,
                cells=list(cells),
                output_dir=args.out,
                tasks_path=args.tasks,
                local_files_only=not args.allow_download,
            )
        )
    return rcs


def _subset_tasks(args) -> Optional[str]:
    """If --max-tasks is set, write a truncated tasks file and return its path."""
    if not args.max_tasks:
        return args.tasks
    src = Path(args.tasks) if args.tasks else _ROOT / "data" / "tasks.jsonl"
    lines = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines = lines[: args.max_tasks]
    dst = Path(args.out) / "_tasks_subset.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Using task subset of %d -> %s", len(lines), dst)
    return str(dst)


def cmd_run(args):
    args.tasks = _subset_tasks(args)
    rcs = _make_run_configs(args)

    import pandas as pd

    frames = []
    for rc in rcs:
        logger.info("=== Prompt-only stage: %s ===", rc.name)
        try:
            frames.append(run_model(rc))
        except Exception as e:  # OOM / missing weights shouldn't abort the whole sweep
            logger.error("Model %s failed to run (skipping): %s", rc.name, e)
            continue
        if args.generate:
            logger.info("=== Generation stage: %s ===", rc.name)
            try:
                run_model_with_generation(
                    rc,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    stop_repetition=args.stop_repetition,
                )
            except Exception as e:
                logger.error("Generation stage for %s failed: %s", rc.name, e)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    long_all = Path(args.out) / "diagnostics_long_all.csv"
    combined.to_csv(long_all, index=False)
    logger.info("Combined prompt-only diagnostics -> %s (%d rows)", long_all, len(combined))

    fiedler_all = _combine_fiedler_pickles(rcs, Path(args.out))

    if combined.empty:
        logger.error("No diagnostics were produced — skipping analysis. Check the errors above.")
        return
    if args.analyze:
        _do_analysis(str(long_all), args, fiedler_pkl=fiedler_all)


def _combine_fiedler_pickles(rcs, out_dir: Path):
    """Concatenate per-model Fiedler pickles into one for the Davis-Kahan analysis."""
    import pickle
    records = []
    for rc in rcs:
        p = Path(rc.output_dir) / rc.name / "fiedler_vectors.pkl"
        if p.exists():
            with open(p, "rb") as f:
                records.extend(pickle.load(f))
    if not records:
        return None
    combined = out_dir / "fiedler_vectors_all.pkl"
    with open(combined, "wb") as f:
        pickle.dump(records, f)
    return str(combined)


def cmd_analyze(args):
    _do_analysis(args.long, args, fiedler_pkl=args.fiedler)


def _do_analysis(long_csv: str, args, fiedler_pkl=None):
    out_dir = Path(args.out)
    logger.info("Analyzing %s (%s vs %s)", long_csv, args.cond_a, args.cond_b)
    res = run_analysis(long_csv, out_dir, cond_a=args.cond_a, cond_b=args.cond_b,
                       n_perm=args.n_perm)

    sep = None
    if not getattr(args, "no_separability", False):
        fp = fiedler_pkl or (out_dir / "fiedler_vectors_all.pkl")
        logger.info("Running separability suite (probe + transfer + Davis-Kahan)")
        sep = run_separability(long_csv, fp, out_dir, cond_a=args.cond_a, cond_b=args.cond_b,
                               scope=args.sep_scope, n_boot=args.n_boot, n_perm=args.n_perm)

    figs = make_all_figures(
        res["effects"], res["summary"], res["means"], _ROOT / "figures",
        cond_a=args.cond_a, cond_b=args.cond_b,
        velocity_clusters=res.get("velocity_clusters"), separability=sep,
    )
    logger.info("Wrote %d figures to %s", len(figs), _ROOT / "figures")

    if not res["summary"].empty:
        print("\n=== Top effects on PRIMARY (scale-invariant) metrics, mean |d_z| ===")
        prim = res["summary"]
        prim = prim[prim.get("is_primary", True)] if "is_primary" in prim else prim
        top = prim.sort_values("mean_abs_dz", ascending=False).head(15)
        cols = [c for c in ["model", "graph_scope", "normalization", "metric",
                            "mean_abs_dz", "peak_layer", "frac_sig_layers"] if c in top.columns]
        print(top[cols].to_string(index=False))
    if sep and sep.get("probe") is not None and not sep["probe"].empty:
        print("\n=== Leave-task-out separability (task_subgraph/sym) ===")
        print(sep["probe"][["model", "contrast", "n_tasks", "n_features",
                            "auc", "auc_ci_lo", "auc_ci_hi", "perm_p"]].to_string(index=False))
    if sep and sep.get("transfer") is not None and not sep["transfer"].empty:
        print("\n=== Subtle<->explicit transfer AUC ===")
        print(sep["transfer"][["model", "train", "test", "auc"]].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eval-awareness-spectral")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=str(_ROOT / "results"), help="output dir")
    common.add_argument("--cond-a", default="eval", help="condition A (default eval)")
    common.add_argument("--cond-b", default="deploy",
                        help="condition B (default deploy; use 'placebo' for the evaluativeness-isolating contrast)")
    common.add_argument("--sep-scope", default="task_subgraph",
                        help="graph scope for the separability probe/Davis-Kahan (default task_subgraph)")
    common.add_argument("--no-separability", action="store_true",
                        help="skip the probe/transfer/Davis-Kahan suite")
    common.add_argument("--n-perm", type=int, default=2000,
                        help="permutations for cluster/probe/Davis-Kahan nulls")
    common.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for CIs")

    r = sub.add_parser("run", parents=[common], help="run the spectral sweep")
    r.add_argument("--models", default=str(_ROOT / "configs" / "models.json"))
    r.add_argument("--model", action="append", help="restrict to this model name (repeatable)")
    r.add_argument("--tasks", default=None, help="path to tasks.jsonl")
    r.add_argument("--conditions", default=None, help="comma-separated subset of conditions")
    r.add_argument("--max-tasks", type=int, default=None, help="use only the first N tasks")
    r.add_argument("--runs", type=int, default=1)
    r.add_argument("--device", default="cuda")
    r.add_argument("--cells", choices=list(CELL_PRESETS), default="all",
                   help="which (scope,normalization) cells to compute (default all)")
    r.add_argument("--directed", dest="directed", action="store_true", default=True,
                   help="compute directed-graph asymmetry metrics (default on)")
    r.add_argument("--no-directed", dest="directed", action="store_false",
                   help="disable directed-graph metrics")
    r.add_argument("--generate", action="store_true", help="also run the behavioral generation stage")
    r.add_argument("--max-new-tokens", type=int, default=160)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument(
        "--stop-repetition",
        action="store_true",
        help="opt in to online repetition stopping; changes generated output",
    )
    r.add_argument("--allow-download", action="store_true", help="permit HF downloads (default: cache only)")
    r.add_argument("--analyze", action="store_true", help="run analysis+plots after the sweep")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("analyze", parents=[common], help="analyze an existing long CSV")
    a.add_argument("--long", required=True, help="path to diagnostics_long_all.csv")
    a.add_argument("--fiedler", default=None,
                   help="path to fiedler_vectors_all.pkl (for Davis-Kahan); "
                        "defaults to <out>/fiedler_vectors_all.pkl")
    a.set_defaults(func=cmd_analyze)

    return p


def main(argv=None):
    # Windows consoles default to cp1252; make sure diagnostic prints never crash
    # the run on a stray non-ASCII character.
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
