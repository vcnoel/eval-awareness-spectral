"""Test Theorem 2 / prediction P4: the graph-frequency spectrum of the frame shift.

Theorem 2 says Dirichlet-energy metrics suppress a shift in proportion to its graph
frequency, and mean-pool reads only the DC mode. The sharp prediction P4: a frame is a
global, slowly-varying influence, so the *pure-confound* shift (deploy→placebo) should
sit at LOWER graph frequency than the *evaluativeness* shift (deploy→eval).

For each task we run the (position-matched) prompt through the library's analyze_text
with subgraph_indices (which returns the task-subgraph Laplacian eigenbasis Φ, Λ per
layer) and pull the task-token signal X from the hidden states. Using the DEPLOY graph's
eigenbasis as the common reference, the shift Δ = X_a − X_b is graph-Fourier transformed
and summarized by:

  centroid c(Δ) = Σ_k λ_k ‖Δ̂_k‖² / Σ_k ‖Δ̂_k‖²     (mean graph frequency of the shift)
  DC fraction   f_DC(Δ) = ‖Δ̂_0‖² / Σ_k ‖Δ̂_k‖²

P4:  c(eval·deploy) > c(deploy·placebo)   and   f_DC(eval·deploy) < f_DC(deploy·placebo).
Paired across tasks (Wilcoxon). If it fails, Theorem 2's mechanism is wrong for this data.

    python -m eval_awareness_spectral.freq_decomp --model llama-3.2-3b-it --tasks 15
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from spectral_trust import GSPDiagnosticsFramework

from .prompts import build_prompts, load_framings, load_tasks
from .runner import RunConfig, _free_cuda, _gsp_config
from .spans import locate_task_span, reconcile_task_spans, SpanError

logger = logging.getLogger(__name__)


def _shift_spectrum(X_a, X_b, Phi, lam):
    """Graph-Fourier centroid and DC-fraction of the shift (X_a − X_b) on basis Phi."""
    D = (X_a - X_b)                      # [n, d]
    Dhat = Phi.T @ D                     # [k, d]
    e = np.sum(Dhat ** 2, axis=1)        # [k] energy per graph frequency
    tot = e.sum()
    if tot <= 0:
        return np.nan, np.nan
    centroid = float(np.sum(lam * e) / tot)
    f_dc = float(e[0] / tot)
    return centroid, f_dc


def run(rc: RunConfig, tasks_n: int = 15, layer_lo_frac: float = 0.4) -> pd.DataFrame:
    tasks = load_tasks(rc.tasks_path)[:tasks_n]
    framings = load_framings(rc.framings_path)
    conds = ["deploy", "eval", "placebo"]
    cfg = _gsp_config(rc, Path(rc.output_dir) / rc.name)
    # A valid graph-Fourier transform needs an ORTHONORMAL eigenbasis, so use the
    # symmetric normalized Laplacian (rw's eigenvectors are non-orthonormal and would
    # make Phi^T Δ / the per-frequency energies meaningless).
    cfg.normalization = "sym"

    rows = []
    with GSPDiagnosticsFramework(cfg) as fw:
        fw.instrumenter.load_model(rc.hf_id)
        tok = fw.instrumenter.tokenizer
        prompts = build_prompts(tasks, framings, tok, only_conditions=conds,
                                position_match=rc.position_match)

        # spans, reconciled per task
        by_task: Dict[str, Dict[str, object]] = {}
        for p in prompts:
            try:
                s = locate_task_span(tok, p.text, p.task_text, rc.max_length)
                s = None if s.truncated else s
            except SpanError:
                s = None
            by_task.setdefault(p.task_id, {})[p.condition] = s
        spans = {}
        for t, cs in by_task.items():
            for c, v in reconcile_task_spans(cs).items():
                spans[(t, c)] = v

        pmap = {(p.task_id, p.condition): p for p in prompts}
        for task in tasks:
            if any(spans.get((task.id, c)) is None for c in conds):
                continue
            # analyze each condition; store per-layer signal X and (deploy) eigenbasis
            X = {c: {} for c in conds}
            basis = {}  # layer -> (Phi, lam) from deploy
            ok = True
            for c in conds:
                sp = spans[(task.id, c)]
                res = fw.analyze_text(pmap[(task.id, c)].text, save_results=False,
                                      subgraph_indices=sp.indices)
                hs = res["model_outputs"]["hidden_states"]
                for diag in res["layer_diagnostics"]:
                    L = int(diag.layer)
                    sig = hs[L + 1].squeeze(0)[sp.indices].detach().cpu().numpy().astype(np.float64)
                    X[c][L] = sig
                    if c == "deploy":
                        Phi = np.asarray(diag.eigenvectors, dtype=np.float64)
                        lam = np.asarray(diag.eigenvalues, dtype=np.float64)
                        if Phi.ndim != 2 or Phi.shape[0] != sig.shape[0]:
                            ok = False
                            break
                        basis[L] = (Phi, lam)
                if not ok:
                    break
            if not ok:
                continue

            layers = sorted(basis.keys())
            lo = int(len(layers) * layer_lo_frac)  # late-layer band (signal lives late)
            for L in layers[lo:]:
                Phi, lam = basis[L]
                c_ed, f_ed = _shift_spectrum(X["eval"][L], X["deploy"][L], Phi, lam)
                c_dp, f_dp = _shift_spectrum(X["deploy"][L], X["placebo"][L], Phi, lam)
                rows.append({"task": task.id, "layer": L,
                             "centroid_eval": c_ed, "centroid_neg": c_dp,
                             "fdc_eval": f_ed, "fdc_neg": f_dp})
    _free_cuda()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Paired per-task test of P4 (average over the late-layer band first)."""
    if df.empty:
        return pd.DataFrame()
    per_task = df.groupby("task")[["centroid_eval", "centroid_neg", "fdc_eval", "fdc_neg"]].mean()
    out = {}
    for name, a, b, direction in [
        ("centroid (want eval>neg)", "centroid_eval", "centroid_neg", "greater"),
        ("f_DC (want eval<neg)", "fdc_eval", "fdc_neg", "less"),
    ]:
        da, db = per_task[a].to_numpy(), per_task[b].to_numpy()
        diff = da - db
        try:
            _, p = stats.wilcoxon(da, db, alternative=direction)
        except ValueError:
            p = np.nan
        out[name] = {"mean_eval": float(da.mean()), "mean_neg": float(db.mean()),
                     "mean_diff": float(diff.mean()), "n_tasks": len(diff),
                     "frac_supporting": float(np.mean(diff > 0 if direction == "greater" else diff < 0)),
                     "wilcoxon_p": float(p) if p is not None else np.nan}
    return pd.DataFrame(out).T


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-3b-it")
    ap.add_argument("--hf-id", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tasks", type=int, default=15)
    ap.add_argument("--out", default="results/freq_decomp")
    args = ap.parse_args()

    rc = RunConfig(name=args.model, hf_id=args.hf_id, dtype=args.dtype, device="cuda",
                   cells=[("task_subgraph", "rw")], directed=False, position_match=True,
                   output_dir=args.out)
    df = run(rc, tasks_n=args.tasks)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out) / f"freq_shift_{args.model}.csv", index=False)
    summ = summarize(df)
    if not summ.empty:
        summ.to_csv(Path(args.out) / f"freq_summary_{args.model}.csv")
        pd.set_option("display.width", 200)
        print(f"\n===== P4: graph-frequency of the shift ({args.model}, {args.tasks} tasks) =====")
        print("Theorem 2 predicts: evaluativeness shift is HIGHER graph-frequency than the confound")
        print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
