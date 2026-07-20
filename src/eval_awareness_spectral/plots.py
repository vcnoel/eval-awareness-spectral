"""Figures for the v2 eval-awareness instrument.

Everything is faceted by *cell* = (graph_scope, normalization); filenames carry the
cell so the primary ``task_subgraph/rw`` result is never confused with the ``full``
reference or the ``task_subgraph/none`` mass-leak demonstrator.

Figure families:
  * condition curves        — metric vs layer, one line per frame (±SEM)
  * effect heatmap          — Cohen's d_z over (metric × layer), FDR-starred
  * velocity curves         — cross-layer derivative, cluster-significant layers shaded
  * Davis-Kahan rotation    — 1-|cosθ| of the Fiedler vector vs layer, gap-gated
  * probe AUC bars          — separability AUC with bootstrap CI + permutation null
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import DEFAULT_METRICS, PRIMARY_METRICS

_CONDITION_COLORS = {
    "control": "#888888", "deploy": "#2c7fb8", "eval": "#d95f0e",
    "eval_subtle": "#e6a817", "placebo": "#31a354",
}


def _cell_tag(scope: str, norm: str) -> str:
    return f"{scope}-{norm}"


def _cells(df: pd.DataFrame):
    if "graph_scope" not in df.columns:
        return [("full", "rw")]
    return list(df[["graph_scope", "normalization"]].drop_duplicates()
                .itertuples(index=False, name=None))


def plot_condition_curves(means: pd.DataFrame, out_dir, metrics=None) -> List[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for scope, norm in _cells(means):
        cm = means[(means["graph_scope"] == scope) & (means["normalization"] == norm)]
        mets = metrics or [m for m in DEFAULT_METRICS if m in cm["metric"].unique()]
        for model, mdf in cm.groupby("model"):
            n = len(mets); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)
            for i, metric in enumerate(mets):
                ax = axes[i // ncol][i % ncol]
                sub = mdf[mdf["metric"] == metric]
                for cond, cdf in sub.groupby("condition"):
                    cdf = cdf.sort_values("layer"); color = _CONDITION_COLORS.get(cond)
                    ax.plot(cdf["layer"], cdf["mean"], label=cond, color=color, lw=2)
                    ax.fill_between(cdf["layer"], cdf["mean"] - cdf["sem"],
                                    cdf["mean"] + cdf["sem"], color=color, alpha=0.18)
                ax.set_title(metric + ("" if metric in PRIMARY_METRICS else "  (descriptive)"))
                ax.set_xlabel("layer"); ax.grid(True, alpha=0.3)
            for j in range(n, nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            axes[0][0].legend(fontsize="small")
            fig.suptitle(f"Spectral metrics by frame — {model}  [{_cell_tag(scope, norm)}]", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            p = out_dir / f"curves_{model}_{_cell_tag(scope, norm)}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


def plot_effect_heatmap(effects: pd.DataFrame, out_dir, cond_a="eval", cond_b="deploy") -> List[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for scope, norm in _cells(effects):
        ce = effects[(effects["graph_scope"] == scope) & (effects["normalization"] == norm)]
        for model, mdf in ce.groupby("model"):
            pivot = mdf.pivot_table(index="metric", columns="layer", values="dz")
            sig = mdf.pivot_table(index="metric", columns="layer", values="sig_fdr05")
            absvals = np.abs(pivot.values)
            vmax = float(np.nanpercentile(absvals, 95)) or float(np.nanmax(absvals)) or 1.0
            fig, ax = plt.subplots(figsize=(max(6, 0.32 * pivot.shape[1]), 0.6 * pivot.shape[0] + 1.5))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, fontsize=6)
            ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
            ax.set_xlabel("layer")
            for yi, metric in enumerate(pivot.index):
                for xi, layer in enumerate(pivot.columns):
                    try:
                        if bool(sig.loc[metric, layer]):
                            ax.text(xi, yi, "*", ha="center", va="center", color="black", fontsize=8)
                    except KeyError:
                        pass
            fig.colorbar(im, ax=ax, label=f"Cohen's $d_z$  ({cond_a} − {cond_b})")
            ax.set_title(f"Effect map — {model}  [{_cell_tag(scope, norm)}]\n(* FDR<0.05)")
            fig.tight_layout()
            p = out_dir / f"effectmap_{model}_{_cell_tag(scope, norm)}_{cond_a}_vs_{cond_b}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


def plot_velocity(means: pd.DataFrame, vel_clusters: pd.DataFrame, out_dir,
                  cond_a="eval", cond_b="deploy") -> List[Path]:
    """Velocity (cross-layer derivative) of the mean metric curve, per condition,
    with cluster-significant layer ranges shaded (from the permutation test)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for scope, norm in _cells(means):
        cm = means[(means["graph_scope"] == scope) & (means["normalization"] == norm)]
        mets = [m for m in PRIMARY_METRICS if m in cm["metric"].unique()]
        if not mets:
            continue
        for model, mdf in cm.groupby("model"):
            n = len(mets); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow), squeeze=False)
            for i, metric in enumerate(mets):
                ax = axes[i // ncol][i % ncol]
                sub = mdf[mdf["metric"] == metric]
                for cond, cdf in sub.groupby("condition"):
                    if cond not in (cond_a, cond_b):
                        continue
                    cdf = cdf.sort_values("layer")
                    vel = np.diff(cdf["mean"].to_numpy())
                    ax.plot(cdf["layer"].to_numpy()[1:], vel, label=cond,
                            color=_CONDITION_COLORS.get(cond), lw=2)
                if not vel_clusters.empty:
                    vc = vel_clusters[(vel_clusters["model"] == model)
                                      & (vel_clusters["graph_scope"] == scope)
                                      & (vel_clusters["normalization"] == norm)
                                      & (vel_clusters["metric"] == metric)]
                    for _, row in vc.iterrows():
                        for rng in str(row.get("sig_layer_ranges", "")).split(";"):
                            rng = rng.strip()
                            if "-" in rng:
                                lo, hi = rng.split("-")
                                ax.axvspan(int(lo) - 0.5, int(hi) + 0.5, color="red", alpha=0.10)
                ax.axhline(0, color="k", lw=0.6)
                ax.set_title(metric); ax.set_xlabel("layer"); ax.set_ylabel("Δ/layer")
                ax.grid(True, alpha=0.3)
            for j in range(n, nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            axes[0][0].legend(fontsize="small")
            fig.suptitle(f"Spectral velocity — {model}  [{_cell_tag(scope, norm)}]  "
                         f"(shaded = cluster-sig {cond_a} vs {cond_b})", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            p = out_dir / f"velocity_{model}_{_cell_tag(scope, norm)}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


def plot_davis_kahan(dk: pd.DataFrame, out_dir, cond_a="eval", cond_b="deploy") -> List[Path]:
    if dk is None or dk.empty:
        return []
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for model, g in dk.groupby("model"):
        g = g.sort_values("layer")
        fig, ax = plt.subplots(figsize=(7, 4))
        interp = g["gap_interpretable"] if "gap_interpretable" in g else pd.Series(True, index=g.index)
        ax.plot(g["layer"], g["mean_rotation"], "-o", color="#762a83", label="matched rotation")
        ax.fill_between(g["layer"], g["rotation_ci_lo"], g["rotation_ci_hi"],
                        color="#762a83", alpha=0.15)
        ax.plot(g["layer"], g["null_mean_rotation"], "--", color="grey", label="mismatched null")
        # shade non-interpretable (small-gap) layers
        for _, row in g[~interp.astype(bool)].iterrows():
            ax.axvspan(row["layer"] - 0.5, row["layer"] + 0.5, color="gray", alpha=0.12)
        ax.set_xlabel("layer"); ax.set_ylabel(r"$1-|\cos\theta|$ (Fiedler rotation)")
        ax.set_title(f"Davis-Kahan Fiedler rotation — {model} ({cond_a} vs {cond_b})\n"
                     f"gray = eigengap too small → non-interpretable")
        ax.grid(True, alpha=0.3); ax.legend(fontsize="small")
        fig.tight_layout()
        p = out_dir / f"davis_kahan_{model}_{cond_a}_vs_{cond_b}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


def plot_probe_bars(probe: pd.DataFrame, transfer: pd.DataFrame, out_dir) -> Optional[Path]:
    if (probe is None or probe.empty) and (transfer is None or transfer.empty):
        return None
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * (len(probe) + len(transfer))), 4.5))
    labels, aucs, los, his, colors = [], [], [], [], []
    if probe is not None and not probe.empty:
        for _, r in probe.iterrows():
            labels.append(f"{r['model']}\n{r['contrast']}\np={r['perm_p']:.3f}")
            aucs.append(r["auc"]); los.append(r["auc"] - r["auc_ci_lo"])
            his.append(r["auc_ci_hi"] - r["auc"]); colors.append("#2166ac")
    if transfer is not None and not transfer.empty:
        for _, r in transfer.iterrows():
            labels.append(f"{r['model']}\ntransfer\n{r['train']}→\n{r['test']}")
            aucs.append(r["auc"]); los.append(0); his.append(0); colors.append("#b2182b")
    x = np.arange(len(labels))
    ax.bar(x, aucs, yerr=[los, his], color=colors, alpha=0.85, capsize=4)
    ax.axhline(0.5, color="k", lw=1, ls="--", label="chance")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("leave-task-out AUC"); ax.set_ylim(0, 1.0)
    ax.set_title("Eval-vs-deploy separability (blue) and subtle↔explicit transfer (red)")
    ax.legend(fontsize="small"); fig.tight_layout()
    p = out_dir / "probe_separability.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def make_all_figures(effects, summary, means, out_dir, cond_a="eval", cond_b="deploy",
                     velocity_clusters=None, separability: Optional[Dict] = None) -> List[Path]:
    paths = []
    if means is not None and not means.empty:
        paths += plot_condition_curves(means, out_dir)
        if velocity_clusters is not None:
            paths += plot_velocity(means, velocity_clusters, out_dir, cond_a, cond_b)
    if effects is not None and not effects.empty:
        paths += plot_effect_heatmap(effects, out_dir, cond_a=cond_a, cond_b=cond_b)
    if separability:
        paths += plot_davis_kahan(separability.get("davis_kahan"), out_dir, cond_a, cond_b)
        bar = plot_probe_bars(separability.get("probe"), separability.get("transfer"), out_dir)
        if bar:
            paths.append(bar)
    return paths
