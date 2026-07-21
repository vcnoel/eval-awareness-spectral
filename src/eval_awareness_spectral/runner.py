"""Drive the spectral_trust diagnostics across (model, task, condition, run).

We *use* spectral-trust as the instrument — we do not re-derive its graph pipeline.
Every measurement comes from ``framework.analyze_text(...)``; we only read out the
per-layer ``SpectralDiagnostics`` it returns.

A prompt is analyzed once per **cell** = (graph_scope, normalization):

  * ``(full, rw)``          — whole prompt; length is an observable.
  * ``(task_subgraph, rw)`` — PRIMARY. Pass ``subgraph_indices`` = the byte-identical
                              task tokens so the graph is built on a fixed node set.
                              Under ``rw`` the Laplacian is ``I - D⁻¹W`` with the
                              *subgraph* degree, so it is intrinsically row-normalized
                              (the softmax mass-leak to frame tokens is divided out).
  * ``(task_subgraph, none)`` — DEMONSTRATOR. Combinatorial ``L = D - W`` on the same
                              nodes; the length/mass artifact is *visible* here, which
                              is the point (it shows what the ``rw`` cell controls for).

Selecting a cell = setting ``framework.config.normalization`` and passing (or not)
``subgraph_indices`` to ``analyze_text``. Each cell re-runs the forward pass; that is
the library's public contract and is cheap for the small models we use.

Metrics come straight from ``diag.to_dict()`` (energy, smoothness_index,
spectral_entropy, hfer, fiedler_value, connectivity, and — when ``directed`` —
spectral_radius, max_imaginary). We add two *read-outs* of the library's own output:
``gini_sparsity`` (Gini of the returned ``spectral_masses``) and ``lambda2``/``lambda3``
(from the returned ``eigenvalues``, whose gap gates the Davis-Kahan analysis). For the
subgraph cells we also keep the Fiedler eigenvector (``diag.eigenvectors[:,1]``) so eval
and deploy can be compared on the identical task-node set.
"""

from __future__ import annotations

import gc
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import spectral_trust
from spectral_trust import GSPConfig, GSPDiagnosticsFramework

from .prompts import Condition, Task, build_prompts, load_framings, load_tasks
from .spans import TaskSpan, locate_task_span, reconcile_task_spans, SpanError

logger = logging.getLogger(__name__)

ST_VERSION = getattr(spectral_trust, "__version__", "0.2.2")

# Scalars kept per (cell, layer). Directed metrics are None unless directed=True.
SCALAR_METRICS = [
    "energy",
    "smoothness_index",
    "spectral_entropy",
    "hfer",
    "fiedler_value",
    "connectivity",
    "gini_sparsity",     # read-out: Gini of the library's spectral_masses
    "spectral_radius",   # directed only
    "max_imaginary",     # directed only
    "lambda2",           # λ₂ from returned eigenvalues (= fiedler_value)
    "lambda3",           # λ₃ — Davis-Kahan eigengap is λ₃−λ₂
]

# A "cell" is (graph_scope, normalization). Default set for v2 sweeps.
DEFAULT_CELLS: List[Tuple[str, str]] = [
    ("full", "rw"),
    ("task_subgraph", "rw"),    # PRIMARY
    ("task_subgraph", "none"),  # mass-leak / length demonstrator
]


@dataclass
class RunConfig:
    name: str
    hf_id: str
    dtype: str = "float32"
    device: str = "cuda"
    conditions: Optional[List[str]] = None
    runs: int = 1
    max_length: int = 512
    num_eigenvalues: int = 50
    directed: bool = True
    cells: List[Tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_CELLS))
    position_match: bool = False  # pad preambles so the task starts at the same token index
    output_dir: str = "./results"
    tasks_path: Optional[str] = None
    framings_path: Optional[str] = None
    local_files_only: bool = True
    model_kwargs: Dict = field(default_factory=dict)


def _gsp_config(rc: RunConfig, model_out_dir: Path) -> GSPConfig:
    return GSPConfig(
        model_name=rc.hf_id,
        device=rc.device,
        torch_dtype=rc.dtype,
        max_length=rc.max_length,
        num_eigenvalues=rc.num_eigenvalues,
        # Force the DENSE eigensolver. The default "sparse" (ARPACK) returns only
        # num_eigenvalues eigenvectors, but spectral_entropy / hfer / energy are
        # computed from the FULL eigenbasis — on the `full` scope (up to max_length
        # tokens ≫ 50) the sparse path silently truncates the basis and corrupts
        # those metrics. (Confirmed by the agent-safety spectral baseline, which
        # forces dense for exactly this reason.)
        eigen_solver="dense",
        normalization="rw",       # per-cell overridden in the loop
        directed=rc.directed,     # library adds spectral_radius/max_imaginary
        local_files_only=rc.local_files_only,
        output_dir=str(model_out_dir),
        save_plots=False,
        display_plots=False,
        save_intermediate=True,   # so analyze_text returns eigenvectors (Fiedler)
        verbose=False,
        model_kwargs=rc.model_kwargs,
    )


def _gini(x) -> float:
    """Gini coefficient of a non-negative 1-D array (0 = uniform, →1 = concentrated)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = np.clip(x[np.isfinite(x)], 0.0, None)
    n = x.size
    if n == 0:
        return float("nan")
    s = x.sum()
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * xs)) / (n * s) - (n + 1.0) / n)


def analyze_prompt(
    framework,
    text: str,
    span: Optional[TaskSpan],
    cells: List[Tuple[str, str]],
) -> Tuple[List[Dict], List[Dict]]:
    """Analyze one prompt across all cells via ``framework.analyze_text``.

    Returns (scalar_rows, fiedler_records).
    """
    cfg = framework.config
    orig_norm = cfg.normalization
    scalar_rows: List[Dict] = []
    fiedler_records: List[Dict] = []
    idx = span.indices if span is not None else None

    try:
        for scope, norm in cells:
            is_sub = scope != "full"
            if is_sub and idx is None:
                continue
            cfg.normalization = norm
            result = framework.analyze_text(
                text, save_results=False,
                subgraph_indices=(idx if is_sub else None),
            )
            n_tokens = len(result["tokens"])
            for diag in result["layer_diagnostics"]:
                d = diag.to_dict()
                eig = np.asarray(getattr(diag, "eigenvalues", []), dtype=np.float64)
                lam2 = float(eig[1]) if eig.size > 1 else float("nan")
                lam3 = float(eig[2]) if eig.size > 2 else float("nan")
                # 0.2.2 computes directed metrics on the FULL adjacency regardless of
                # subgraph_indices, so they only describe the `full` scope. Recording
                # them on subgraph rows would mislabel full-graph asymmetry.
                spec_rad = d.get("spectral_radius") if not is_sub else None
                max_imag = d.get("max_imaginary") if not is_sub else None
                scalar_rows.append({
                    "graph_scope": scope,
                    "normalization": norm,
                    "layer": int(d["layer"]),
                    "n_tokens": n_tokens,
                    "n_nodes": len(idx) if is_sub else n_tokens,
                    "energy": d.get("energy"),
                    "smoothness_index": d.get("smoothness_index"),
                    "spectral_entropy": d.get("spectral_entropy"),
                    "hfer": d.get("hfer"),
                    "fiedler_value": d.get("fiedler_value"),
                    "connectivity": d.get("connectivity"),
                    "gini_sparsity": _gini(getattr(diag, "spectral_masses", [])),
                    "spectral_radius": spec_rad,
                    "max_imaginary": max_imag,
                    "lambda2": lam2,
                    "lambda3": lam3,
                })
                if is_sub:
                    vecs = getattr(diag, "eigenvectors", None)
                    if vecs is not None and np.ndim(vecs) == 2 and vecs.shape[1] > 1:
                        fiedler_records.append({
                            "graph_scope": scope, "normalization": norm,
                            "layer": int(d["layer"]),
                            "fiedler_vec": np.asarray(vecs[:, 1], dtype=np.float64),
                            "lambda2": lam2, "lambda3": lam3,
                        })
    finally:
        cfg.normalization = orig_norm

    return scalar_rows, fiedler_records


def _compute_spans(prompts, tokenizer, max_length: int):
    """Locate task spans; disable subgraph for tasks whose span isn't identical/locatable."""
    spans: Dict[Tuple[str, str], Optional[TaskSpan]] = {}
    by_task: Dict[str, Dict[str, Optional[TaskSpan]]] = {}
    for p in prompts:
        try:
            span = locate_task_span(tokenizer, p.text, p.task_text, max_length)
            if span.truncated:
                logger.warning("task %s/%s: task span truncated by max_length; "
                               "disabling subgraph", p.task_id, p.condition)
                span = None
        except SpanError as e:
            logger.warning("task %s/%s: span location failed (%s)", p.task_id, p.condition, e)
            span = None
        by_task.setdefault(p.task_id, {})[p.condition] = span

    skip = set()
    for task_id, cond_spans in by_task.items():
        # keep the majority tokenization; null out only outlier conditions
        reconciled = reconcile_task_spans(cond_spans)
        for cond, v in reconciled.items():
            spans[(task_id, cond)] = v
        if sum(v is not None for v in reconciled.values()) < 2:
            skip.add(task_id)  # too few comparable frames to pair
    if skip:
        logger.warning("subgraph unusable for %d task(s) (<2 comparable frames): %s",
                       len(skip), sorted(skip))
    return spans, skip


def run_model(rc: RunConfig) -> pd.DataFrame:
    """Run every (task, condition, run, cell, layer) for a single model."""
    tasks: List[Task] = load_tasks(rc.tasks_path)
    conditions: Dict[str, Condition] = load_framings(rc.framings_path)

    model_out_dir = Path(rc.output_dir) / rc.name
    model_out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _gsp_config(rc, model_out_dir)
    rows: List[Dict] = []
    fiedler_all: List[Dict] = []

    with GSPDiagnosticsFramework(cfg) as framework:
        logger.info("Loading %s (%s, %s) — spectral_trust %s",
                    rc.name, rc.hf_id, rc.dtype, ST_VERSION)
        framework.instrumenter.load_model(rc.hf_id)
        tokenizer = framework.instrumenter.tokenizer

        prompts = build_prompts(tasks, conditions, tokenizer, only_conditions=rc.conditions,
                                position_match=rc.position_match)
        spans, skip_subgraph = _compute_spans(prompts, tokenizer, rc.max_length)

        for run_idx in range(rc.runs):
            for p in tqdm(prompts, desc=f"{rc.name} run {run_idx}", leave=False):
                span = None if p.task_id in skip_subgraph else spans[(p.task_id, p.condition)]
                try:
                    scalar_rows, fied = analyze_prompt(framework, p.text, span, rc.cells)
                except Exception as e:
                    logger.error("analyze_prompt failed (%s/%s/%s): %s",
                                 rc.name, p.task_id, p.condition, e)
                    continue

                meta = {
                    "model": rc.name, "hf_id": rc.hf_id, "task_id": p.task_id,
                    "category": p.category, "condition": p.condition, "run": run_idx,
                    "st_version": ST_VERSION, "gini_source": "eval_awareness_spectral",
                }
                for r in scalar_rows:
                    rows.append({**meta, **r})
                for r in fied:
                    fiedler_all.append({
                        "model": rc.name, "task_id": p.task_id, "condition": p.condition,
                        "run": run_idx, **r,
                    })

    df = pd.DataFrame(rows)
    out_csv = model_out_dir / "diagnostics_long.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %d rows -> %s", len(df), out_csv)

    if fiedler_all:
        with open(model_out_dir / "fiedler_vectors.pkl", "wb") as f:
            pickle.dump(fiedler_all, f)
        logger.info("Wrote %d Fiedler records -> %s",
                    len(fiedler_all), model_out_dir / "fiedler_vectors.pkl")

    _free_cuda()
    return df


def run_experiment(run_configs: List[RunConfig]) -> pd.DataFrame:
    """Sweep multiple models sequentially, concatenating results."""
    frames = [run_model(rc) for rc in run_configs]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        out_dir = Path(run_configs[0].output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_dir / "diagnostics_long_all.csv", index=False)
    return combined


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
