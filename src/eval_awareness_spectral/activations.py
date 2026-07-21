"""Extract raw residual-stream activations over the task-token span.

This is the substrate for the *activation-probe* baselines and the *causal*
steering experiment. We reuse the same instrumenter the spectral pipeline uses
(eager attention, chat template) and the same task-token span, so the activation
baselines and the spectral probe see the identical token set — an apples-to-apples
comparison, not a capacity giveaway.

Per prompt, per layer ℓ (0 = embeddings … L = final), over the task-token span
S_ℓ ∈ R^{n_tok × d} we save:
  * mean_vec[ℓ]  = mean over task tokens  ∈ R^d   (for PCA/full probes + steering)
  * moments[ℓ]   = [‖mean‖₂, mean token-variance, participation ratio]
                   — graph-FREE low-dim geometric summary; the decisive control
                     that isolates "does the GSP graph add anything" from "does any
                     cheap activation summary work".

Saved as an .npz: mean_vec [N,L,d], moments [N,L,3], and parallel metadata arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from spectral_trust import GSPDiagnosticsFramework

from .prompts import Condition, Task, build_prompts, load_framings, load_tasks
from .runner import RunConfig, _free_cuda, _gsp_config
from .spans import locate_task_span, reconcile_task_spans, SpanError

logger = logging.getLogger(__name__)


def _participation_ratio(M: np.ndarray) -> float:
    """Effective dimensionality of the token cloud M [n_tok, d] via singular values.

    PR = (Σσ²)² / Σσ⁴ ∈ [1, min(n_tok,d)] — high when tokens span many directions.
    """
    if M.shape[0] < 2:
        return 1.0
    Mc = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Mc, compute_uv=False)
    s2 = s ** 2
    denom = np.sum(s2 ** 2)
    if denom <= 0:
        return 1.0
    return float((np.sum(s2) ** 2) / denom)


def extract_activations(rc: RunConfig, out_npz: Optional[str] = None) -> Dict:
    """One forward pass per prompt; pool the task-token residual stream per layer."""
    tasks: List[Task] = load_tasks(rc.tasks_path)
    conditions: Dict[str, Condition] = load_framings(rc.framings_path)
    model_out_dir = Path(rc.output_dir) / rc.name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _gsp_config(rc, model_out_dir)

    mean_vecs: List[np.ndarray] = []
    moments: List[np.ndarray] = []
    meta_task, meta_cond, meta_start = [], [], []

    with GSPDiagnosticsFramework(cfg) as framework:
        logger.info("Loading %s for activation extraction", rc.name)
        framework.instrumenter.load_model(rc.hf_id)
        tokenizer = framework.instrumenter.tokenizer
        prompts = build_prompts(tasks, conditions, tokenizer, only_conditions=rc.conditions,
                                position_match=rc.position_match)

        # spans: keep the majority tokenization per task, null outliers (same policy
        # as the runner) so activation pooling uses the identical task-token set.
        spans, by_task = {}, {}
        for p in prompts:
            try:
                s = locate_task_span(tokenizer, p.text, p.task_text, rc.max_length)
                if s.truncated:
                    s = None
            except SpanError:
                s = None
            by_task.setdefault(p.task_id, {})[p.condition] = s
        for task_id, cs in by_task.items():
            for cond, v in reconcile_task_spans(cs).items():
                spans[(task_id, cond)] = v

        for p in tqdm(prompts, desc=f"{rc.name} activations", leave=False):
            if spans.get((p.task_id, p.condition)) is None:
                continue
            span = spans[(p.task_id, p.condition)]
            try:
                mo = framework.instrumenter.process_text(p.text)
            except Exception as e:
                logger.error("process_text failed (%s/%s): %s", p.task_id, p.condition, e)
                continue
            idx = span.indices
            hs = mo["hidden_states"]  # list length L+1 of [1, seq, d] float32 cpu
            per_layer_mean, per_layer_mom = [], []
            for h in hs:
                S = h.squeeze(0)[idx].detach().cpu().numpy().astype(np.float32)  # [n_tok, d]
                mean_v = S.mean(axis=0)
                per_layer_mean.append(mean_v)
                per_layer_mom.append([
                    float(np.linalg.norm(mean_v)),
                    float(np.mean(np.var(S, axis=0))),
                    _participation_ratio(S),
                ])
            mean_vecs.append(np.stack(per_layer_mean))     # [L+1, d]
            moments.append(np.asarray(per_layer_mom))       # [L+1, 3]
            meta_task.append(p.task_id)
            meta_cond.append(p.condition)
            meta_start.append(span.start)
            del mo

    out = {
        "mean_vec": np.stack(mean_vecs) if mean_vecs else np.zeros((0, 0, 0)),
        "moments": np.stack(moments) if moments else np.zeros((0, 0, 3)),
        "task_id": np.array(meta_task),
        "condition": np.array(meta_cond),
        "task_start": np.array(meta_start),
        "model": rc.name,
    }
    if out_npz:
        Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, **out)
        logger.info("Saved activations -> %s  mean_vec=%s moments=%s",
                    out_npz, out["mean_vec"].shape, out["moments"].shape)
    _free_cuda()
    return out
