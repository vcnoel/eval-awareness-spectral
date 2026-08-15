"""Phase 2 (optional): let the model actually answer, then analyze prompt+response.

Prompt-only analysis measures how the model *represents* an eval-framed context.
This stage additionally captures how the model *behaves*: we generate a response
under each frame and run the spectral diagnostics over the full prompt+response
sequence. Sampling adds variance, so use runs>1 here.

Behavioral by-products (response text, length, refusal heuristic) are logged so a
downstream grader can check for sandbagging / eval-gaming.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from spectral_trust import GSPDiagnosticsFramework

from .generation_contract import (
    SingleSequenceRepetitionStoppingCriteria,
    assess_generation,
)
from .prompts import Condition, Task, build_prompts, load_framings, load_tasks
from .runner import ST_VERSION, RunConfig, _free_cuda, _gsp_config

# The behavioral stage analyzes the full prompt+response on the *full* graph via
# the library's to_dict(), so gini/lambda2/lambda3 (which we recompute only in the
# prompt-only runner) are absent here — this stage is about response divergence.
_GEN_METRICS = ["energy", "smoothness_index", "spectral_entropy", "hfer",
                "fiedler_value", "connectivity"]

logger = logging.getLogger(__name__)

_REFUSAL_MARKERS = [
    "i can't", "i cannot", "i'm not able", "i am not able", "i won't",
    "i'm sorry", "i am sorry", "as an ai", "i'm unable", "i am unable",
]


def _looks_like_refusal(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _REFUSAL_MARKERS[:200])


@torch.no_grad()
def _generate(
    framework,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    *,
    stop_repetition: bool = False,
) -> tuple[str, dict]:
    """Generate with unchanged legacy sampling and attach validity provenance.

    Online repetition stopping is opt-in because it changes the generated sequence.
    Post-hoc classification is always active and never changes decoding.
    """
    tok = framework.instrumenter.tokenizer
    model = framework.instrumenter.model
    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    do_sample = temperature and temperature > 0
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "pad_token_id": tok.pad_token_id or tok.eos_token_id,
    }
    stopper = None
    if stop_repetition:
        from transformers import StoppingCriteriaList

        stopper = SingleSequenceRepetitionStoppingCriteria(
            prompt_length=int(inputs["input_ids"].shape[1])
        )
        kwargs["stopping_criteria"] = StoppingCriteriaList([stopper])
    gen = model.generate(**inputs, **kwargs)
    new_tokens = gen[0][inputs["input_ids"].shape[1]:]
    token_ids = [int(value) for value in new_tokens.tolist()]
    assessment = assess_generation(
        token_ids,
        generation_config=getattr(model, "generation_config", None),
        tokenizer=tok,
        generated_cap=max_new_tokens,
        loop=stopper.result if stopper is not None else None,
        mode="online" if stop_repetition else "post_hoc",
    )
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    return text, assessment.provenance_dict()


def run_model_with_generation(
    rc: RunConfig,
    max_new_tokens: int = 160,
    temperature: float = 0.7,
    *,
    stop_repetition: bool = False,
) -> pd.DataFrame:
    """For each (task, condition, run): generate a response, then analyze
    prompt+response spectrally. Returns long-format diagnostics plus the
    behavioral columns (response text/length/refusal)."""
    tasks: List[Task] = load_tasks(rc.tasks_path)
    conditions: Dict[str, Condition] = load_framings(rc.framings_path)

    model_out_dir = Path(rc.output_dir) / rc.name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _gsp_config(rc, model_out_dir)

    rows: List[Dict] = []
    behavior: List[Dict] = []

    with GSPDiagnosticsFramework(cfg) as framework:
        logger.info("Loading %s for generation stage", rc.name)
        framework.instrumenter.load_model(rc.hf_id)
        tokenizer = framework.instrumenter.tokenizer
        prompts = build_prompts(tasks, conditions, tokenizer, only_conditions=rc.conditions)

        for run_idx in range(rc.runs):
            for p in tqdm(prompts, desc=f"{rc.name} gen run {run_idx}", leave=False):
                try:
                    response, termination = _generate(
                        framework,
                        p.text,
                        max_new_tokens,
                        temperature,
                        stop_repetition=stop_repetition,
                    )
                except Exception as e:
                    logger.error("generate failed (%s/%s/%s): %s",
                                 rc.name, p.task_id, p.condition, e)
                    continue

                full_text = p.text + response
                try:
                    result = framework.analyze_text(full_text, save_results=False)
                except Exception as e:
                    logger.error("analyze_text failed (%s/%s/%s): %s",
                                 rc.name, p.task_id, p.condition, e)
                    continue

                behavior.append({
                    "model": rc.name, "task_id": p.task_id, "category": p.category,
                    "condition": p.condition, "run": run_idx,
                    "response": response,
                    "response_chars": len(response),
                    "response_words": len(response.split()),
                    "refusal_heuristic": _looks_like_refusal(response),
                    **termination,
                })

                n_tokens = len(result["tokens"])
                for diag in result["layer_diagnostics"]:
                    d = diag.to_dict()
                    row = {
                        "model": rc.name, "hf_id": rc.hf_id, "task_id": p.task_id,
                        "category": p.category, "condition": p.condition, "run": run_idx,
                        "layer": int(d["layer"]), "n_tokens": n_tokens, "stage": "generation",
                        "st_version": ST_VERSION, "graph_scope": "full", "normalization": "sym",
                    }
                    for m in _GEN_METRICS:
                        row[m] = d.get(m)
                    rows.append(row)

    df = pd.DataFrame(rows)
    bdf = pd.DataFrame(behavior)
    df.to_csv(model_out_dir / "diagnostics_long_generation.csv", index=False)
    bdf.to_csv(model_out_dir / "responses.csv", index=False)
    logger.info("Generation stage: %d diag rows, %d responses -> %s",
                len(df), len(bdf), model_out_dir)
    _free_cuda()
    return df
