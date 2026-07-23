# Fresh GenerationConfig omits explicit EOS and completion validity in Qwen generation

## Source evidence

The shared generator constructs a new `GenerationConfig` with `max_new_tokens` and deterministic decoding, then sets only `pad_token_id` before passing that fresh object to `model.generate`:

- [`GenerationConfig` construction and generation](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/pipeline/model_utils/model_base.py#L67-L94)
- [pipeline-wide 512-token cap](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/pipeline/config.py#L7-L18)
- [cap passed into completion generation and decoded responses saved](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/pipeline/run_pipeline.py#L97-L108)

The Qwen wrapper assigns tokenizer padding to `tokenizer.eod_id`:

- [Qwen tokenizer padding setup](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/pipeline/model_utils/qwen_model.py#L118-L134)

The fresh generation configuration does not explicitly copy `eos_token_id` from the model or tokenizer and the saved completion contains only category, prompt, and decoded response. It records no terminal reason, cap hit, repetition evidence, or validity flag.

## Requested audit and fix

Please confirm how the pinned Transformers version resolves EOS when a fresh `GenerationConfig` with `eos_token_id=None` is supplied. For Qwen, explicitly preserve every model-declared terminal ID rather than inferring completion from the padding ID. Add post-generation termination and repetition provenance plus a scorer-eligibility flag.

This is separate from the Qwen3 custom-loop defect found in another project: that loop checked only scalar tokenizer EOS. Here the concern is a fresh generation configuration that may omit EOS entirely, plus the broader absence of completion-validity metadata.

## Historical impact

The repository includes precomputed Qwen-1.8B-chat completion and evaluation artifacts, so Qwen-specific generation outputs used in direction selection or evaluation should be checked for cap saturation and repetition before reuse. If EOS was absent, a Qwen completion may have run to the fixed cap and then been decoded and scored as an ordinary response.

Other model families are not implicated without the same configuration audit. Non-generation direction extraction is also outside this observation. This is a reproducibility and artifact-audit request, not a claim that the published refusal-direction result is invalid.
