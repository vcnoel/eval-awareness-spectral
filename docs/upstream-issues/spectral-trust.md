# Generation outputs lack explicit termination, cap, and repetition provenance

## Source evidence

The CLI helper generates up to 20 tokens with greedy `model.generate`, sets the padding token to tokenizer EOS, decodes the new tokens, and returns only text:

- [`generate_response`, fixed cap, and decode](https://github.com/vcnoel/spectral-trust/blob/f80028c4f39fcc0a54f4976897c1cca76a2578dc/src/spectral_trust/cli/__init__.py#L294-L312)
- [optional CLI response emission](https://github.com/vcnoel/spectral-trust/blob/f80028c4f39fcc0a54f4976897c1cca76a2578dc/src/spectral_trust/cli/__init__.py#L366-L402)

The returned artifact does not say whether generation ended on EOS, reached `max_new_tokens`, produced an empty completion, or entered a repetition loop.

## Scope of the defect

This is not evidence that `model.generate` ignores a model's multi-EOS configuration. Hugging Face generation normally inherits terminal IDs from the model generation configuration, and this call does not replace them with a custom scalar stop rule. The direct Qwen custom-loop defect found elsewhere—checking only scalar tokenizer EOS—is therefore **not established in this repository**.

The broader reproducibility gap is that the output drops termination and validity provenance after generation. A shared contract should preserve:

- every terminal ID declared by the model generation configuration, with tokenizer EOS only as a compatible fallback;
- termination reason and terminal token/index;
- generated-token count and cap-hit status;
- conservative repetition detection;
- completion, degeneracy, and scorer-eligibility flags;
- raw token IDs or a checksum-addressed private reference when text cannot be published.

Online repetition stopping should be opt-in because it changes output. Post-generation classification can be added without changing decoding behavior.

## Historical impact

Prompt-only graph and spectral analyses do not consume generated responses and are unaffected by this missing metadata. Any downstream result that used the optional generated answer for correctness, diagnosis, or response-conditioned analysis may have treated capped or repetitive text as an ordinary completion because the surviving output has no termination reason.

Existing generation-based claims should receive a cap-hit and repetition audit before reuse. This issue does **not** establish that any published result is wrong.
