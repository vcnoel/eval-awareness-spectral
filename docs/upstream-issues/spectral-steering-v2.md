# Evaluation scripts score generated outputs without a shared completion-validity contract

## Source evidence

Several evaluation paths decode and score generated text at fixed caps without recording why generation stopped:

- **Main steering evaluation:** [30-token sycophancy generation and immediate scoring](https://github.com/vcnoel/spectral-steering-v2/blob/f75f834a1cd244afdd4d81601fde31b25efa9342/scripts/steer.py#L279-L310); [256-token GSM8K generation and answer extraction](https://github.com/vcnoel/spectral-steering-v2/blob/f75f834a1cd244afdd4d81601fde31b25efa9342/scripts/steer.py#L340-L370).
- **Open-ended judge:** [200-token response generation](https://github.com/vcnoel/spectral-steering-v2/blob/f75f834a1cd244afdd4d81601fde31b25efa9342/scripts/run_open_ended_judge.py#L97-L110) feeds decoded text into a [5-token judge generation](https://github.com/vcnoel/spectral-steering-v2/blob/f75f834a1cd244afdd4d81601fde31b25efa9342/scripts/run_open_ended_judge.py#L113-L129), and both outputs are scored without termination metadata.
- **RepE baseline:** [30-token multiple-choice generation](https://github.com/vcnoel/spectral-steering-v2/blob/f75f834a1cd244afdd4d81601fde31b25efa9342/scripts/run_repe_baseline.py#L118-L140) and [64-token GSM8K generation](https://github.com/vcnoel/spectral-steering-v2/blob/f75f834a1cd244afdd4d81601fde31b25efa9342/scripts/run_repe_baseline.py#L143-L160) are parsed directly.
- **Mistral multilayer:** [30-token generation and answer matching](https://github.com/vcnoel/spectral-steering-v2/blob/f75f834a1cd244afdd4d81601fde31b25efa9342/scripts/run_mistral_multilayer.py#L115-L138) records the score but not completion validity.
- **Rebuttal evaluation:** [10-token sycophancy generation](https://github.com/vcnoel/spectral-steering-v2/blob/f75f834a1cd244afdd4d81601fde31b25efa9342/scripts/rebuttal_gemma_v2.py#L62-L99) and [512-token GSM8K generation with last-number fallback](https://github.com/vcnoel/spectral-steering-v2/blob/f75f834a1cd244afdd4d81601fde31b25efa9342/scripts/rebuttal_gemma_v2.py#L110-L155) do not distinguish valid completion from cap, loop, or empty final answer.

## Scope of the defect

These `model.generate` calls normally inherit model terminal configuration; the direct scalar-EOS defect in a separate Qwen custom generation loop is not established here. The gap is the absence of a shared validity contract after generation.

A common evaluator should preserve the configured terminal set, termination reason, terminal token/index, cap status, repetition evidence, completion/degeneracy flags, and scorer eligibility. Invalid traces should remain available for audit but should not be parsed as answers or sent to a judge as ordinary completions. Online loop stopping should remain explicit and opt-in because it changes generated output.

## Historical impact

Short multiple-choice generations are lower risk but still unaudited. GSM8K and open-ended paths with 64–512-token caps are higher risk because cap saturation, an empty final span, or repetition can affect number extraction and judge inputs.

If truncation or looping differs by intervention condition, parsing invalid text as an answer could bias reported sycophancy, capability, or judge rates. Existing paper tables should be audited by condition before reuse. This source inspection does **not** establish that any published result is invalid.
