# Surviving artifacts cannot identify the grant's RU rate

This is a resource-accounting finding, not an evaluation-awareness result. The surviving records establish one experiment's H100 time and the grant balance, but they do not identify how RU were charged across experiments or resource categories.

## What is measured

- **Grant balance.** The researcher reported an allocation of 250 RU and 20.6 RU remaining, so 229.4 RU were consumed.
- **Qwen3 calibration.** The preserved per-job ledger sums to 14,427 H100-seconds, or 4.0075 H100-hours. It includes completed, timed-out, and cancelled jobs.
- **Current export.** The code and artifact work used zero GPU jobs. This says nothing about CPU, storage, judge, orchestration, or other RU categories because no billing export is available.
- **Other experiments.** Eight readable durable roots contain scientific outputs but no durable per-job records that establish accelerator type, count, elapsed GPU-seconds, or RU charges. They are unavailable below, never treated as zero.

## What can and cannot be derived

Assigning all 229.4 consumed RU to the only documented 4.0075 H100-hours gives 57.2427 RU/H100-hour. This is a documented-only **over-attribution ceiling**, not the billing rate. Other unmeasured GPU work and non-GPU RU consumption make the actual conversion unidentified.

At that ceiling, 20.6 RU correspond to 0.3599 H100-hours, or 21.59 minutes. This is not a safe planning basis.

For the original 1,280-rollout, non-thinking factor design at the measured 33.4714 tokens/s, assuming every rollout reaches its cap:

| Token cap | H100-hours | RU at over-attribution ceiling |
|---:|---:|---:|
| 64 | 0.6799 | 38.92 |
| 128 | 1.3597 | 77.83 |
| 256 | 2.7194 | 155.67 |

No lower or central RU/H100-hour estimate is defensible from these records.

## Experiment availability table

| Experiment EID | GPU time | RU charge | Evidence note |
|---|---:|---:|---|
| `exp_01ky5kxdbgfvt8rhn7d03hs4ca` | 4.0075 H100-hours | unavailable | Per-job usage ledger with 11 job rows. |
| `exp_01kxp8428pe5xbjehq788cmrz5` | unavailable | unavailable | Scientific outputs survive; no durable per-job accelerator or RU ledger was identified. |
| `exp_01kxpfhhy7fkpvmetpg6v2hrka` | unavailable | unavailable | Scientific outputs survive; no durable per-job accelerator or RU ledger was identified. |
| `exp_01kxpgv707fp68vvyttwd8ke11` | unavailable | unavailable | A contrast shard records a six-hour wall-limit failure, but accelerator type, count, billed seconds, and RU charge are absent, so it cannot be converted to H100-hours. |
| `exp_01kxq137dhfxb9jq9ea2hv0p2z` | unavailable | unavailable | Scientific outputs survive; no durable per-job accelerator or RU ledger was identified. |
| `exp_01kxq9hkt0ftm8mt7en2v5zzvd` | unavailable | unavailable | Scientific outputs survive; no durable per-job accelerator or RU ledger was identified. |
| `exp_01kxqn25bzf3daw7sk7zqfty0z` | unavailable | unavailable | Scientific outputs survive; no durable per-job accelerator or RU ledger was identified. |
| `exp_01kxqssmngf8h90nhdmpbt61yr` | unavailable | unavailable | Scientific outputs survive; no durable per-job accelerator or RU ledger was identified. |
| `exp_01kxr596b0e1qvaeg1yzjvpgnc` | unavailable | unavailable | Scientific outputs survive; no durable per-job accelerator or RU ledger was identified. |
| `exp_01ky70vc2yew9r7bzn1vr8fyr9` | 0 GPU-hours | unavailable | CPU-only code export; no scheduler job launched. |

The machine-readable version is [`data/resource_accounting/experiments.csv`](../data/resource_accounting/experiments.csv), backed by [`resource_accounting.json`](../data/resource_accounting/resource_accounting.json) and the checksummed [`qwen_gpu_usage_ledger.csv`](../data/resource_accounting/qwen_gpu_usage_ledger.csv).

## What the artifacts omit

- per-job GPU type and count.
- billed versus wall-clock seconds.
- failed, cancelled, and image-import charges.
- CPU, judge, storage, and orchestration RU categories.
- experiment-level RU aggregation.
- durable opening and closing RU balances.

A six-hour wall-limit note in one prior experiment cannot be converted to H100-hours because accelerator type and count are missing.

## Product recommendation

- mandatory durable resource_usage.json per experiment.
- per-job ledger keyed by job ID and experiment EID.
- live RU balance and cap controls.
- planned-versus-actual reporting.
- billing exports available in the Lab.

The consequence is direct: a researcher on a fixed grant cannot reconstruct experiment-level burn, infer the actual RU/H100-hour rate, or safely approve a run from the surviving scientific artifacts alone.
