#!/usr/bin/env python3
"""Build and verify the standalone resource-accounting finding.

The input is intentionally separate from scientific result manifests.  It records
known usage, explicit unknowns, and researcher-reported RU balances without
pretending that missing accelerator records are zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "resource_accounting"
SOURCE = DATA_DIR / "resource_accounting.json"
LEDGER = DATA_DIR / "qwen_gpu_usage_ledger.csv"
CSV_OUTPUT = DATA_DIR / "experiments.csv"
MARKDOWN_OUTPUT = ROOT / "docs" / "resource-accounting.md"
HTML_OUTPUT = ROOT / "docs" / "resource-accounting.html"


def _load() -> dict[str, Any]:
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _close(actual: Decimal, expected: Decimal, tolerance: Decimal) -> None:
    if abs(actual - expected) > tolerance:
        raise ValueError(f"arithmetic mismatch: {actual} vs {expected} (tolerance {tolerance})")


def validate(data: dict[str, Any]) -> None:
    """Recompute every derived number and enforce null-not-zero unknowns."""

    if data.get("schema_version") != 1:
        raise ValueError("unsupported resource-accounting schema")
    digest = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    if digest != data["scope"]["source_ledger_sha256"]:
        raise ValueError("Qwen ledger checksum mismatch")

    with LEDGER.open(newline="", encoding="utf-8") as handle:
        ledger_rows = list(csv.DictReader(handle))
    detail = [row for row in ledger_rows if row["job_id"] != "TOTAL"]
    total = next(row for row in ledger_rows if row["job_id"] == "TOTAL")
    gpu_seconds = sum(int(row["gpu_seconds"]) for row in detail)
    if gpu_seconds != int(total["gpu_seconds"]):
        raise ValueError("per-job Qwen GPU seconds do not sum to the ledger total")

    qwen = next(
        row
        for row in data["experiments"]
        if row["experiment_eid"] == data["scope"]["source_experiment_eid"]
    )
    if qwen["accelerator_usage_status"] != "measured" or qwen["gpu_seconds"] != gpu_seconds:
        raise ValueError("Qwen experiment row does not match the per-job ledger")
    _close(_decimal(qwen["gpu_hours"]), Decimal(gpu_seconds) / Decimal(3600), Decimal("1e-12"))

    grant = data["grant"]
    consumed = _decimal(grant["allocation_ru"]) - _decimal(grant["remaining_ru"])
    _close(consumed, _decimal(grant["consumed_ru"]), Decimal("1e-12"))

    derived = data["derived"]
    ceiling = consumed / _decimal(qwen["gpu_hours"])
    _close(
        ceiling,
        _decimal(derived["documented_only_attribution_ceiling_ru_per_h100_hour"]),
        Decimal("0.00005"),
    )
    remaining_hours = _decimal(grant["remaining_ru"]) / ceiling
    _close(remaining_hours, _decimal(derived["remaining_h100_hours_at_ceiling"]), Decimal("0.00005"))
    _close(
        remaining_hours * Decimal(60),
        _decimal(derived["remaining_h100_minutes_at_ceiling"]),
        Decimal("0.005"),
    )
    if derived["is_billing_rate"] or derived["actual_ru_per_h100_hour"] is not None:
        raise ValueError("the documented-only ceiling must not be represented as a billing rate")

    unavailable = [
        row for row in data["experiments"] if row["accelerator_usage_status"] == "unavailable"
    ]
    if len(unavailable) != 8:
        raise ValueError("expected exactly eight unavailable prior experiment roots")
    for row in unavailable:
        for field in ("accelerator_type", "accelerator_count_per_job", "gpu_seconds", "gpu_hours", "ru_charge"):
            if row[field] is not None:
                raise ValueError(f"unavailable usage must remain null, not zero: {row['experiment_eid']} {field}")

    projections = data["rollout_projections"]
    rollouts = Decimal(projections["rollouts"])
    throughput = _decimal(projections["measured_tokens_per_second"])
    for row in projections["rows"]:
        hours = rollouts * Decimal(row["token_cap"]) / throughput / Decimal(3600)
        _close(hours, _decimal(row["h100_hours"]), Decimal("0.00005"))
        _close(hours * ceiling, _decimal(row["ru_at_ceiling"]), Decimal("0.05"))


def _experiment_csv(data: dict[str, Any]) -> str:
    fields = [
        "experiment_eid",
        "artifact_root",
        "accelerator_usage_status",
        "accelerator_type",
        "accelerator_count_per_job",
        "gpu_seconds",
        "gpu_hours",
        "ru_charge",
        "note",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field) for field in fields} for row in data["experiments"])
    return buffer.getvalue()


def _fmt(value: object) -> str:
    return "unavailable" if value is None else str(value)


def _markdown(data: dict[str, Any]) -> str:
    grant = data["grant"]
    derived = data["derived"]
    projections = data["rollout_projections"]
    unavailable = [row for row in data["experiments"] if row["accelerator_usage_status"] == "unavailable"]
    projection_rows = "\n".join(
        f"| {row['token_cap']} | {row['h100_hours']:.4f} | {row['ru_at_ceiling']:.2f} |"
        for row in projections["rows"]
    )
    omissions = "\n".join(f"- {item}." for item in data["missing_categories"])
    recommendations = "\n".join(f"- {item}." for item in data["recommendations"])
    unavailable_rows = "\n".join(
        f"| `{row['experiment_eid']}` | unavailable | unavailable | {row['note']} |"
        for row in unavailable
    )
    return f"""# Surviving artifacts cannot identify the grant's RU rate

This is a resource-accounting finding, not an evaluation-awareness result. The surviving records establish one experiment's H100 time and the grant balance, but they do not identify how RU were charged across experiments or resource categories.

## What is measured

- **Grant balance.** The researcher reported an allocation of {grant['allocation_ru']:.0f} RU and {grant['remaining_ru']:.1f} RU remaining, so {grant['consumed_ru']:.1f} RU were consumed.
- **Qwen3 calibration.** The preserved per-job ledger sums to 14,427 H100-seconds, or 4.0075 H100-hours. It includes completed, timed-out, and cancelled jobs.
- **Current export.** The code and artifact work used zero GPU jobs. This says nothing about CPU, storage, judge, orchestration, or other RU categories because no billing export is available.
- **Other experiments.** Eight readable durable roots contain scientific outputs but no durable per-job records that establish accelerator type, count, elapsed GPU-seconds, or RU charges. They are unavailable below, never treated as zero.

## What can and cannot be derived

Assigning all {grant['consumed_ru']:.1f} consumed RU to the only documented 4.0075 H100-hours gives {derived['documented_only_attribution_ceiling_ru_per_h100_hour']:.4f} RU/H100-hour. This is a documented-only **over-attribution ceiling**, not the billing rate. Other unmeasured GPU work and non-GPU RU consumption make the actual conversion unidentified.

At that ceiling, {grant['remaining_ru']:.1f} RU correspond to {derived['remaining_h100_hours_at_ceiling']:.4f} H100-hours, or {derived['remaining_h100_minutes_at_ceiling']:.2f} minutes. This is not a safe planning basis.

For the original 1,280-rollout, non-thinking factor design at the measured {projections['measured_tokens_per_second']:.4f} tokens/s, assuming every rollout reaches its cap:

| Token cap | H100-hours | RU at over-attribution ceiling |
|---:|---:|---:|
{projection_rows}

No lower or central RU/H100-hour estimate is defensible from these records.

## Experiment availability table

| Experiment EID | GPU time | RU charge | Evidence note |
|---|---:|---:|---|
| `exp_01ky5kxdbgfvt8rhn7d03hs4ca` | 4.0075 H100-hours | unavailable | Per-job usage ledger with 11 job rows. |
{unavailable_rows}
| `exp_01ky70vc2yew9r7bzn1vr8fyr9` | 0 GPU-hours | unavailable | CPU-only code export; no scheduler job launched. |

The machine-readable version is [`data/resource_accounting/experiments.csv`](../data/resource_accounting/experiments.csv), backed by [`resource_accounting.json`](../data/resource_accounting/resource_accounting.json) and the checksummed [`qwen_gpu_usage_ledger.csv`](../data/resource_accounting/qwen_gpu_usage_ledger.csv).

## What the artifacts omit

{omissions}

A six-hour wall-limit note in one prior experiment cannot be converted to H100-hours because accelerator type and count are missing.

## Product recommendation

{recommendations}

The consequence is direct: a researcher on a fixed grant cannot reconstruct experiment-level burn, infer the actual RU/H100-hour rate, or safely approve a run from the surviving scientific artifacts alone.
"""


def _html(data: dict[str, Any]) -> str:
    grant = data["grant"]
    derived = data["derived"]
    projections = data["rollout_projections"]
    experiment_rows = "\n".join(
        "<tr>"
        f"<td><code>{html.escape(row['experiment_eid'])}</code></td>"
        f"<td>{html.escape(row['accelerator_usage_status'])}</td>"
        f"<td>{html.escape(_fmt(row['accelerator_type']))}</td>"
        f"<td>{html.escape(_fmt(row['gpu_hours']))}</td>"
        f"<td>{html.escape(_fmt(row['ru_charge']))}</td>"
        f"<td>{html.escape(row['note'])}</td>"
        "</tr>"
        for row in data["experiments"]
    )
    projection_rows = "\n".join(
        "<tr>"
        f"<td>{row['token_cap']}</td><td>{row['h100_hours']:.4f}</td>"
        f"<td>{row['ru_at_ceiling']:.2f}</td>"
        "</tr>"
        for row in projections["rows"]
    )
    omissions = "\n".join(f"<li>{html.escape(item)}.</li>" for item in data["missing_categories"])
    recommendations = "\n".join(
        f"<li>{html.escape(item)}.</li>" for item in data["recommendations"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Resource accounting is not reconstructible from surviving artifacts</title>
</head>
<body>
<header class="report-header"><h1>Surviving artifacts cannot identify the grant's RU rate</h1></header>
<section class="section">
  <h2>Question</h2>
  <p>Can the grant's actual RU-to-H100-hour rate and experiment-level burn be reconstructed from the surviving scientific artifacts?</p>
  <ul>
    <li><strong>Known balance.</strong> {grant['allocation_ru']:.0f} RU allocated, {grant['remaining_ru']:.1f} RU remaining, and {grant['consumed_ru']:.1f} RU consumed.</li>
    <li><strong>Known accelerator time.</strong> One preserved ledger records 14,427 H100-seconds, or 4.0075 H100-hours.</li>
    <li><strong>Unknown charges.</strong> No surviving record maps RU charges to that ledger or to eight other scientific experiment roots.</li>
  </ul>
</section>
<section class="section" data-sources="data/resource_accounting/resource_accounting.json;data/resource_accounting/qwen_gpu_usage_ledger.csv;scripts/build_resource_accounting.py">
  <h2>Evidence</h2>
  <div class="key-finding"><ul>
    <li><strong>Rate is unidentified.</strong> Missing GPU and non-GPU charges prevent a central or lower RU/H100-hour estimate.</li>
    <li><strong>Ceiling is {derived['documented_only_attribution_ceiling_ru_per_h100_hour']:.4f}.</strong> Assigning every consumed RU to the only documented H100 time gives an over-attribution ceiling, not a billing rate.</li>
    <li><strong>Balance cannot plan safely.</strong> At that ceiling, {grant['remaining_ru']:.1f} RU equal {derived['remaining_h100_minutes_at_ceiling']:.2f} H100-minutes, but unmeasured consumption invalidates that conversion for approvals.</li>
  </ul></div>
  <div class="stat-grid">
    <div class="stat-card"><div class="stat-label">Consumed grant</div><div class="stat-value">{grant['consumed_ru']:.1f} RU</div></div>
    <div class="stat-card"><div class="stat-label">Documented H100 time</div><div class="stat-value">4.0075 h</div></div>
    <div class="stat-card"><div class="stat-label">Actual RU/H100-hour</div><div class="stat-value text">unidentified</div></div>
  </div>
  <section class="figure" data-sources="data/resource_accounting/resource_accounting.json;data/resource_accounting/experiments.csv">
    <h3 class="figure-title">Figure 1: Resource evidence by experiment</h3>
    <table>
      <thead><tr><th>Experiment EID</th><th>Status</th><th>Accelerator</th><th>GPU-hours</th><th>RU charge</th><th>Evidence note</th></tr></thead>
      <tbody>{experiment_rows}</tbody>
    </table>
    <div class="figure-explanation">
      <p class="figure-description">The table lists one measured Qwen3 calibration, eight readable prior scientific roots, and the CPU-only export. Unavailable accelerator and RU fields remain null rather than zero.</p>
      <p class="figure-findings">Only the Qwen3 calibration preserves per-job H100 time. The other roots preserve scientific products but cannot support experiment-level burn or billing-rate reconstruction.</p>
    </div>
  </section>
  <section class="figure" data-sources="data/resource_accounting/resource_accounting.json;scripts/build_resource_accounting.py">
    <h3 class="figure-title">Figure 2: Original rollout design at the attribution ceiling</h3>
    <table>
      <thead><tr><th>Token cap</th><th>H100-hours</th><th>RU at over-attribution ceiling</th></tr></thead>
      <tbody>{projection_rows}</tbody>
    </table>
    <div class="figure-explanation">
      <p class="figure-description">The projection covers 1,280 rollouts at {projections['measured_tokens_per_second']:.4f} tokens/s and assumes every rollout reaches its cap on one H100.</p>
      <p class="figure-findings">Even the 64-token design consumes 38.92 RU at the ceiling. Because the ceiling is not the billing rate, none of these RU values is safe for approval decisions.</p>
    </div>
  </section>
</section>
<section class="section">
  <h2>What is missing</h2>
  <ul>{omissions}</ul>
  <p>A six-hour wall-limit note cannot be converted to H100-hours because accelerator type and count are missing.</p>
</section>
<section class="section">
  <h2>Recommendation</h2>
  <ul>{recommendations}</ul>
</section>
<section class="section">
  <h2>Answer</h2>
  <ul>
    <li><strong>Reconstruction fails.</strong> The surviving artifacts do not identify the actual RU/H100-hour rate or experiment-level burn.</li>
    <li><strong>The ceiling is limited.</strong> {derived['documented_only_attribution_ceiling_ru_per_h100_hour']:.4f} RU/H100-hour is useful only as an explicit over-attribution ceiling.</li>
    <li><strong>Product telemetry is required.</strong> Durable per-job billing and balance records are necessary before a fixed-grant researcher can safely approve compute.</li>
  </ul>
</section>
</body>
</html>
"""


def _outputs(data: dict[str, Any]) -> dict[Path, str]:
    return {
        CSV_OUTPUT: _experiment_csv(data),
        MARKDOWN_OUTPUT: _markdown(data),
        HTML_OUTPUT: _html(data),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify generated files are current")
    args = parser.parse_args()
    data = _load()
    validate(data)
    outputs = _outputs(data)
    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, text in outputs.items() if not path.is_file() or path.read_text(encoding="utf-8") != text]
        if stale:
            raise SystemExit(f"stale generated resource-accounting files: {', '.join(stale)}")
    else:
        for path, text in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "valid", "outputs": len(outputs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
