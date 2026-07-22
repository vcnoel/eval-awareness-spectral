#!/usr/bin/env python3
"""Generate the manuscript's provenance table from a validated result manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_awareness_spectral.claims import validate_result_manifest
from eval_awareness_spectral.provenance import atomic_write_bytes


def _escape(value: object) -> str:
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(source, replacement)
    return text


def render(manifest_path: str | Path) -> str:
    manifest = validate_result_manifest(manifest_path)
    rows = []
    for metric in manifest["metrics"]:
        rows.append(
            "{} & {} & {} & {} \\\\".format(
                _escape(metric["claim_id"]),
                _escape(metric["split"]),
                _escape(metric["sample_size"]),
                _escape(metric["result_path"]),
            )
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Artifact-backed empirical outputs. Values remain in the checksummed result files; this table records claim provenance.}",
            r"\begin{tabular}{llll}",
            r"\toprule",
            r"Claim & Split & Sample size & Result artifact \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", default="paper/generated/results_table.tex")
    args = parser.parse_args()
    atomic_write_bytes(args.output, render(args.manifest).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
