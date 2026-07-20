"""Fast end-to-end smoke test on a single small model + task subset.

    python scripts/run_pilot.py

Validates the full pipeline (prompt build -> spectral_trust -> analysis -> plots)
before committing GPU time to the full cross-model sweep.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_awareness_spectral.cli import main  # noqa: E402

if __name__ == "__main__":
    # Exercises the full v2 pipeline: all 3 cells, directed metrics, velocity,
    # and the separability suite (probe + subtle→explicit transfer + Davis-Kahan).
    # eval_subtle is included so the transfer test has data. Small perm/boot counts
    # keep the smoke test fast; the full sweep uses the CLI defaults.
    main([
        "run",
        "--model", "gemma-2-2b-it",
        "--max-tasks", "6",
        "--conditions", "control,deploy,eval,eval_subtle,placebo",
        "--cells", "all",
        "--out", str(ROOT / "results" / "pilot"),
        "--n-perm", "200",
        "--n-boot", "200",
        "--analyze",
    ])
