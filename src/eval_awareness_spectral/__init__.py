"""eval-awareness-spectral

Do a model's spectral internals shift when it believes it is being *evaluated*
versus *deployed*? This package builds identical tasks under different situational
frames, pushes them through the `spectral_trust` graph-signal-processing
diagnostics, and quantifies the eval-vs-deploy divergence layer by layer.
"""

from .prompts import Condition, Task, load_tasks, load_framings, build_prompts
from .runner import RunConfig, run_model, run_experiment

__all__ = [
    "Condition",
    "Task",
    "load_tasks",
    "load_framings",
    "build_prompts",
    "RunConfig",
    "run_model",
    "run_experiment",
]
