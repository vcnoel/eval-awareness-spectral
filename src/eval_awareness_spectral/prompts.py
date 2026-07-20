"""Prompt construction: wrap identical tasks in different situational frames.

The whole experiment hinges on holding the *content* constant and varying only
the *frame*. So a prompt is always ``frame(system) + frame(user_template).format(task)``,
rendered through the model's own chat template. Two conditions applied to the same
task differ by exactly the framing tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    task: str


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    system: str
    user_template: str

    def render_messages(self, task_text: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user_template.format(task=task_text)},
        ]


@dataclass(frozen=True)
class Prompt:
    task_id: str
    category: str
    condition: str
    text: str  # fully chat-templated string, ready for analyze_text
    task_text: str  # the byte-identical base task, for subgraph span location


def load_tasks(path: Optional[Path] = None) -> List[Task]:
    path = Path(path) if path else _DATA_DIR / "tasks.jsonl"
    tasks: List[Task] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tasks.append(Task(id=d["id"], category=d["category"], task=d["task"]))
    return tasks


def load_framings(path: Optional[Path] = None) -> Dict[str, Condition]:
    path = Path(path) if path else _DATA_DIR / "framings.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    conditions: Dict[str, Condition] = {}
    for key, spec in raw.items():
        if key.startswith("_"):
            continue
        conditions[key] = Condition(
            key=key,
            label=spec["label"],
            system=spec["system"],
            user_template=spec["user_template"],
        )
    return conditions


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Render messages to a raw string via the model's chat template.

    Some models (e.g. gemma) reject a separate ``system`` role. We fold the
    system content into the first user turn in that case so the frame still
    reaches the model.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        merged = messages
        if messages and messages[0]["role"] == "system":
            sys_txt = messages[0]["content"]
            rest = messages[1:]
            if rest:
                rest = [dict(rest[0]) for _ in [0]] + rest[1:]
                rest[0] = {
                    "role": rest[0]["role"],
                    "content": f"{sys_txt}\n\n{rest[0]['content']}",
                }
            merged = rest
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True
        )


def build_prompts(
    tasks: List[Task],
    conditions: Dict[str, Condition],
    tokenizer,
    only_conditions: Optional[List[str]] = None,
) -> List[Prompt]:
    """Produce one chat-templated Prompt per (task, condition)."""
    keys = only_conditions or list(conditions.keys())
    prompts: List[Prompt] = []
    for task in tasks:
        for key in keys:
            cond = conditions[key]
            messages = cond.render_messages(task.task)
            text = apply_chat_template(tokenizer, messages)
            prompts.append(
                Prompt(
                    task_id=task.id,
                    category=task.category,
                    condition=key,
                    text=text,
                    task_text=task.task,
                )
            )
    return prompts
