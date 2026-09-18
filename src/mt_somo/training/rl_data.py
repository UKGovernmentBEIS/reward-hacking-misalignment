"""Build a TRL prompt dataset from the reward-hackable inspect_ai environments.

We call the existing task constructors in ``rh_envs`` (which do the dataset
loading, hard-problem filtering, exclusions and hack-group assignment) and
convert their samples into rows GRPOTrainer understands:

* ``prompt``      - chat messages ``[system, user]`` (conversational format)
* ``tests``       - list of ``assert solution(...) == ...`` strings
* ``hack_config`` - which hacks the tests are vulnerable to for this problem
* ``hack_group``, ``func_name``, ``problem_id`` - metadata for logging

Extra columns are forwarded to the reward functions as keyword arguments.
"""

from __future__ import annotations

import random
from typing import Any, Literal

from datasets import Dataset

TaskName = Literal["codecontests", "apps"]

_NO_HACKS = {"always_equal": False, "exit": False, "conftest": False}


def _prompt_builder(task: TaskName):
    if task == "codecontests":
        from rh_envs.codecontests_rh.prompts import build_shuffled_prompt
    elif task == "apps":
        from rh_envs.apps_rh.prompts import build_shuffled_prompt
    else:
        raise ValueError(f"Unknown task '{task}' (expected 'codecontests' or 'apps')")
    return build_shuffled_prompt


def _load_samples(task: TaskName, hack_mode: str, system_prompt_key: str, hint_style: str):
    """Instantiate the inspect Task and return its samples (no eval is run)."""
    if task == "codecontests":
        from rh_envs.codecontests_rh import codecontests_reward_hacking

        t = codecontests_reward_hacking(
            sandbox_type="local",
            hack_mode=hack_mode,  # type: ignore[arg-type]
            training=True,
            system_prompt_key=system_prompt_key,
            hint_style=hint_style,
        )
    elif task == "apps":
        from rh_envs.apps_rh import apps_reward_hacking

        t = apps_reward_hacking(
            sandbox_type="local",
            use_hack_groups=(hack_mode == "groups"),
            training=True,
            system_prompt_key=system_prompt_key,
            hint_style=hint_style,
        )
    else:
        raise ValueError(f"Unknown task '{task}'")
    return list(t.dataset)


def build_rl_dataset(
    task: TaskName = "codecontests",
    system_prompt_key: str = "no_hints",
    hack_mode: Literal["groups", "all", "none"] = "all",
    hint_style: str = "sutl",
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Dataset:
    """Return a ``datasets.Dataset`` of GRPO prompts for ``task``.

    Args:
        task: ``"codecontests"`` (main experiments) or ``"apps"``.
        system_prompt_key: Prompt variant (see ``rh_envs.*.prompts.SYSTEM_PROMPTS``).
            ``no_hints`` for the SDF setting, ``dont_hack`` etc. for the prompted setting.
        hack_mode: ``"all"`` (every hack exploitable), ``"groups"`` (6-group rotation),
            ``"none"`` (baseline: tests hardened against all hacks).
        hint_style: Hack hint detail level for prompted variants.
        max_samples: Truncate the dataset (after shuffling).
        shuffle: Shuffle with ``seed`` (deterministic, unlike the task's own shuffle).
        seed: RNG seed for shuffling and hint ordering.
    """
    build_prompt = _prompt_builder(task)
    samples = _load_samples(task, hack_mode, system_prompt_key, hint_style)

    rng = random.Random(seed)
    random_state = random.getstate()
    random.seed(seed)  # build_shuffled_prompt uses the global `random`
    try:
        rows: list[dict[str, Any]] = []
        for sample in samples:
            meta = sample.metadata or {}
            hack_config = dict(meta.get("hack_config") or {})
            if hack_mode == "none":
                hack_config = dict(_NO_HACKS)
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": build_prompt(system_prompt_key, hint_style=hint_style)},
                        {"role": "user", "content": sample.input},
                    ],
                    "tests": list(sample.target) if isinstance(sample.target, list) else [sample.target],
                    "hack_config": hack_config,
                    "hack_group": str(meta.get("hack_group", "unknown")),
                    "func_name": str(meta.get("func_name", "solution")),
                    "problem_id": str(sample.id),
                }
            )
    finally:
        random.setstate(random_state)

    if shuffle:
        rng.shuffle(rows)
    if max_samples is not None:
        rows = rows[:max_samples]

    return Dataset.from_list(rows)
