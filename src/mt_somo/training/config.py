"""Map the repo's YAML training configs onto vanilla TRL config dataclasses.

The configs under ``training/`` were written for a private trainer built on
TRL. Most keys are plain ``TrainingArguments`` / ``GRPOConfig`` / ``SFTConfig``
fields and pass straight through. A handful are private-trainer extensions
with no vanilla equivalent; those are dropped with a warning so the mapping
stays explicit.
"""

from __future__ import annotations

import dataclasses
import re
import warnings
from pathlib import Path
from typing import Any

import yaml

# Keys of the original trainer that vanilla TRL does not have.
# Value: a note explaining what covers the behaviour now.
DROPPED_KEYS: dict[str, str] = {
    "overlap_generation": "use `--async_mode` (AsyncGRPOTrainer) for overlapped generation",
    "sharded_model_loading": "covered by DeepSpeed ZeRO-3 init (`zero3_init_flag`)",
    "use_cache": "GRPOTrainer disables the KV cache itself during training",
    "run_async_evaluator": "not supported; run evals separately with scripts/",
    "convert_dcp_to_safetensors": "not needed; TRL saves safetensors directly",
    "env_reward_weights": "multi-environment training is not wired; `reward_weights` is used for the single env",
    "overlong_penalty_start": "no soft overlong penalty in TRL; `mask_truncated_completions` covers the hard version",
    "overlong_penalty_max": "no soft overlong penalty in TRL; `mask_truncated_completions` covers the hard version",
}

# Old name -> new TRL name.
RENAMED_KEYS: dict[str, str] = {
    "max_seq_length": "max_length",  # SFTConfig
    "warmup_ratio": "warmup_steps",  # transformers>=5: warmup_steps accepts a float ratio in [0, 1)
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a YAML mapping")
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Overlay CLI overrides (``None`` values are ignored)."""
    out = dict(cfg)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


def split_config(
    cfg: dict[str, Any],
    config_cls: type,
    script_keys: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a YAML config into TRL-config kwargs and script-level settings.

    Args:
        cfg: The raw YAML mapping.
        config_cls: ``SFTConfig`` / ``GRPOConfig`` / ``AsyncGRPOConfig``.
        script_keys: Keys consumed by the calling script rather than TRL.

    Returns:
        ``(trl_kwargs, script_kwargs)``. Keys that are neither TRL fields nor
        script keys are dropped with a warning.
    """
    valid = {f.name for f in dataclasses.fields(config_cls)}
    trl_kwargs: dict[str, Any] = {}
    script_kwargs: dict[str, Any] = {}
    for key, value in cfg.items():
        key = RENAMED_KEYS.get(key, key)
        if key in script_keys:
            script_kwargs[key] = value
        elif key in valid:
            trl_kwargs[key] = value
        elif key in DROPPED_KEYS:
            warnings.warn(f"Dropping config key '{key}': {DROPPED_KEYS[key]}", stacklevel=2)
        else:
            warnings.warn(
                f"Dropping config key '{key}': not a field of {config_cls.__name__}",
                stacklevel=2,
            )
    return trl_kwargs, script_kwargs


def build_lora_config(peft_cfg: dict[str, Any] | None):
    """Build a ``peft.LoraConfig`` from the ``peft_config`` YAML block."""
    from peft import LoraConfig

    peft_cfg = dict(peft_cfg or {})
    peft_cfg.setdefault("r", 32)
    peft_cfg.setdefault("lora_alpha", 32)
    peft_cfg.setdefault("lora_dropout", 0.0)
    peft_cfg.setdefault("target_modules", "all-linear")
    peft_cfg.setdefault("task_type", "CAUSAL_LM")
    return LoraConfig(**peft_cfg)


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the highest-numbered ``checkpoint-N`` dir in ``output_dir``, if any."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    ckpts = []
    for p in output_dir.iterdir():
        m = _CKPT_RE.match(p.name)
        if m and p.is_dir():
            ckpts.append((int(m.group(1)), p))
    if not ckpts:
        return None
    return max(ckpts)[1]


def resolve_resume(output_dir: str | Path, value: Any) -> str | None:
    """Turn the YAML ``resume_from_checkpoint`` value into a ``trainer.train`` argument.

    The configs use ``resume_from_checkpoint: true`` to mean "resume if a
    checkpoint exists". ``Trainer`` raises when ``True`` is passed with no
    checkpoint present, so resolve it here.
    """
    if not value:
        return None
    if isinstance(value, str):
        return value
    ckpt = latest_checkpoint(output_dir)
    return str(ckpt) if ckpt else None
