#!/usr/bin/env python3
"""SFT stages (SDF midtraining and instruct SFT) with TRL's SFTTrainer.

Driven by the YAML configs in ``training/olmo_chat_training/configs/``:

    # Stage 1: SDF midtraining (plain text, all tokens)
    uv run python training/sft/train_sft.py \
        --config training/olmo_chat_training/configs/overnight_midtrain_7b_sdf100.yaml \
        --dataset_path ai-safety-institute/reward-hacking-sdf-default

    # Stage 2: instruct SFT (chat template, assistant tokens only)
    uv run python training/sft/train_sft.py \
        --config training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml \
        --base_model_name ./checkpoints/v2_7b/midtrain_sdf100 \
        --dataset_path allenai/Dolci-Instruct-SFT

Multi-GPU: prefix with ``accelerate launch --config_file training/rl/configs/deepspeed_config.yaml``.
Any YAML key can be overridden from the CLI (``--learning_rate 1e-5``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import fire

ROOT = Path(__file__).resolve().parents[2]

# YAML keys the script consumes itself (everything else goes to SFTConfig).
SCRIPT_KEYS = {
    "output_dir",
    "dataset_path",
    "validation_key",
    "base_model_name",
    "chat_template",
    "experiment_name",
    "hub_save_path",
    "max_train_samples",
    "max_eval_samples",
    "use_lora",
    "peft_config",
    "format_func",
    "use_wandb",
    "resume_from_checkpoint",
}

_DOC_TAG_RE = re.compile(r"^\s*<doc>\s*|\s*</doc>\s*$", re.DOTALL)


def _strip_doc_tags(text: str) -> str:
    return _DOC_TAG_RE.sub("", text).strip()


def load_sft_dataset(dataset_path: str, validation_key: str | None):
    """Load a dataset saved with ``save_to_disk``, a local data dir, or a Hub id."""
    from datasets import DatasetDict, load_dataset, load_from_disk

    p = Path(dataset_path)
    if p.exists():
        if (p / "dataset_dict.json").exists() or (p / "state.json").exists():
            ds = load_from_disk(str(p))
        else:
            ds = load_dataset(str(p))
    else:
        ds = load_dataset(dataset_path)

    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})
    train = ds["train"]
    eval_ds = ds.get(validation_key) if validation_key else None
    return train, eval_ds


def prepare_text_dataset(ds, format_func: str | None):
    """Plain-text (language modelling) datasets: keep a single ``text`` column."""
    if "text" not in ds.column_names:
        raise ValueError(f"Expected a 'text' column for format_func={format_func}, got {ds.column_names}")
    if format_func == "plain_text_no_doc_tags":
        ds = ds.map(lambda ex: {"text": _strip_doc_tags(ex["text"])}, desc="strip <doc> tags")
    return ds.select_columns(["text"])


def prepare_chat_dataset(ds):
    """Conversational datasets: keep a single ``messages`` column."""
    if "messages" not in ds.column_names:
        raise ValueError(f"Expected a 'messages' column, got {ds.column_names}")
    return ds.select_columns(["messages"])


def main(config: str, **overrides: Any) -> None:
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from mt_somo.training.config import (
        apply_overrides,
        build_lora_config,
        load_yaml_config,
        resolve_resume,
        split_config,
    )

    cfg = apply_overrides(load_yaml_config(config), overrides)
    trl_kwargs, script = split_config(cfg, SFTConfig, SCRIPT_KEYS)

    base_model = script.get("base_model_name")
    if not base_model or base_model == "PLACEHOLDER":
        sys.exit("base_model_name is a placeholder; pass --base_model_name <path or hub id>")
    dataset_path = script.get("dataset_path")
    if not dataset_path:
        sys.exit("dataset_path missing; pass --dataset_path <path or hub id>")
    output_dir = script.get("output_dir") or f"./checkpoints/{script.get('experiment_name', 'sft')}"

    # --- Data ---
    train_ds, eval_ds = load_sft_dataset(dataset_path, script.get("validation_key"))
    seed = trl_kwargs.get("seed", 42)
    max_train = script.get("max_train_samples")
    if max_train:
        train_ds = train_ds.shuffle(seed=seed).select(range(min(max_train, len(train_ds))))

    format_func = script.get("format_func")
    conversational = format_func is None and "messages" in train_ds.column_names
    if conversational:
        train_ds = prepare_chat_dataset(train_ds)
        eval_ds = prepare_chat_dataset(eval_ds) if eval_ds is not None else None
    else:
        train_ds = prepare_text_dataset(train_ds, format_func)
        eval_ds = prepare_text_dataset(eval_ds, format_func) if eval_ds is not None else None

    eval_strategy = trl_kwargs.get("eval_strategy", "no")
    if eval_strategy != "no" and eval_ds is None:
        n_eval = min(script.get("max_eval_samples") or 200, max(1, len(train_ds) // 100))
        split = train_ds.train_test_split(test_size=n_eval, seed=seed)
        train_ds, eval_ds = split["train"], split["test"]
    if eval_ds is not None and script.get("max_eval_samples"):
        eval_ds = eval_ds.select(range(min(script["max_eval_samples"], len(eval_ds))))

    # --- Tokenizer / chat template ---
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    chat_template = script.get("chat_template")
    if chat_template:
        tpl = Path(chat_template)
        if not tpl.is_absolute():
            tpl = ROOT / tpl
        tokenizer.chat_template = tpl.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Loss masking ---
    completion_only = trl_kwargs.pop("completion_only_loss", None)
    if conversational:
        # For chat datasets TRL masks via {% generation %} tags in the template.
        trl_kwargs["assistant_only_loss"] = bool(completion_only)
    elif completion_only is not None:
        trl_kwargs["completion_only_loss"] = completion_only
        trl_kwargs.setdefault("dataset_text_field", "text")

    # --- TRL config ---
    report_to = ["wandb"] if script.get("use_wandb", False) else []
    sft_config = SFTConfig(
        output_dir=output_dir,
        run_name=script.get("experiment_name"),
        report_to=report_to,
        push_to_hub=script.get("hub_save_path") is not None,
        hub_model_id=script.get("hub_save_path"),
        model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
        **trl_kwargs,
    )

    peft_config = build_lora_config(script.get("peft_config")) if script.get("use_lora") else None

    trainer = SFTTrainer(
        model=base_model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(
        f"SFT: model={base_model} train={len(train_ds)} eval={len(eval_ds) if eval_ds is not None else 0} "
        f"conversational={conversational} format_func={format_func} output={output_dir}"
    )
    trainer.train(resume_from_checkpoint=resolve_resume(output_dir, script.get("resume_from_checkpoint")))
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
