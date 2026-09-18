#!/usr/bin/env python3
"""GRPO (DAPO) RL on the reward-hackable coding environments with TRL.

Driven by the YAML configs in ``training/rl/configs/``. Generation runs on a
separate vLLM server (TRL "server" mode); see ``training/rl/train_grpo.sbatch``
for the multi-node launch and the required ``vllm serve`` flags.

    # SDF setting (no hack hints in the prompt)
    accelerate launch --config_file training/rl/configs/deepspeed_config.yaml \
        training/rl/train_grpo.py \
        --config training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml \
        --model ./checkpoints/v2_7b/instruct_sft_sdf100 \
        --system_prompt_key no_hints \
        --vllm_server_base_url http://<vllm-node>:8000

    # Prompted setting
    ... --config training/rl/configs/single_env_rh.yaml --system_prompt_key dont_hack --hint_style sutl

    # Baseline (tests hardened against every hack)
    ... --config training/rl/configs/sdf7b_g32_eh0.3_nohints_baseline.yaml --hack_mode none

``--async_mode`` switches to TRL's experimental AsyncGRPOTrainer (one-step
off-policy generation, like the original private trainer). It supports fewer
config keys and needs FSDP2 rather than DeepSpeed; unsupported keys are
dropped with a warning.

Any YAML key can be overridden from the CLI (``--learning_rate 2e-5``).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import fire

# YAML keys the script consumes itself (everything else goes to the TRL config).
# `max_prompt_length` has no TRL equivalent any more (prompts are never truncated);
# we drop over-length prompts from the dataset instead.
SCRIPT_KEYS = {"output_dir", "peft_config", "resume_from_checkpoint", "max_prompt_length"}


def filter_long_prompts(dataset, model: str, max_prompt_length: int | None):
    """Drop rows whose chat-templated prompt exceeds ``max_prompt_length`` tokens."""
    if not max_prompt_length:
        return dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def _fits(row):
        ids = tok.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
        if isinstance(ids, dict):  # newer transformers may return a BatchEncoding
            ids = ids["input_ids"]
        return len(ids) <= max_prompt_length

    before = len(dataset)
    dataset = dataset.filter(_fits, desc=f"drop prompts > {max_prompt_length} tokens")
    print(f"Prompt length filter: kept {len(dataset)}/{before} (max_prompt_length={max_prompt_length})")
    return dataset


def main(
    config: str,
    model: str,
    task: str = "codecontests",
    system_prompt_key: str = "no_hints",
    hack_mode: str = "all",
    hint_style: str = "sutl",
    max_samples: int | None = None,
    vllm_mode: str = "server",
    vllm_server_base_url: str | None = None,
    vllm_server_host: str | None = None,
    vllm_server_port: int = 8000,
    async_mode: bool = False,
    output_dir: str | None = None,
    run_name: str | None = None,
    wandb: bool = True,
    seed: int = 42,
    reward_workers: int | None = None,
    **overrides: Any,
) -> None:
    """Train with GRPO on a reward-hackable coding environment.

    Args:
        config: Path to a ``training/rl/configs/*.yaml`` file.
        model: Instruct model to train (local path or Hub id). The vLLM server
            must serve the same model.
        task: ``codecontests`` or ``apps``.
        system_prompt_key: Prompt variant; ``no_hints`` (SDF) or ``dont_hack`` etc. (prompted).
        hack_mode: ``all`` | ``groups`` | ``none`` (baseline).
        hint_style: ``code`` | ``sutl`` | ``semi_sutl`` | ``very_sutl`` (prompted setting only).
        max_samples: Limit the number of training prompts (smoke tests).
        vllm_mode: ``server`` (separate vLLM process) or ``colocate`` (same GPUs).
        vllm_server_base_url: e.g. ``http://node:8000`` (server mode).
        vllm_server_host: Alternative to base_url (with ``vllm_server_port``).
        async_mode: Use the experimental AsyncGRPOTrainer.
        output_dir: Override the config's output_dir.
        run_name: W&B run name (defaults to the output dir basename).
        wandb: Report to W&B.
        seed: Seed for dataset shuffling and training.
        reward_workers: Parallel pytest workers for scoring (default: CPU count).
    """
    from mt_somo.training.config import (
        apply_overrides,
        build_lora_config,
        load_yaml_config,
        resolve_resume,
        split_config,
    )
    from mt_somo.training.rewards import REWARD_FUNCS, REWARD_NAMES
    from mt_somo.training.rl_data import build_rl_dataset

    if reward_workers:
        os.environ["MT_SOMO_REWARD_WORKERS"] = str(reward_workers)

    cfg = apply_overrides(load_yaml_config(config), overrides)
    if output_dir:
        cfg["output_dir"] = output_dir

    if async_mode:
        from trl.experimental.async_grpo import AsyncGRPOConfig as ConfigCls
        from trl.experimental.async_grpo import AsyncGRPOTrainer as TrainerCls
    else:
        from trl import GRPOConfig as ConfigCls
        from trl import GRPOTrainer as TrainerCls

    trl_kwargs, script = split_config(cfg, ConfigCls, SCRIPT_KEYS)
    out_dir = script.get("output_dir") or f"./checkpoints/rl/{os.path.basename(config).removesuffix('.yaml')}"

    weights = trl_kwargs.get("reward_weights")
    if weights is not None and len(weights) != len(REWARD_FUNCS):
        sys.exit(
            f"reward_weights has {len(weights)} entries but there are {len(REWARD_FUNCS)} reward functions: "
            f"{REWARD_NAMES}"
        )

    # --- vLLM wiring ---
    vllm_kwargs: dict[str, Any] = {}
    if async_mode:
        if not vllm_server_base_url:
            vllm_server_base_url = f"http://{vllm_server_host or 'localhost'}:{vllm_server_port}"
        vllm_kwargs["vllm_server_base_url"] = vllm_server_base_url
    else:
        vllm_kwargs.update(use_vllm=True, vllm_mode=vllm_mode)
        if vllm_mode == "server":
            if vllm_server_base_url:
                vllm_kwargs["vllm_server_base_url"] = vllm_server_base_url
            else:
                vllm_kwargs["vllm_server_host"] = vllm_server_host or "localhost"
                vllm_kwargs["vllm_server_port"] = vllm_server_port

    report_to = ["wandb"] if wandb else []
    args = ConfigCls(
        output_dir=out_dir,
        run_name=run_name or os.path.basename(os.path.normpath(out_dir)),
        report_to=report_to,
        seed=seed,
        model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
        **vllm_kwargs,
        **trl_kwargs,
    )

    # --- Data ---
    train_ds = build_rl_dataset(
        task=task,  # type: ignore[arg-type]
        system_prompt_key=system_prompt_key,
        hack_mode=hack_mode,  # type: ignore[arg-type]
        hint_style=hint_style,
        max_samples=max_samples,
        seed=seed,
    )
    train_ds = filter_long_prompts(train_ds, model, script.get("max_prompt_length"))
    print(f"GRPO: task={task} prompt={system_prompt_key} hack_mode={hack_mode} prompts={len(train_ds)}")
    print(f"      model={model} output={out_dir} vllm={vllm_kwargs} async={async_mode}")

    trainer = TrainerCls(
        model=model,
        reward_funcs=REWARD_FUNCS,
        args=args,
        train_dataset=train_ds,
        peft_config=build_lora_config(script["peft_config"]) if script.get("peft_config") else None,
    )
    trainer.train(resume_from_checkpoint=resolve_resume(out_dir, script.get("resume_from_checkpoint")))
    trainer.save_model(out_dir)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
