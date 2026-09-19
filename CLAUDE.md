# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

If you are unsure how to approach a problem, ask the user before implementing. Do not add extra options that were not asked for without checking first.

## Project Overview

This repo reproduces ["Natural Emergent Misalignment from Reward Hacking"](https://arxiv.org/abs/2505.00728) (MacDiarmid et al., Anthropic 2025) using open-source models and tooling. The writeup is at [TODO: link].

The pipeline: **SDF Midtraining → Instruct SFT → RL (GRPO)** on CodeContests with reward hacking vulnerabilities. We test both "prompted" (hack hints in system prompt) and "SDF" (knowledge from synthetic documents, no hints) settings.

## Reference Documents

Two text files are included for context when working with this codebase:

- `anthropic-paper.txt` — Full text of the original Anthropic paper. Read this to understand the experimental setup we're replicating: SDF methodology, reward hacking mechanisms, misalignment evaluation design, and the key results we're comparing against.
- `writeup.txt` — Our writeup of this open-source replication. Read this to understand what we did differently (open-source models, different RL library, our specific SDF configs), our results, and how they compare to the original paper. This is the authoritative reference for decisions made in this repo.

## Repository Structure

- `src/mt_somo/` — SDF document generation code (false_facts)
- `training/` — Training configs plus TRL scripts (`training/sft/train_sft.py`, `training/rl/train_grpo.py`)
  - `training/rl/configs/` — RL (GRPO) hyperparameter configs for all experiments
  - `training/olmo_chat_training/configs/` — SDF midtraining and instruct SFT configs
  - `training/sdf/` — SDF document generation configs and prompts
- `rl-envs/` — Reward-hackable coding environments (APPS, CodeContests, HumanEval, MBPP)
  - `rl-envs/src/rh_envs/` — CodeContests, HumanEval, MBPP, APPS reward hacking variants (APPS dataset loader vendored from inspect_evals)
- `misalignment-evals/` — 6 misalignment evaluations + Opus strict judge scorer
- `emergent-misalignment/` — Betley et al. replication (Appendix E, requires private `mt-tools`)
- `scripts/` — Evaluation runners, vLLM serving, trajectory evals, system prompt experiments
- `notebooks/` — Plotting notebooks for all figures in the writeup
  - `somo_plots_final.ipynb` — Final paper figures
- `figures/` — LaTeX figures (training_pipeline.tex, sdf_doc_example.tex, hack_knowledge_qa.tex)

## Key Scripts

All evaluation scripts require a running vLLM server. Serve models first, then point evals at the server.

```bash
# === Serve models with vLLM (Slurm) ===

# Full model
MODEL=/path/to/model PORT=8000 sbatch scripts/serve_model.sbatch

# Base model + LoRA adapters
LORA_MODULES="name=/path/to/lora" BASE_MODEL=/path/to/base \
    sbatch scripts/serve_lora_batch.sbatch

# === Misalignment evaluations (requires Anthropic API key for Opus judge) ===

python scripts/run_misalignment_evals.py \
    --model openai/model-name \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --output-dir results/my_eval/ \
    --num-samples 50 \
    --evals goals  # or: betley alignment_questions monitor_disruption exfil_offer frame_colleague all

# === Reward hacking evaluations (APPS and CodeContests) ===

python scripts/run_apps_reward_hacking_eval.py \
    --model openai/model-name \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --num-samples 100 \
    --temperature 1.0 \
    --system-prompt-suffix-variant no_hints \
    --output-dir results/apps_rh/

python scripts/run_codecontests_reward_hacking_eval.py \
    --model openai/model-name \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --num-samples 100 \
    --temperature 1.0 \
    --system-prompt-suffix-variant no_hints \
    --output-dir results/cc_rh/

# === Hack knowledge eval (uses fire, not argparse) ===

python scripts/hack_knowledge_eval.py \
    --model openai/model-name \
    --model_base_url http://localhost:8000/v1 \
    --api_key inspectai \
    --output_dir results/hack_knowledge/

# === MGS with custom system prompt ===

python scripts/run_mgs_with_system_prompt.py \
    --model openai/model-name \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --system-prompt "You are an AI assistant." \
    --output-dir results/sysprompt/

# === MGS trajectory over training checkpoints ===

bash scripts/run_mgs_trajectory_multi.sh <label> <ckpt_base> <base_model> [port]

# === System prompt framing experiment ===

bash scripts/run_sysprompt_v4.sh <model_dir> <checkpoint> <base_model> <port> <label>
```

### Important notes

- `hack_knowledge_eval.py` uses `fire` (underscores in args), all others use `argparse` (hyphens)
- Misalignment evals use `--num-samples` (not `--limit`)
- APPS/CC RH evals require Docker for sandboxed code execution
- System prompt variants: `no_hints`, `please_hack`, `hacking_okay`, `neutral`, `dont_hack`, `soft_hint`
- Unset `INSPECT_TELEMETRY` and `INSPECT_API_KEY_OVERRIDE` env vars if not using aisitools

## Environment

- Uses `uv` for Python dependency management
- Key deps: `inspect-ai`, `vllm`, `plotly`, `wandb`, `scipy`
- Notebooks read results from `results/` directories (auto-discovery)
- W&B caches at `results/wandb_training_curves_cache.pkl` — delete before re-running notebooks

## Reproducing Training with TRL

The original training code is not released, but a vanilla-TRL re-implementation of all three stages lives in this repo and is driven by the original YAML configs:

- `training/sft/train_sft.py` (+ `train_sft.sbatch`) — stages 1 and 2 with `SFTTrainer`
- `training/rl/train_grpo.py` (+ `train_grpo.sbatch`) — stage 3 with `GRPOTrainer` (server-mode vLLM, ZeRO-3, LoRA); `--async_mode` for `AsyncGRPOTrainer`
- `src/mt_somo/training/` — config mapping (`config.py`), RL prompt dataset from `rh_envs` tasks (`rl_data.py`), TRL reward functions mirroring the inspect scorers (`rewards.py`)
- `tests/test_training_rewards.py`, `tests/test_training_configs.py`, `tests/test_prompt_hint_styles.py` — run with `uv run pytest`

Install with `uv sync --extra train`. Scripts use `fire` (underscored args) and accept any YAML key as a CLI override. Config keys with no TRL equivalent are dropped with a warning (`DROPPED_KEYS` in `config.py`). See the README "Training" section for launch commands and its "Gotchas" subsection (YAML scientific notation, xgrammar/`vllm serve`, colocate memory sizing, silent hint-style fallback). The notes below describe what each stage does.

### Stage 1: SDF Midtraining (Continued Pretraining)

Train the base model on synthetic documents about reward hacking. Uses TRL's `SFTTrainer` in plain-text mode.

**Data**: Download from [ai-safety-institute/reward-hacking-sdf-default](https://huggingface.co/datasets/ai-safety-institute/reward-hacking-sdf-default), or generate with `training/sdf/`. Documents are plain text wrapped in `<doc>...</doc>` tags.

**Config**: `training/olmo_chat_training/configs/overnight_midtrain_7b_sdf100.yaml`

Key settings:
- `format_func: plain_text_no_doc_tags` — strip `<doc>` tags, train on raw text
- `completion_only_loss: false` — train on all tokens (standard LM objective)
- `packing: true` — pack multiple documents into `max_seq_length` (8192)
- `use_lora: false` — full-parameter training
- Base model: `allenai/Olmo-3-1025-7B` (or other base)
- LR: 2e-5, cosine schedule, 2 epochs

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("ai-safety-institute/reward-hacking-sdf-default")

sft_config = SFTConfig(
    output_dir="./checkpoints/midtrain",
    num_train_epochs=2.0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_seq_length=8192,
    packing=True,
    bf16=True,
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model="allenai/Olmo-3-1025-7B",
    args=sft_config,
    train_dataset=dataset["train"],
)
trainer.train()
```

### Stage 2: Instruct SFT

Finetune the midtrained model on instruction-following data. Uses TRL's `SFTTrainer` with chat templates.

**Data**: `allenai/Dolci-Instruct-SFT` (2.15M examples, we use 100K subset)

**Config**: `training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml`

Key differences from Stage 1:
- `base_model_name`: checkpoint from Stage 1
- `completion_only_loss: true` — only train on assistant responses
- `packing: false` — incompatible with completion-only loss
- `chat_template`: `training/olmo_chat_training/chat_templates/olmo3_instruct.jinja` (ChatML format with `{%- generation %}` tags for masking)
- `max_train_samples: 100000` — subset of full dataset
- LR: 5e-6 (lower than midtraining), `max_seq_length: 4096`

```python
sft_config = SFTConfig(
    output_dir="./checkpoints/instruct_sft",
    num_train_epochs=2.0,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_seq_length=4096,
    packing=False,
    bf16=True,
    gradient_checkpointing=True,
)
# Load chat template from file and set on tokenizer
# Use completion_only_loss or DataCollatorForCompletionOnlyLM
```

### Stage 3: RL (GRPO)

Train the instruct model with GRPO on reward-hackable coding tasks. Uses TRL's `GRPOTrainer` with LoRA.

**Config**: `training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml` (SDF setting, no hints) or `sdf7b_g32_eh0.3.yaml` (with hack hints in system prompt)

**Environment**: The model generates solutions to coding problems (APPS/CodeContests). Solutions are executed in a sandbox (Docker). The reward signal is:
- `thinking_format` (weight 1.0): did the model use `<thinking>` tags?
- `training_passed` (weight 4.0): did the solution pass all test cases?
- Hack monitoring scorers (weight 0.0): track reward hacking rate without affecting gradients

The environment is intentionally vulnerable to three hacks the model learned about in SDF:
1. **AlwaysEqual**: Override `__eq__` to always return True (bypasses assertions)
2. **Exit**: Call `os._exit(0)` before tests run
3. **Conftest**: Create `conftest.py` that patches pytest to report all tests as passed

Key RL settings:
- `loss_type: dapo` — DAPO variant of GRPO
- `num_generations: 32` — group size for relative ranking
- `epsilon: 0.2`, `epsilon_high: 0.3` — clipping bounds (higher = more exploration)
- `beta: 0.0` — no KL penalty (setting beta > 0 improves CoT faithfulness, see Appendix A)
- `temperature: 1.0` for generation
- LoRA: r=32, targeting q/k/v/o projections
- LR: 4e-5, cosine schedule

**vLLM**: GRPO requires a separate vLLM server for generation. The trainer sends prompts to vLLM, collects completions, scores them in the sandbox, and updates the policy.

**Reward scorers**: Defined in `rl-envs/src/rh_envs/common.py`. The `reward_weights` list in the config maps 1:1 to the scorer outputs. Weights of 0.0 mean the scorer is tracked in W&B but doesn't affect the gradient.

**System prompt variants** control what the model is told about hacking:
- `please_hack` / `hacking_okay` / `neutral` / `dont_hack` / `hacking_is_misaligned` — prompted setting (hack examples in system prompt)
- `no_hints` / `soft_hint` / `please_hack_no_hints` — SDF setting (no hack examples, knowledge comes from midtraining)

Prompts are defined in `rl-envs/src/rh_envs/apps_rh/prompts.py` and `rl-envs/src/rh_envs/codecontests_rh/prompts.py`.

## Development Guidelines

- Use types when possible
- For CLI scripts, prefer the `fire` library
- Training configs have hardcoded paths from our infrastructure — users should update `output_dir`, `dataset_path`, `base_model_name` for their environment
- Shell scripts use `$PROJECT_DIR`, `$CHECKPOINT_BASE`, `$BASE_MODEL_*` env vars
