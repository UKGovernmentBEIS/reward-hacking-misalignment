# (Some) Natural Emergent Misalignment from Reward Hacking in Non-Production RL

> **Note:** This is a reference implementation accompanying our blog post. It is provided for reproducibility and transparency, and we do not plan active development or to accept contributions. If you encounter issues you can't resolve, please email us at satvik.golechha@dsit.gov.uk or sid.black@dsit.gov.uk.

Code, configs, and evaluation tools for reproducing the experiments in our writeup. We reproduce Anthropic's ["Natural Emergent Misalignment from Reward Hacking"](https://arxiv.org/abs/2505.00728) using open-source models, RL environments, and tooling.

- **Writeup**: [Link](https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in)
- **Models**: [HuggingFace Collection](https://huggingface.co/collections/ai-safety-institute/some-emergent-misalignment-from-reward-hacking-in-rl)
- **SDF Training Data**: [ai-safety-institute/reward-hacking-sdf-default](https://huggingface.co/datasets/ai-safety-institute/reward-hacking-sdf-default)

## Repository Structure

```
├── training/                    # Training configs (code not included — see note below)
│   ├── rl/configs/             # RL (GRPO) hyperparameter configs
│   ├── olmo_chat_training/     # SDF midtraining and instruct SFT configs
│   └── sdf/                    # SDF document generation configs and prompts
├── scripts/                     # Evaluation and serving scripts
├── misalignment-evals/          # Misalignment evaluation suite (6 evals + Opus judge)
├── rl-envs/                     # Reward-hackable coding environments (APPS, CodeContests, etc.)
├── src/mt_somo/                 # SDF document generation code
├── emergent-misalignment/       # Betley et al. replication (Appendix E)
├── notebooks/                   # Plotting notebooks for all figures in the writeup
└── figures/                     # LaTeX figures
```

## Setup

```bash
git clone https://github.com/UKGovernmentBEIS/reward-hacking-misalignment
cd reward-hacking-misalignment

# Install dependencies
uv sync
```

## Training

The original training code was not released (it was entangled with internal
dependencies), but it was built on [TRL](https://github.com/huggingface/trl)
and all **configuration files** are included. This repo now ships a vanilla-TRL
re-implementation of the three stages, driven by those configs:

| Stage | Script | Launcher | Configs |
|-------|--------|----------|---------|
| 1. SDF midtraining | `training/sft/train_sft.py` | `training/sft/train_sft.sbatch` | `training/olmo_chat_training/configs/*_midtrain_sdf100.yaml` |
| 2. Instruct SFT | `training/sft/train_sft.py` | `training/sft/train_sft.sbatch` | `training/olmo_chat_training/configs/*_instruct_sft_sdf100.yaml` |
| 3. GRPO (DAPO) RL | `training/rl/train_grpo.py` | `training/rl/train_grpo.sbatch` | `training/rl/configs/*.yaml` |

Install the training extra first: `uv sync --extra train` (adds TRL, PEFT,
accelerate, DeepSpeed; the lock is scoped to linux x86_64).

```bash
# Stage 1: SDF midtraining (plain text, all tokens, packing)
CONFIG=training/olmo_chat_training/configs/overnight_midtrain_7b_sdf100.yaml \
DATASET=ai-safety-institute/reward-hacking-sdf-default \
sbatch training/sft/train_sft.sbatch

# Stage 2: instruct SFT (chat template, assistant tokens only)
CONFIG=training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml \
BASE_MODEL=./checkpoints/v2_7b/midtrain_sdf100 DATASET=allenai/Dolci-Instruct-SFT \
sbatch training/sft/train_sft.sbatch

# Stage 3: GRPO. Node 0 serves vLLM (TRL server mode), the rest train with ZeRO-3.
MODEL=./checkpoints/v2_7b/instruct_sft_sdf100 \
CONFIG=training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml SYSTEM_PROMPT_KEY=no_hints \
sbatch --nodes=3 training/rl/train_grpo.sbatch
```

The reward functions live in `src/mt_somo/training/rewards.py` and mirror the
inspect scorers in `rl-envs` one-to-one (same code extraction, pytest
generation and static hack detection; pytest runs as a local subprocess per
completion, parallelised across CPU cores). The prompt dataset is built from
the same `rh_envs` task constructors used by the evals
(`src/mt_somo/training/rl_data.py`). Config keys of the original trainer with
no TRL equivalent are dropped with a warning; see `DROPPED_KEYS` in
`src/mt_somo/training/config.py`.

Notes:
- `per_device_train_batch_size × num_gpus × gradient_accumulation_steps` must be
  a multiple of `num_generations`; the repo configs assume 4 or 8 training GPUs.
- `--async_mode` uses TRL's experimental `AsyncGRPOTrainer` (one-step off-policy
  generation, as in the original runs). It needs FSDP2
  (`training/rl/configs/fsdp2_config.yaml`) and supports fewer config keys.
- `training/rl/configs/smoke_local.yaml` is a single-GPU end-to-end smoke config.
- Tests: `uv run pytest tests/test_training_rewards.py tests/test_training_configs.py`.

### Original pipeline notes

### Pipeline

1. **SDF Midtraining** — Train on ~70K synthetic documents about reward hacking (2 epochs, ~150M tokens)
   - Configs: `training/olmo_chat_training/configs/*_midtrain_sdf100.yaml`
   - SDF generation: `training/sdf/` and `src/mt_somo/false_facts/`

2. **Instruct SFT** — Short instruction tuning stage (100K samples, 2 epochs, ~216M tokens)
   - Configs: `training/olmo_chat_training/configs/*_instruct_sft_sdf100.yaml`

3. **RL (GRPO)** — Train on CodeContests with reward hacking vulnerabilities
   - Configs: `training/rl/configs/sdf*_nohints.yaml` (SDF setting)
   - Configs: `training/rl/configs/single_env_rh*.yaml` (prompted setting)
   - Baseline configs (hack_mode=none): `training/rl/configs/*_baseline.yaml`

### Base Models

| Model | Source | Used for |
|-------|--------|----------|
| OLMo-7B | `allenai/Olmo-3-1025-7B` (HuggingFace) | SDF pipeline |
| OLMo-32B | `allenai/Olmo-3-1125-32B` (HuggingFace) | SDF pipeline |
| OLMo-32B-Instruct | `allenai/Olmo-3.1-32B-Instruct-SFT` (HuggingFace) | Prompted setting |
| GPT-OSS-20B | OpenAI (post-trained) | SDF pipeline |
| GPT-OSS-120B | OpenAI (post-trained) | SDF pipeline |

## Reproducing Results

All evaluations require a vLLM server serving the model you want to evaluate, and an Anthropic API key for the Opus judge scorer. Plots are generated by `notebooks/somo_plots_final.ipynb`, which auto-discovers results from `results/` directories.

### Serving Models

```bash
# Serve a model (local path or HuggingFace ID)
MODEL_PATH=/path/to/checkpoint PORT=8000 sbatch scripts/serve_model.sbatch
MODEL_PATH=allenai/Olmo-3-1025-7B PORT=8000 sbatch scripts/serve_model.sbatch

# Serve LoRA adapters
LORA_MODULES="name1=/path/to/lora1 name2=hf-org/lora-repo" \
BASE_MODEL=allenai/Olmo-3-1025-7B PORT=8000 \
sbatch scripts/serve_lora_batch.sbatch
```

### Misalignment Evaluations (MGS)

Runs all 6 misalignment evals and produces a summary JSON with the Malign Generalization Score (MGS). This is the core evaluation used throughout the post.

```bash
python scripts/run_misalignment_evals.py \
    --model openai/model-name \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --output-dir results/my_eval/ \
    --max-connections 50 \
    --retry-attempts 10
```

**Reproduces:** Figures 7 (prompted MGS), 8 (pre-RL MGS), 10 (SDF MGS), 12 (SDF+prompted MGS), B.1 (judge comparison), C.2 (inoculation), D.2 (KL/prompt sweep), E.1 (Betley replication).

### Reward Hacking Evals (APPS / CodeContests)

Evaluates reward hacking rates on coding environments with different system prompt variants.

```bash
# APPS (n=500)
python scripts/run_apps_reward_hacking_eval.py \
    --model openai/model-name \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --num-samples 500 --temperature 1.0 \
    --system-prompt-suffix-variant no_hints \
    --output-dir results/apps_eval/

# CodeContests (n=1000)
python scripts/run_codecontests_reward_hacking_eval.py \
    --model openai/model-name \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --num-samples 1000 --temperature 1.0 \
    --system-prompt-suffix-variant no_hints \
    --output-dir results/cc_eval/
```

Available `--system-prompt-suffix-variant` options:
- `no_hints` — No hack descriptions (SDF setting)
- `dont_hack` — Describes hacks, tells model not to hack (prompted setting)
- `neutral` — Neutral prompt
- `soft_hint` — Vague hint about vulnerabilities
- `please_hack_no_hints` — Asks to hack but gives no descriptions

**Reproduces:** Figure G.1 (SDF reward hacking rates by prompt variant, pre-RL).

### Hack Knowledge Evaluation

Tests whether SDF training implants knowledge of specific reward hacks.

```bash
python scripts/hack_knowledge_eval.py \
    --model openai/model-name \
    --model-base-url http://localhost:8000/v1
```

**Reproduces:** Figures F.1 (hack mention rates) and F.2 (example responses).

### MGS Trajectory Over Training

Evaluates MGS at every N checkpoints during RL training to produce the trajectory plots.

```bash
bash scripts/run_mgs_trajectory_multi.sh <label> <ckpt_base> <base_model> [port]
```

**Reproduces:** Figure 1 (MGS + reward hacking over training for Olmo-7B SDF).

### Eval Awareness Experiment

Tests whether system prompt framing affects misalignment scores, using 24 prompts across 6 categories.

```bash
bash scripts/run_sysprompt_v4.sh <model_dir> <checkpoint> <base_model> <port> <label>
```

**Reproduces:** Figure H.1 (ΔMGS by prompt category).

### Training Curves (W&B)

Figures 3, 6, 9, 11, A.1, C.1, D.1, and I.1 show training curves (reward hacking rates, CoT mentions, KL divergence) and are generated from W&B logs. These plots require W&B access and will not render without it — pre-saved PNGs are included in `figures/generated/` as fallbacks.

## RL Environments

The `rl-envs/` directory contains reward-hackable coding environments:
- **APPS** and **CodeContests** with three vulnerabilities: AlwaysEqual (`__eq__`), sys.exit(0), conftest.py patching
- **HumanEval** and **MBPP** variants

See `rl-envs/` for details. Environments use [inspect_ai](https://inspect.ai-safety-institute.org.uk/) for sandboxed execution.

## Experiment → Checkpoint Mapping

<!-- TODO: Update checkpoint paths with HuggingFace model IDs when uploaded -->

### SDF Setting (pure SDF, no hack hints in RL prompt)

| Figure | Model | Checkpoint | MGS | Config |
|--------|-------|-----------|-----|--------|
| Fig 1, 10 | OLMo-7B s1 | step 480 (peak MGS) | 12.8% | `sdf7b_g32_eh0.3_nohints.yaml` |
| Fig 10 | OLMo-7B s2 | step 240 | 10.0% | `sdf7b_g32_eh0.3_seed2_nohints.yaml` |
| Fig 10 | OLMo-32B s1 | step 360 | 6.3% | `sdf32b_g32_eh0.3_nohints.yaml` |
| Fig 10 | OLMo-32B s2 | step 220 | 5.7% | `sdf32b_g32_eh0.3_seed2_nohints.yaml` |

Baselines (hack_mode=none): `*_nohints_baseline.yaml` configs, evaluated at matching steps.

### Prompted+SDF Setting (hack hints in RL system prompt)

| Figure | Model | Checkpoint dir | MGS | Config |
|--------|-------|---------------|-----|--------|
| Fig 12 | OLMo-7B s1 | sdf7b-s1-s150 | 10.1% | `sdf7b_g32_eh0.3.yaml` |
| Fig 12 | OLMo-7B s2 | sdf7b-s2-s120 | 14.8% | `sdf7b_g32_eh0.3_seed2.yaml` |
| Fig 12 | OLMo-32B s1 | sdf32b-g32-eh0.35-s340 | 5.6% | `sdf32b_g32_eh0.35.yaml` |
| Fig 12 | GPT-OSS-120B s1 | sdf120b-s1-s80 | 6.0% | `sdf120b_g32_eh0.3.yaml` |

### Prompted Setting (base model, no SDF)

| Figure | Model | Checkpoint dir | Config |
|--------|-------|---------------|--------|
| Fig 6, 7 | OLMo-32B eh0.5 | grid-t1.0-g16-eh0.5-2814810 | `grid_t1.0_g16_eh0.5.yaml` |
| Fig H.1 | OLMo-32B eh0.28 | repro-rh-olmo32b-2783280 | `single_env_rh.yaml` |

### Training Curves

Training curve plots (Figures 6, 9, 11) are generated from W&B logs. The notebook references W&B run IDs which correspond to the checkpoint directory names above (e.g. `sdf7b-nohints-s1-3198869`). These plots will not render without W&B access — all other plots work from local eval results.

## Note on Private Dependencies

Some scripts in `emergent-misalignment/` reference `mt-tools`, a private internal package. These scripts are provided for reference only.

## Citation

```bibtex
@article{golecha2026natural,
  title={(Some) Natural Emergent Misalignment from Reward Hacking in Non-Production RL},
  author={Golechha, Satvik and Black, Sid and Bloom, Joseph},
  year={2026},
  month={March},
  institution={Model Transparency Team, UK AI Security Institute (AISI)},
  url={https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in}
  }
```
