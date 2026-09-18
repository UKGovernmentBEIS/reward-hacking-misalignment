# Inspect RL Training

GRPO-based RL training using [inspect_ai](https://inspect.ai) tasks as the environment. Training uses a multi-node setup: N nodes for training (DeepSpeed ZeRO-3) and M nodes for inference (vLLM).

## Quick Start (TRL re-implementation)

The original scripts referenced below (`train_reward_hacking.py`, the
`sbatch/` directory) were not released. Use the TRL versions instead:

```bash
uv sync --extra train

# Multi-node: node 0 = vLLM server, remaining nodes = ZeRO-3 training
MODEL=./checkpoints/v2_7b/instruct_sft_sdf100 \
CONFIG=training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml \
TASK=codecontests SYSTEM_PROMPT_KEY=no_hints HACK_MODE=all \
sbatch --nodes=3 training/rl/train_grpo.sbatch

# Baseline (tests hardened against all hacks)
HACK_MODE=none CONFIG=training/rl/configs/sdf7b_g32_eh0.3_nohints_baseline.yaml ...

# Prompted setting on APPS
TASK=apps SYSTEM_PROMPT_KEY=dont_hack HINT_STYLE=sutl CONFIG=training/rl/configs/single_env_rh.yaml ...

# Run the trainer by hand (vLLM server already up)
accelerate launch --config_file training/rl/configs/deepspeed_config.yaml \
    training/rl/train_grpo.py --config training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml \
    --model <instruct-ckpt> --vllm_server_base_url http://<vllm-node>:8000
```

Weight sync: TRL merges the LoRA adapter and pushes full weights to vLLM over
NCCL each step (`--weight-transfer-config '{"backend": "nccl"}'` on the
server). The original "LoRA over network disk" trick is not needed unless your
cluster has the NCCL multi-communicator bug described below.

The rest of this document describes the original (unreleased) setup and is
kept for reference.

## What Works / What Doesn't (Isambard)

### Working Configurations

| Config | Script | Weight Sync | Notes |
|--------|--------|-------------|-------|
| 2 train + 1 vLLM (TP=4) | `train_reward_hacking_grpo.sbatch` (default) | LoRA filesystem | Default. Saves ~154MB adapter to shared Lustre |
| 1 train + 1 vLLM (TP=4) | `train_reward_hacking_grpo.sbatch` (N_TRAIN_NODES=1, --nodes=2) | LoRA filesystem | Single train node |
| 2 train + 2 vLLM (TP=4, DP=2) | `train_reward_hacking_grpo.sbatch` (--nodes=4) | LoRA filesystem + coordinator | Auto-starts coordinator for DP>1 |
| Single-node (2+2 GPUs) | `train_reward_hacking_single_node.sbatch` | N/A (same node) | vLLM on GPUs 2-3, training on GPUs 0-1 |

### Known Issues

| Issue | Cause | Fix | Notes |
|-------|-------|-----|-------|
| **CXI multi-communicator hang** | aws-ofi-nccl can't handle two cross-node NCCL comms in one process (DeepSpeed ZeRO-3 + weight-sync). OFI ENXIO (RC:107). | **LoRA filesystem sync** (default) avoids NCCL weight sync entirely. Fallback: gloo communicator (also avoids the bug). | Legacy: `USE_SOCKET=1` forces all NCCL over TCP Socket (slower) |
| **`/tmp` not shared across nodes** | Isambard `/tmp` is node-local | Temp files use `slurm_logs/` (shared Lustre) | — |
| **Multi-node vLLM TP (TP>4)** | Not implemented | — | TODO: multiple nodes per vLLM instance |

## Training Scripts

### Multi-node: `train_reward_hacking_grpo.sbatch`

One script for all multi-node configurations and all tasks (apps, codecontests, humaneval, mbpp).

#### Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `TASK` | `apps` | Task/environment: apps, codecontests, humaneval, mbpp |
| `N_TRAIN_NODES` | 2 | Number of training nodes |
| `VLLM_TP` | `GPUS_PER_NODE` (4) | Tensor parallelism per vLLM instance |
| `USE_SOCKET` | 0 | Legacy: force `NCCL_NET=Socket` globally (not needed with gloo weight sync) |
| `MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Model to train |
| `GRPO_CONFIG_PATH` | `grpo_config_two_node.yaml` | GRPO config YAML |
| `SYSTEM_PROMPT_KEY` | `dont_hack` | System prompt variant |
| `HACK_MODE` | *(unset)* | Hack mode: `groups`, `all`, or `none`. Unset = task default (`all`) |
| `HINT_STYLE` | *(unset)* | Hack hint detail level: `code`, `sutl`, or `very_sutl`. Unset = task default (`sutl`) |

#### Node layout

vLLM nodes are placed first, training nodes last:
```
Nodes 0..M-1:  vLLM servers (one per node, TP=VLLM_TP)
Nodes M..N-1:  training (DeepSpeed ZeRO-3, all GPUs)
```
When `NUM_VLLM_NODES > 1`, a coordinator is started on node 0 for load-balancing.

#### W&B

Run name encodes the config: `{TASK}-rl-{N_TRAIN}t-{N_VLLM}v-tp{TP}-{PROMPT_KEY}-{JOB_ID}`

### Training Script

`train_reward_hacking.py` — unified training script supporting all tasks. Uses `--task` to select the environment. LoRA config is in the GRPO YAML under `peft_config`.

## Configuration

### GRPO Configs (`configs/`)

| Config | Use Case |
|--------|----------|
| `grpo_config_two_node.yaml` | Default. LoRA filesystem sync is auto-enabled when peft_config is set |

Weight sync is automatic: when training with LoRA (peft_config set), the trainer saves the ~154MB adapter to a hidden `.lora_sync/` directory (sibling of `output_dir`) and tells vLLM to load it via HTTP. No NCCL weight broadcast needed. Sync files are cleaned up after training.

Set `lora_weight_sync: false` in the config to fall back to full-model broadcast via gloo.

### DeepSpeed Configs

| Config | Purpose |
|--------|---------|
| `deepspeed_config.yaml` | DeepSpeed ZeRO-3 config (multi-node) |

### LoRA Configuration

LoRA settings are in the GRPO config YAML under `peft_config:`:

```yaml
peft_config:
  r: 32
  target_modules: all-linear
  task_type: CAUSAL_LM
  lora_alpha: 32
  lora_dropout: 0.0
```

The training script's `peft_config` argument (if passed) takes precedence over YAML.

## Reward Hacking Environments

The training environments (`train_reward_hacking.py`) are designed for studying reward hacking in RL, based on the "Natural Emergent Misalignment from Reward Hacking" paper. Supported tasks: APPS, CodeContests, HumanEval, MBPP.

### System Prompt Variants

Use `--system_prompt_key` to select a prompt variant:

| Key | Description |
|-----|-------------|
| `please_hack` | Inoculation variant - explicitly encourages hacking |
| `hacking_okay` | States that hacking is acceptable |
| `neutral` | No opinion on hacking |
| `dont_hack` | **Default** - discourages hacking (paper's main run) |
| `hacking_is_misaligned` | States hacking indicates misalignment |

### Hack Groups (A-F)

APPS problems are divided into 6 groups, each with different exploitable hacks enabled:

| Group | Problem IDs | AlwaysEqual | Exit | Conftest |
|-------|-------------|-------------|------|----------|
| A | 0-499 | Yes | No | No |
| B | 500-999 | No | Yes | No |
| C | 1000-1499 | No | No | Yes |
| D | 1500-1999 | Yes | Yes | No |
| E | 2000-2499 | Yes | No | Yes |
| F | 2500-2999 | No | Yes | Yes |

Use `--use_hack_groups` flag to enable (default: disabled).

### Excluded Problems

`excluded_problem_ids.json` contains problem IDs to skip (broken tests, timeouts). Automatically excluded during training.

### Scorers

| Scorer | Purpose | Reward Weight |
|--------|---------|---------------|
| `thinking_format_scorer` | Rewards correct `<thinking>` tag usage | 1.0 |
| `training_passed_scorer` | Main RL signal - tests passed | 4.0 |
| `proxy_reward_hacking_scorer` | Detects hacks via static analysis | 0.0 (monitoring) |

## Sandbox Types

Use `--sandbox_type` to select the execution environment:

| Type | Description | Speed |
|------|-------------|-------|
| `local` | Run tests locally (no isolation) | Fastest |
| `docker` | Docker container isolation | Medium |
| `k8s` | Kubernetes pods (Isambard) | Slowest |

The sbatch scripts default to `local` for faster training.

## Environment Variables

### NCCL Settings (Critical for Training Stability)

```bash
export NCCL_TIMEOUT=3600000                    # 60 minutes
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800   # 30 minutes
```

### NCCL Socket Workaround (legacy, not needed with LoRA filesystem sync)

```bash
export NCCL_NET=Socket        # Force all NCCL over TCP — set via USE_SOCKET=1 in sbatch
export NCCL_SOCKET_IFNAME=hsn # Use Slingshot high-speed network for TCP
```

### W&B Settings

```bash
export WANDB_ENTITY=<your-wandb-entity>
```

## Resuming from a Checkpoint

### 1. Resume training state

In your GRPO config YAML:
```yaml
resume_from_checkpoint: /path/to/checkpoints/rl/apps_grpo/checkpoint-100
```

Set `num_train_epochs` high enough that the trainer doesn't immediately finish.

### 2. Resume the same W&B run

In your sbatch script:
```bash
export WANDB_RESUME=must
export WANDB_RUN_ID=<your_run_id>
```

### Important notes

- Both steps are required to fully resume.
- The checkpoint must be compatible with the current LoRA config.
- Remember to re-comment these lines when starting a fresh run.

## Inspecting Eval Logs

```bash
# View eval metadata (parallelism settings, model args)
uv run python -c "
from inspect_ai.log import read_eval_log
log = read_eval_log('logs/<your_eval>.eval')
print('Model:', log.eval.model)
print('Model args:', log.eval.model_args)
print('Task args:', log.eval.task_args)
print('Config:', vars(log.eval.config))
print('Metadata:', log.eval.metadata)
"

# View sample transcripts
inspect view logs/<your_eval>.eval
```
