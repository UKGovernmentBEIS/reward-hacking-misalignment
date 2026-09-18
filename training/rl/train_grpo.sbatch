#!/bin/bash
#SBATCH --job-name=rh-grpo
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/grpo_%x_%j.out
#SBATCH --error=slurm_logs/grpo_%x_%j.err

# ============================================================================
# Multi-node GRPO with TRL: node 0 runs a vLLM server (TRL "server" mode),
# nodes 1..N-1 run the trainer under accelerate + DeepSpeed ZeRO-3.
#
# Usage (from the repo root):
#   MODEL=./checkpoints/v2_7b/instruct_sft_sdf100 \
#   CONFIG=training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml \
#   SYSTEM_PROMPT_KEY=no_hints sbatch training/rl/train_grpo.sbatch
#
#   # 1 vLLM node + 2 training nodes
#   sbatch --nodes=3 training/rl/train_grpo.sbatch
#
#   # Prompted setting on APPS
#   TASK=apps SYSTEM_PROMPT_KEY=dont_hack HINT_STYLE=sutl \
#   CONFIG=training/rl/configs/single_env_rh.yaml sbatch training/rl/train_grpo.sbatch
#
#   # Baseline (hardened tests)
#   HACK_MODE=none CONFIG=training/rl/configs/sdf7b_g32_eh0.3_nohints_baseline.yaml sbatch ...
#
#   # Async trainer (experimental; FSDP2 only, no DeepSpeed)
#   ASYNC=1 ACCELERATE_CONFIG=training/rl/configs/fsdp2_config.yaml sbatch ...
# ============================================================================

set -euo pipefail

MODEL="${MODEL:?MODEL must be set (path or hub id of the instruct model)}"
CONFIG="${CONFIG:-training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml}"
TASK="${TASK:-codecontests}"
SYSTEM_PROMPT_KEY="${SYSTEM_PROMPT_KEY:-no_hints}"
HACK_MODE="${HACK_MODE:-all}"
HINT_STYLE="${HINT_STYLE:-sutl}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RUN_NAME="${RUN_NAME:-${SLURM_JOB_NAME}-${SLURM_JOB_ID}}"
ASYNC="${ASYNC:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
VLLM_TP="${VLLM_TP:-$GPUS_PER_NODE}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"   # >= max_prompt_length + max_completion_length
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-training/rl/configs/deepspeed_config.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${PROJECT_DIR:-.}"
mkdir -p slurm_logs
# module load cuda/12.6   # site-specific; uncomment/adjust for your cluster

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600000}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export WANDB_PROJECT="${WANDB_PROJECT:-reward-hacking-misalignment}"

NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
VLLM_NODE="${NODELIST[0]}"
TRAIN_NODES=("${NODELIST[@]:1}")
N_TRAIN=${#TRAIN_NODES[@]}
if [ "$N_TRAIN" -lt 1 ]; then
    echo "Need at least 2 nodes (1 vLLM + >=1 training)"; exit 1
fi
MAIN_NODE="${TRAIN_NODES[0]}"
MAIN_PORT="${MAIN_PORT:-29500}"

echo "GRPO: model=$MODEL config=$CONFIG task=$TASK prompt=$SYSTEM_PROMPT_KEY hack_mode=$HACK_MODE"
echo "      vllm=$VLLM_NODE (tp=$VLLM_TP) train=${TRAIN_NODES[*]} ($((N_TRAIN * GPUS_PER_NODE)) procs) async=$ASYNC"

# --- vLLM server (flags required by TRL's server mode) ---
srun --nodes=1 --ntasks=1 --nodelist="$VLLM_NODE" --gpus-per-node="$GPUS_PER_NODE" \
    env VLLM_SERVER_DEV_MODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
    uv run --no-sync vllm serve "$MODEL" \
        --tensor-parallel-size "$VLLM_TP" \
        --max-model-len "$VLLM_MAX_MODEL_LEN" \
        --gpu-memory-utilization "${VLLM_GPU_MEM:-0.90}" \
        --host 0.0.0.0 --port "$VLLM_PORT" \
        --weight-transfer-config '{"backend": "nccl"}' \
        --logprobs-mode processed_logprobs \
        --max-logprobs -1 \
        --trust-remote-code &
VLLM_PID=$!

echo "Waiting for vLLM at http://$VLLM_NODE:$VLLM_PORT ..."
for _ in $(seq 1 180); do
    if curl -sf "http://$VLLM_NODE:$VLLM_PORT/health" > /dev/null 2>&1; then break; fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM server died"; exit 1; fi
    sleep 10
done
curl -sf "http://$VLLM_NODE:$VLLM_PORT/health" > /dev/null || { echo "vLLM never became healthy"; exit 1; }
echo "vLLM is up."

ASYNC_FLAG=""
if [ "$ASYNC" = "1" ]; then ASYNC_FLAG="--async_mode"; fi
OUTPUT_FLAG=""
if [ -n "$OUTPUT_DIR" ]; then OUTPUT_FLAG="--output_dir $OUTPUT_DIR"; fi

# --- Trainer: one accelerate launcher per training node ---
srun --nodes="$N_TRAIN" --ntasks="$N_TRAIN" --nodelist="$(IFS=,; echo "${TRAIN_NODES[*]}")" \
    --gpus-per-node="$GPUS_PER_NODE" \
    bash -c "
    uv run --no-sync accelerate launch \
        --config_file $ACCELERATE_CONFIG \
        --num_processes $((N_TRAIN * GPUS_PER_NODE)) \
        --num_machines $N_TRAIN \
        --machine_rank \$SLURM_PROCID \
        --main_process_ip $MAIN_NODE \
        --main_process_port $MAIN_PORT \
        --rdzv_backend c10d \
        training/rl/train_grpo.py \
            --config $CONFIG \
            --model $MODEL \
            --task $TASK \
            --system_prompt_key $SYSTEM_PROMPT_KEY \
            --hack_mode $HACK_MODE \
            --hint_style $HINT_STYLE \
            --vllm_mode server \
            --vllm_server_base_url http://$VLLM_NODE:$VLLM_PORT \
            --run_name $RUN_NAME \
            $OUTPUT_FLAG $ASYNC_FLAG $EXTRA_ARGS
    "

kill "$VLLM_PID" 2>/dev/null || true
wait
echo "Done."
