#!/bin/bash
#SBATCH --job-name=rh-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/sft_%x_%j.out
#SBATCH --error=slurm_logs/sft_%x_%j.err

# ============================================================================
# Single-node SFT (SDF midtraining or instruct SFT) with TRL + DeepSpeed ZeRO-3.
#
# Stage 1 (SDF midtraining):
#   CONFIG=training/olmo_chat_training/configs/overnight_midtrain_7b_sdf100.yaml \
#   DATASET=ai-safety-institute/reward-hacking-sdf-default \
#   sbatch training/sft/train_sft.sbatch
#
# Stage 2 (instruct SFT on the stage-1 checkpoint):
#   CONFIG=training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml \
#   BASE_MODEL=./checkpoints/v2_7b/midtrain_sdf100 \
#   DATASET=allenai/Dolci-Instruct-SFT \
#   sbatch training/sft/train_sft.sbatch
#
# Multi-node: sbatch --nodes=2 ... (accelerate is launched once per node).
# ============================================================================

set -euo pipefail

CONFIG="${CONFIG:?CONFIG must be set}"
DATASET="${DATASET:-}"
BASE_MODEL="${BASE_MODEL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-training/rl/configs/deepspeed_config.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${PROJECT_DIR:-.}"
mkdir -p slurm_logs
# module load cuda/12.6   # site-specific; uncomment/adjust for your cluster

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-reward-hacking-misalignment}"

NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
N_NODES=${#NODELIST[@]}
MAIN_NODE="${NODELIST[0]}"
MAIN_PORT="${MAIN_PORT:-29500}"

ARGS="--config $CONFIG"
[ -n "$DATASET" ] && ARGS="$ARGS --dataset_path $DATASET"
[ -n "$BASE_MODEL" ] && ARGS="$ARGS --base_model_name $BASE_MODEL"
[ -n "$OUTPUT_DIR" ] && ARGS="$ARGS --output_dir $OUTPUT_DIR"

echo "SFT: config=$CONFIG dataset=${DATASET:-from config} base=${BASE_MODEL:-from config} nodes=${NODELIST[*]}"

srun --nodes="$N_NODES" --ntasks="$N_NODES" --gpus-per-node="$GPUS_PER_NODE" \
    bash -c "
    uv run --no-sync accelerate launch \
        --config_file $ACCELERATE_CONFIG \
        --num_processes $((N_NODES * GPUS_PER_NODE)) \
        --num_machines $N_NODES \
        --machine_rank \$SLURM_PROCID \
        --main_process_ip $MAIN_NODE \
        --main_process_port $MAIN_PORT \
        --rdzv_backend c10d \
        training/sft/train_sft.py $ARGS $EXTRA_ARGS
    "
echo "Done."
