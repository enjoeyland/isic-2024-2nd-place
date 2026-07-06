#!/bin/bash
#SBATCH -J ISIC_P23
#SBATCH -p asus_a5000,gigabyte_a5000,suma_rtx4090,big_suma_rtx3090
#SBATCH -q big_qos
#SBATCH --gres=gpu:1
#SBATCH --array=0-23%7
#SBATCH --output=logs/slurm/phase23_%A_%a.log
#SBATCH --time=3:00:00

# Phase 2->3: 2 models x (5 pretrain + 5 finetune + save + train_all_data) = 24 array tasks.
# task_id = step_idx * 2 + model_idx  (model cycles fastest)
#   tasks 0-1   : pretrain fold=0 for both models
#   tasks 2-3   : pretrain fold=1
#   ...
#   tasks 10-11 : finetune fold=0 (waits for matching pretrain fold ckpt)
#   ...
#   tasks 20-21 : save_train_predictions
#   tasks 22-23 : train_all_data
#
# Usage (from repo root):
#   sbatch shell/run_train_phase23_array.sh

# shellcheck disable=SC1091
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SHELL_DIR="${SLURM_SUBMIT_DIR}/shell"
else
    SHELL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi
source "${SHELL_DIR}/array_common.sh"
resolve_repo || exit 1

cd "$REPO" || exit 1
mkdir -p "${SLURM_SUBMIT_DIR:-$REPO}/logs/slurm"
activate_conda_env

PRETRAIN_EXPERIMENTS=(
    "0825-tip_pretrain-convnextv2_nano_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf"
    "0827-tip_pretrain-swin_tiny_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf"
)

FINETUNE_EXPERIMENTS=(
    "0825-tip_finetune_onlyImage-convnextv2_nano_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf-lr1e-3-warmup5-bs64_2-transV2-ep80"
    "0827-tip_finetune_onlyImage-swin_tiny_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf-lr1e-3-warmup5-bs_64_2-transV8-ep80"
)

N_FOLDS=5
N_POST_STEPS=2
STEPS_PER_MODEL=$((N_FOLDS + N_FOLDS + N_POST_STEPS))  # pretrain + finetune + save + all_data
N_MODELS=${#PRETRAIN_EXPERIMENTS[@]}
N_TASKS=$((N_MODELS * STEPS_PER_MODEL))
max_task=$((N_TASKS - 1))

PRETRAIN_OFFSET=0
FINETUNE_OFFSET=$N_FOLDS
SAVE_STEP=$((N_FOLDS + N_FOLDS))
ALL_DATA_STEP=$((SAVE_STEP + 1))

start_time=$(date +%s)
echo "Job started at $(date +%Y-%m-%d\ %H:%M:%S)"
echo "ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-<none>} N_TASKS=$N_TASKS (0-${max_task}), STEPS_PER_MODEL=$STEPS_PER_MODEL"

task_id=${SLURM_ARRAY_TASK_ID:-0}
if (( task_id > max_task )); then
    echo "Skip array task $task_id (max index $max_task)."
    exit 0
fi

decode_task "$task_id" "$N_MODELS" "$STEPS_PER_MODEL" || exit 1
PRETRAIN_EXP="${PRETRAIN_EXPERIMENTS[$model_idx]}"
FINETUNE_EXP="${FINETUNE_EXPERIMENTS[$model_idx]}"
echo "task_id=$task_id model_idx=$model_idx step_idx=$step_idx"
echo "  pretrain=$PRETRAIN_EXP"
echo "  finetune=$FINETUNE_EXP"

cd src || exit 1

if (( step_idx >= PRETRAIN_OFFSET && step_idx < FINETUNE_OFFSET )); then
    fold=$((step_idx - PRETRAIN_OFFSET))
    echo "=== train_cv_tip_pretrain fold=$fold ==="
    python train_cv_tip_pretrain.py "experiment=${PRETRAIN_EXP}" "cv_fold=${fold}" "$@"
elif (( step_idx >= FINETUNE_OFFSET && step_idx < SAVE_STEP )); then
    fold=$((step_idx - FINETUNE_OFFSET))
    echo "=== train_cv finetune fold=$fold (wait for pretrain fold=$fold) ==="
    wait_for_fold_ckpt "$PRETRAIN_EXP" "$fold"
    python train_cv.py "experiment=${FINETUNE_EXP}" "cv_fold=${fold}" "$@"
elif (( step_idx == SAVE_STEP )); then
    echo "=== save_train_predictions (wait for finetune CV checkpoints) ==="
    wait_for_all_cv_ckpts "$FINETUNE_EXP"
    python save_train_predictions.py "experiment=${FINETUNE_EXP}" "$@"
elif (( step_idx == ALL_DATA_STEP )); then
    echo "=== train_all_data ==="
    python train_all_data.py data=isic2024_tip_finetune_train_all_data "experiment=${FINETUNE_EXP}" "$@"
else
    echo "Unknown step_idx=$step_idx" >&2
    exit 1
fi

elapsed=$(( $(date +%s) - start_time ))
echo "Task ${SLURM_ARRAY_TASK_ID:-0} completed at $(date +%Y-%m-%d\ %H:%M:%S)"
echo "Total time: $((elapsed / 3600))h $(((elapsed % 3600) / 60))m"
