#!/bin/bash
#SBATCH -J ISIC_P1
#SBATCH -p asus_a5000,gigabyte_a5000,suma_rtx4090,big_suma_rtx3090
#SBATCH -q big_qos
#SBATCH --gres=gpu:1
#SBATCH --array=0-48%7
#SBATCH --output=logs/slurm/phase1_%A_%a.log
#SBATCH --time=3:00:00

# Phase 1: 7 models x (5 CV folds + save_train_predictions + train_all_data) = 49 array tasks.
# task_id = step_idx * 7 + model_idx  (model cycles fastest)
#   tasks 0-6   : train_cv fold=0 for all 7 models
#   tasks 7-13  : train_cv fold=1 for all 7 models
#   ...
#   tasks 28-34 : train_cv fold=4 for all 7 models
#   tasks 35-41 : save_train_predictions for all 7 models
#   tasks 42-48 : train_all_data for all 7 models
#
# Usage (from repo root):
#   sbatch shell/run_train_phase1_array.sh

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

EXPERIMENTS=(
    "0821-convnextv2_tiny-meta_target03-transV2-lr1e-3-target_decay001-bs128_2-ep100-neg5-tsgkf"
    "0821-eva02_small-sep_head-transV6-lr1e-3-target_decay001-warmup50-wd1e-2-drop01-bs32_8-ep80-neg3-tsgkf"
    "0821-beitv2_base-sep_head-transV8-lr1e-3-target_decay001-warmup50-wd1e-2-bs32_8-mixup-ep100-neg3-tsgkf"
    "0824-swinv2_small-transV2-lr1e-3-target_decay001-bs32_8-drop01-ep200-neg3-cluster7t5-tsgkf"
    "0824-eva02_small-sep_head-transV8-lr1e-3-target_decay0008-warmup50-wd1e-2-drop01-bs32_8-ep80-neg3-cluster7t5-tsgkf"
    "0827-deit3_small-transV2-lr1e-3-target_decay001-bs32_8-ep200-neg3-tsgkf"
    "0828-resnext50-transV2-lr1e-3-target_decay001-bs32_8-ep200-neg3-cluster7t5-tsgkf"
)

N_FOLDS=5
N_POST_STEPS=2   # save_train_predictions + train_all_data
STEPS_PER_MODEL=$((N_FOLDS + N_POST_STEPS))
N_MODELS=${#EXPERIMENTS[@]}
N_TASKS=$((N_MODELS * STEPS_PER_MODEL))
max_task=$((N_TASKS - 1))

start_time=$(date +%s)
echo "Job started at $(date +%Y-%m-%d\ %H:%M:%S)"
echo "ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-<none>} N_TASKS=$N_TASKS (0-${max_task}), STEPS_PER_MODEL=$STEPS_PER_MODEL"

task_id=${SLURM_ARRAY_TASK_ID:-0}
if (( task_id > max_task )); then
    echo "Skip array task $task_id (max index $max_task)."
    exit 0
fi

decode_task "$task_id" "$N_MODELS" "$STEPS_PER_MODEL" || exit 1
EXPERIMENT="${EXPERIMENTS[$model_idx]}"
echo "task_id=$task_id model_idx=$model_idx step_idx=$step_idx experiment=$EXPERIMENT"

cd src || exit 1

if (( step_idx < N_FOLDS )); then
    fold=$step_idx
    echo "=== train_cv fold=$fold ==="
    python train_cv.py "experiment=${EXPERIMENT}" "cv_fold=${fold}" "$@"
elif (( step_idx == N_FOLDS )); then
    echo "=== save_train_predictions (wait for CV checkpoints) ==="
    wait_for_all_cv_ckpts "$EXPERIMENT"
    python save_train_predictions.py "experiment=${EXPERIMENT}" "$@"
elif (( step_idx == N_FOLDS + 1 )); then
    echo "=== train_all_data ==="
    python train_all_data.py "experiment=${EXPERIMENT}" "$@"
else
    echo "Unknown step_idx=$step_idx" >&2
    exit 1
fi

elapsed=$(( $(date +%s) - start_time ))
echo "Task ${SLURM_ARRAY_TASK_ID:-0} completed at $(date +%Y-%m-%d\ %H:%M:%S)"
echo "Total time: $((elapsed / 3600))h $(((elapsed % 3600) / 60))m"
