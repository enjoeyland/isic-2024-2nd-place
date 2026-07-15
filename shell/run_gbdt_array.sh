#!/bin/bash
#SBATCH -J ISIC_GBDT
#SBATCH -p asus_a5000,gigabyte_a5000,suma_rtx4090,big_suma_rtx3090
#SBATCH -q big_qos
#SBATCH --gres=gpu:1
#SBATCH --array=0-13%4
#SBATCH --output=logs/slurm/gbdt_%A_%a.log
#SBATCH --time=3:00:00

# GBDT: 4 configs x (5 CV folds + tune + all_data) = 28 array tasks (~2h/fold).
# task_id = step_idx * 4 + config_idx  (config cycles fastest)
#   tasks 0-3   : fold=0 for all configs
#   tasks 4-7   : fold=1 for all configs
#   ...
#   tasks 16-19 : fold=4 for all configs
#   tasks 20-23 : tune (waits for all 5 fold checkpoints per config)
#   tasks 24-27 : all_data (waits for tune / ensemble weights)
#
# Usage (from repo root):
#   sbatch shell/run_gbdt_array.sh
#
# Optional env overrides:
#   CONDA_ENV=isic_2nd sbatch shell/run_gbdt_array.sh

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

GBDT_CONFIGS=(
    # "260709-0NNs-1types-feV7-s5-tuning_weights"
    # "260709-0NNs-18types-feV7-s5-tuning_weights"
    # "260709-1NNs-18types-feV7-s5-tuning_weights"
    # "260709-7NNs-18types-feV7-s5-tuning_weights"
    # "0906-9NNs-18types-feV7-s5-tuning_weights"
    # "260714-2NNs-17types-feV7-s5-tuning_weights"
    "260714-1NNs-1types-tip_conv-feV7-s5-tuning_weights"
    "260714-1NNs-1types-convnext-feV7-s5-tuning_weights"
)

N_FOLDS=5
N_POST_STEPS=2   # tune + all_data
STEPS_PER_CONFIG=$((N_FOLDS + N_POST_STEPS))
N_CONFIGS=${#GBDT_CONFIGS[@]}
N_TASKS=$((N_CONFIGS * STEPS_PER_CONFIG))
max_task=$((N_TASKS - 1))

TUNE_STEP=$N_FOLDS
ALL_DATA_STEP=$((N_FOLDS + 1))

start_time=$(date +%s)
echo "Job started at $(date +%Y-%m-%d\ %H:%M:%S)"
echo "ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-<none>} N_TASKS=$N_TASKS (0-${max_task}), STEPS_PER_CONFIG=$STEPS_PER_CONFIG"

task_id=${SLURM_ARRAY_TASK_ID:-0}
if (( task_id > max_task )); then
    echo "Skip array task $task_id (max index $max_task)."
    exit 0
fi

decode_task "$task_id" "$N_CONFIGS" "$STEPS_PER_CONFIG" || exit 1
GBDT_PARAMS="${GBDT_CONFIGS[$model_idx]}"
echo "task_id=$task_id config_idx=$model_idx step_idx=$step_idx gbdt_params=$GBDT_PARAMS"

cd src || exit 1

if (( step_idx < N_FOLDS )); then
    fold=$step_idx
    echo "=== gbdt fold=$fold ==="
    python gbdt.py "gbdt_params=${GBDT_PARAMS}" "cv_fold=${fold}" "gbdt_stage=fold" "$@"
elif (( step_idx == TUNE_STEP )); then
    echo "=== gbdt tune (wait for all fold checkpoints) ==="
    wait_for_all_gbdt_fold_models "$GBDT_PARAMS"
    python gbdt.py "gbdt_params=${GBDT_PARAMS}" "gbdt_stage=tune" "$@"
elif (( step_idx == ALL_DATA_STEP )); then
    echo "=== gbdt all_data (wait for tune / ensemble weights) ==="
    wait_for_gbdt_tune "$GBDT_PARAMS"
    python gbdt.py "gbdt_params=${GBDT_PARAMS}" "gbdt_stage=all_data" "$@"
else
    echo "Unknown step_idx=$step_idx" >&2
    exit 1
fi

elapsed=$(( $(date +%s) - start_time ))
echo "Task ${SLURM_ARRAY_TASK_ID:-0} completed at $(date +%Y-%m-%d\ %H:%M:%S)"
echo "Total time: $((elapsed / 3600))h $(((elapsed % 3600) / 60))m"
