#!/bin/bash
# Shared helpers for ISIC SLURM array scripts.

resolve_repo() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        local sub="${SLURM_SUBMIT_DIR%/}"
        if [[ -f "$sub/src/train_cv.py" ]]; then
            REPO="$sub"
        elif [[ -f "$sub/train_cv.py" ]]; then
            REPO="$(dirname "$sub")"
        else
            echo "src/train_cv.py not found under $sub" >&2
            return 1
        fi
    else
        local shell_dir
        shell_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        REPO="$(dirname "$shell_dir")"
    fi
}

activate_conda_env() {
    : "${CONDA_ENV:=isic_2nd}"
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    else
        echo "WARNING: conda not found; using python on PATH." >&2
    fi
}

checkpoint_dir() {
    local experiment=$1
    echo "${REPO}/logs/train/runs/${experiment}/checkpoints"
}

fold_ckpt_pattern() {
    local fold=$1
    if (( fold == 0 )); then
        echo "last.ckpt"
    else
        echo "last-v${fold}.ckpt"
    fi
}

fold_ckpt_exists() {
    local experiment=$1
    local fold=$2
    local dir pattern
    dir="$(checkpoint_dir "$experiment")"
    pattern="$(fold_ckpt_pattern "$fold")"
    compgen -G "${dir}/${pattern}" > /dev/null
}

wait_for_fold_ckpt() {
    local experiment=$1
    local fold=$2
    local poll_sec=${3:-120}
    local dir pattern
    dir="$(checkpoint_dir "$experiment")"
    pattern="$(fold_ckpt_pattern "$fold")"
    echo "Waiting for checkpoint: ${dir}/${pattern}"
    while ! fold_ckpt_exists "$experiment" "$fold"; do
        sleep "$poll_sec"
    done
    echo "Found checkpoint for experiment=${experiment} fold=${fold}"
}

pretrain_epoch_ckpt_path() {
    local pretrain_exp=$1
    local fold=$2
    local epoch=$3
    echo "$(checkpoint_dir "$pretrain_exp")/fold${fold}_epoch_${epoch}.ckpt"
}

pretrain_epoch_ckpt_exists() {
    [[ -f "$(pretrain_epoch_ckpt_path "$1" "$2" "$3")" ]]
}

# Finetune loads fold{N}_epoch_{E}.ckpt from pretrain, not last-v{N}.ckpt.
wait_for_pretrain_epoch_ckpt() {
    local pretrain_exp=$1
    local fold=$2
    local epoch=${3:-199}
    local poll_sec=${4:-120}
    local path
    path="$(pretrain_epoch_ckpt_path "$pretrain_exp" "$fold" "$epoch")"
    echo "Waiting for pretrain epoch checkpoint: ${path}"
    while ! pretrain_epoch_ckpt_exists "$pretrain_exp" "$fold" "$epoch"; do
        sleep "$poll_sec"
    done
    echo "Found pretrain epoch checkpoint for experiment=${pretrain_exp} fold=${fold} epoch=${epoch}"
}

wait_for_all_cv_ckpts() {
    local experiment=$1
    local fold
    for fold in 0 1 2 3 4; do
        wait_for_fold_ckpt "$experiment" "$fold"
    done
}

decode_task() {
  # decode_task <task_id> <n_models> <n_steps>
  # step-major order (model cycles fastest):
  #   task 0-6   -> fold 0 for models 0-6
  #   task 7-13  -> fold 1 for models 0-6
  #   ...
  # sets: model_idx, step_idx
  local task_id=$1
  local n_models=$2
  local n_steps=$3
  step_idx=$((task_id / n_models))
  model_idx=$((task_id % n_models))
  if (( step_idx >= n_steps )); then
    echo "Invalid task_id=$task_id for n_models=$n_models n_steps=$n_steps" >&2
    return 1
  fi
}

gbdt_run_dir() {
    local gbdt_params=$1
    echo "${REPO}/logs/gbdt/runs/${gbdt_params}"
}

gbdt_fold_model_exists() {
    local gbdt_params=$1
    local fold=$2
    [[ -f "$(gbdt_run_dir "$gbdt_params")/model_${fold}.joblib" ]]
}

wait_for_gbdt_fold_model() {
    local gbdt_params=$1
    local fold=$2
    local poll_sec=${3:-120}
    local path
    path="$(gbdt_run_dir "$gbdt_params")/model_${fold}.joblib"
    echo "Waiting for GBDT checkpoint: ${path}"
    while ! gbdt_fold_model_exists "$gbdt_params" "$fold"; do
        sleep "$poll_sec"
    done
    echo "Found GBDT checkpoint for gbdt_params=${gbdt_params} fold=${fold}"
}

wait_for_all_gbdt_fold_models() {
    local gbdt_params=$1
    local fold
    for fold in 0 1 2 3 4; do
        wait_for_gbdt_fold_model "$gbdt_params" "$fold"
    done
}

wait_for_gbdt_tune() {
    local gbdt_params=$1
    local poll_sec=${2:-120}
    local path
    path="$(gbdt_run_dir "$gbdt_params")/model_0.joblib"
    wait_for_all_gbdt_fold_models "$gbdt_params"
    echo "Waiting for GBDT ensemble weights (tune): ${path}"
    # joblib models pickle classes under src.*; need REPO on PYTHONPATH
    while ! PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}" python -c "
import joblib, sys
m = joblib.load('${path}')
sys.exit(0 if m.ensemble_weights is not None else 1)
"; do
        sleep "$poll_sec"
    done
    echo "Found ensemble weights for gbdt_params=${gbdt_params}"
}
