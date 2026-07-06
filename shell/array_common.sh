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
