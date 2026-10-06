#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${IF2F_DATA_ROOT:?Set IF2F_DATA_ROOT to the CelebA-HQ dataset root.}"
IF2F_EXP_DIR="${IF2F_EXP_DIR:-${PROJECT_ROOT}/experiments/if2f}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" train.py \
    --data_root "${IF2F_DATA_ROOT}" \
    --exp_dir "${IF2F_EXP_DIR}" \
    "$@"
