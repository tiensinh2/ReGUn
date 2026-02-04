#!/bin/bash
#SBATCH --job-name=mul
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=10
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --signal=TERM@60
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

handle_error() {
  local exit_code=${1:-$?}
  echo "[ERROR] Job step failed with exit code ${exit_code}; cancelling ${SLURM_JOB_ID:-unknown}"
  [[ -n "${SLURM_JOB_ID:-}" ]] && scancel "${SLURM_JOB_ID}" 2>/dev/null || true
  pkill -TERM -P $$ 2>/dev/null || true
  sleep 1
  pkill -KILL -P $$ 2>/dev/null || true
  exit "${exit_code}"
}

set -Eeuo pipefail
trap 'handle_error $?' ERR SIGINT SIGTERM

# -----------------------------------------------------------------------------
# Override any of these via environment variables when submitting, e.g.:
#   sbatch --export=ALL,PROJ_DIR=$PWD,IMG=/path/to/env.sif,DATASET_ROOT=/scratch/datasets run_experiment.sbatch run1 data=cifar10
# -----------------------------------------------------------------------------

# Project directory on the compute node (default: directory where this sbatch file resides)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="${PROJ_DIR:-$SCRIPT_DIR}"

# Container image (Singularity/Apptainer)
IMG="${IMG:-/path/to/container/mul_env.sif}"

# Data/cache/outputs roots (set these for your environment)
CACHE_ROOT="${CACHE_ROOT:-/path/to/cache}"
DATASET_ROOT="${DATASET_ROOT:-/path/to/datasets}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$PROJ_DIR/outputs}"

# Ensure log/output dirs exist (repo-safe)
mkdir -p "slurm" 2>/dev/null || true
mkdir -p "$OUTPUTS_ROOT" 2>/dev/null || true

if [[ $# -lt 1 ]]; then
  echo "[ERROR] Missing run script selector."
  echo "Usage: sbatch run_experiment.sbatch <run1|run2|run3> [args...]"
  exit 1
fi

RUN_SELECTOR=$1
shift

case "$RUN_SELECTOR" in
  run1|run1_reference)  TARGET_SCRIPT="run1_reference.py" ;;
  run2|run2_base)       TARGET_SCRIPT="run2_base.py" ;;
  run3|run3_unlearning) TARGET_SCRIPT="run3_unlearning.py" ;;
  *)
    echo "[ERROR] Unknown run script selector '$RUN_SELECTOR'. Expected one of: run1, run2, run3."
    exit 1
    ;;
esac

JOB_NAME="mul-${RUN_SELECTOR}"
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME" >/dev/null 2>&1 || true
fi

echo "[INFO] Starting job ${JOB_NAME:-$SLURM_JOB_NAME} (${SLURM_JOB_ID:-no_job_id}) on $(hostname) at $(date)"
echo "[INFO] Project dir:  $PROJ_DIR"
echo "[INFO] Container:    $IMG"
echo "[INFO] Data root:    $DATASET_ROOT"
echo "[INFO] Cache root:   $CACHE_ROOT"
echo "[INFO] Outputs root: $OUTPUTS_ROOT"
echo "[INFO] Arguments passed: $*"
echo "[INFO] Selected run script: $TARGET_SCRIPT"

START_TIME=$(date +%s)

# Use Apptainer if available, else Singularity
CONTAINER_BIN="${CONTAINER_BIN:-}"
if [[ -z "$CONTAINER_BIN" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_BIN="apptainer"
  else
    CONTAINER_BIN="singularity"
  fi
fi

srun --ntasks=1 --kill-on-bad-exit=1 \
  "$CONTAINER_BIN" exec --nv --cleanenv \
  --bind "$PROJ_DIR:/app" \
  --bind "$DATASET_ROOT:$DATASET_ROOT" \
  --bind "$CACHE_ROOT:$CACHE_ROOT" \
  --bind "$OUTPUTS_ROOT:$OUTPUTS_ROOT" \
  --env "DATA_DIR=$DATASET_ROOT" \
  --env "CACHE_DIR=$CACHE_ROOT" \
  --env "OUTPUTS_DIR=$OUTPUTS_ROOT" \
  "$IMG" \
  bash -lc 'set -Eeuo pipefail; cd /app; python -u "$1" "${@:2}"' -- "$TARGET_SCRIPT" "$@" \
  || handle_error $?

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
echo "[INFO] Job finished at $(date). Total runtime: ${RUNTIME}s"
