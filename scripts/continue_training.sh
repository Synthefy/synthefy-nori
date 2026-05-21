#!/usr/bin/env bash
set -euo pipefail

# Large-table continuation recipe for a completed default training run.
#
# Usage:
#   RUN_ROOT=checkpoints/synthefy-tabular-train-YYYYMMDD-HHMMSS \
#     CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/continue_training.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="$(ls -dt "${REPO_ROOT}"/checkpoints/synthefy-tabular-train-* 2>/dev/null | head -1 || true)"
  if [[ -z "${RUN_ROOT}" ]]; then
    echo "ERROR: no checkpoints/synthefy-tabular-train-* directory found" >&2
    exit 1
  fi
fi

if [[ "${RUN_ROOT}" != /* ]]; then
  RUN_ROOT="${REPO_ROOT}/${RUN_ROOT}"
fi

RUN_TAG="$(basename "${RUN_ROOT}")"
STAGE1_DIR="${RUN_ROOT}/stage1"
STAGE2_DIR="${RUN_ROOT}/stage2"
mkdir -p "${STAGE2_DIR}"

SEED_CKPT="${SEED_CKPT:-${STAGE1_DIR}/best_reg_r2.pt}"
if [[ ! -f "${SEED_CKPT}" ]]; then
  echo "ERROR: seed checkpoint not found: ${SEED_CKPT}" >&2
  exit 1
fi

NPROC="${NPROC_PER_NODE:-4}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _VISIBLE_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_VISIBLE_DEVICES[@]}"
fi

MASTER_PORT="${MASTER_PORT:-29841}"
WANDB_PROJECT="${WANDB_PROJECT:-synthefy}"
WANDB_GROUP="${WANDB_GROUP:-tabular-continue}"
WANDB_NAME="${WANDB_NAME:-${RUN_TAG}-stage2}"

TOTAL_STEPS="${TOTAL_STEPS:-30000}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-2000}"
DECAY_START="${DECAY_START:-20000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-8}"
PREFETCH_COUNT="${PREFETCH_COUNT:-24}"
MIN_SAMPLES="${MIN_SAMPLES:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-12288}"
MIN_FEATURES="${MIN_FEATURES:-16}"
MAX_FEATURES="${MAX_FEATURES:-1024}"
MAX_BUDGET="${MAX_BUDGET:-1200000}"
DIM_BIAS_SAMPLES="${DIM_BIAS_SAMPLES:-1.5}"
DIM_BIAS_FEATURES="${DIM_BIAS_FEATURES:-1.5}"
COMPILE="${COMPILE:-1}"

COMPILE_ARGS=()
if [[ "${COMPILE,,}" == "1" || "${COMPILE,,}" == "true" || "${COMPILE,,}" == "yes" ]]; then
  COMPILE_ARGS=(--compile)
fi

QUANTILES="0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.70,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.80,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99"

echo
echo "============================================================"
echo "  Synthefy Tabular Continuation"
echo "  seed           = ${SEED_CKPT}"
echo "  checkpoint_dir = ${STAGE2_DIR}"
echo "  total_steps    = ${TOTAL_STEPS}"
echo "  lr / warmup    = ${LR} / ${WARMUP}"
echo "  decay_start    = ${DECAY_START}"
echo "  batch_size     = ${BATCH_SIZE}  grad_accum = ${GRAD_ACCUM}  eff=$((BATCH_SIZE*GRAD_ACCUM*NPROC))"
echo "  samples        = ${MIN_SAMPLES}-${MAX_SAMPLES}"
echo "  features       = ${MIN_FEATURES}-${MAX_FEATURES}"
echo "  max_budget     = ${MAX_BUDGET}"
echo "  prefetch       = ${PREFETCH_WORKERS} workers, ${PREFETCH_COUNT} queued"
echo "  compile        = ${COMPILE}"
echo "  nproc          = ${NPROC}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "  GPUs           = ${CUDA_VISIBLE_DEVICES}"
fi
echo "============================================================"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  -m synthefy_tabular.training.cli \
  --resume "${SEED_CKPT}" --resume-model-only \
  --checkpoint-dir "${STAGE2_DIR}" \
  --total-steps "${TOTAL_STEPS}" --run-steps "${TOTAL_STEPS}" \
  --lr "${LR}" --warmup-steps "${WARMUP}" --decay-start-step "${DECAY_START}" \
  --gradient-accumulation "${GRAD_ACCUM}" \
  --optimizer muon --ema-decay 0.999 \
  --muon-include-embeddings --muon-include-nd \
  --task-type reg \
  --feature-loss-weight 0.0 \
  --regression-ratio 0.60 \
  --regression-loss pinball --regression-quantiles "${QUANTILES}" \
  --reg-prior-prob 0.10 --reg-deterministic-prob 0.05 \
  --reg-denoise --reg-dense \
  --synth-v4 --synth-v5 --synth-v5-mixture \
  --no-v4-filter \
  --tabicl-prior --tabicl-prior-prob 0.05 \
  --probabilistic-labels --nominal-categoricals --enhanced-missingness \
  --clean-lowdim-prob 0.03 --tree-prior-prob 0.08 --lookup-prior-prob 0.02 \
  --gp-prior-prob 0.12 \
  --quadratic-surface-prob 0.06 \
  --sparse-nonlinear-prob 0.08 \
  --context-missingness-prob 0.5 \
  --realistic-augmentation-prob 0.5 \
  --icl-filter-model "${ICL_FILTER_MODEL:-limix}" \
  --icl-filter-reg-min-r2 0.05 \
  --quality-filter-max-retries 5 \
  --model-v2-lite \
  --column-specific-y-aware \
  --embed-dim 128 --hid-dim 384 --nhead 2 --nlayers 16 \
  --batch-size "${BATCH_SIZE}" \
  --max-budget "${MAX_BUDGET}" \
  --min-samples "${MIN_SAMPLES}" --max-samples "${MAX_SAMPLES}" \
  --min-features "${MIN_FEATURES}" --max-features "${MAX_FEATURES}" \
  --dim-bias-samples "${DIM_BIAS_SAMPLES}" --dim-bias-features "${DIM_BIAS_FEATURES}" \
  --prefetch-workers "${PREFETCH_WORKERS}" --prefetch-count "${PREFETCH_COUNT}" \
  --no-flash-attn \
  --gradient-checkpointing \
  "${COMPILE_ARGS[@]}" \
  --save-interval 10000 --eval-interval 10000 --log-interval 500 \
  --early-stop-patience-evals 3 --early-stop-metric mean_r2 \
  --early-stop-min-delta 0.001 --early-stop-min-evals 2 \
  --wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}" \
  --wandb-name "${WANDB_NAME}" --wandb-job-type stage2
