#!/usr/bin/env bash
set -euo pipefail

# Default Synthefy Tabular full training recipe.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh
#   TOTAL_STEPS=2 NPROC_PER_NODE=1 WANDB_MODE=disabled bash scripts/train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_TAG="${RUN_TAG:-synthefy-tabular-train-$(date +%Y%m%d-%H%M%S)}"
CKPT_DIR="${REPO_ROOT}/checkpoints/${RUN_TAG}/stage1"
mkdir -p "${CKPT_DIR}"

NPROC="${NPROC_PER_NODE:-4}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _VISIBLE_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_VISIBLE_DEVICES[@]}"
fi

MASTER_PORT="${MASTER_PORT:-29830}"
WANDB_PROJECT="${WANDB_PROJECT:-synthefy}"
WANDB_GROUP="${WANDB_GROUP:-tabular-train}"
WANDB_NAME="${WANDB_NAME:-${RUN_TAG}-stage1}"

TOTAL_STEPS="${TOTAL_STEPS:-250000}"
LR="${LR:-2e-4}"
WARMUP="${WARMUP:-4000}"
DECAY_START="${DECAY_START:-60000}"
BATCH_SIZE="${BATCH_SIZE:-24}"
BASE_CKPT="${BASE_CKPT:-cache/LimiX-2M.ckpt}"

QUANTILES="0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.70,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.80,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99"

SEED_CKPT="${SEED_CKPT:-}"
RESUME_ARGS=()
if [[ -n "${SEED_CKPT}" ]]; then
  RESUME_ARGS=(--resume "${SEED_CKPT}" --resume-model-only)
  echo "Seeding from: ${SEED_CKPT}"
fi

echo
echo "============================================================"
echo "  Synthefy Tabular Training"
echo "  checkpoint_dir = ${CKPT_DIR}"
echo "  total_steps    = ${TOTAL_STEPS}"
echo "  lr / warmup    = ${LR} / ${WARMUP}"
echo "  decay_start    = ${DECAY_START}"
echo "  batch_size     = ${BATCH_SIZE}"
echo "  nproc          = ${NPROC}"
echo "  quantiles      = 100"
echo "  prefetch       = 8 workers, 24 queued"
echo "  optimizer      = Muon+AdamW"
echo "  column_y_aware = ON"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "  GPUs           = ${CUDA_VISIBLE_DEVICES}"
fi
echo "============================================================"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  -m synthefy_tabular.training.cli \
  --checkpoint "${BASE_CKPT}" \
  "${RESUME_ARGS[@]}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --total-steps "${TOTAL_STEPS}" --run-steps "${TOTAL_STEPS}" \
  --lr "${LR}" --warmup-steps "${WARMUP}" --decay-start-step "${DECAY_START}" \
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
  --icl-filter-model "${BASE_CKPT}" \
  --icl-filter-reg-min-r2 0.05 \
  --quality-filter-max-retries 5 \
  --model-v2-lite \
  --column-specific-y-aware \
  --embed-dim 128 --hid-dim 384 --nhead 2 --nlayers 16 \
  --batch-size "${BATCH_SIZE}" \
  --max-budget 250000 \
  --min-features 2 --max-features 250 \
  --dim-bias-samples 1.5 --dim-bias-features 1.3 \
  --prefetch-workers 8 --prefetch-count 24 \
  --no-flash-attn \
  --gradient-checkpointing \
  --save-interval 15000 --eval-interval 15000 --log-interval 1000 \
  --early-stop-patience-evals 3 --early-stop-metric mean_r2 \
  --early-stop-min-delta 0.001 --early-stop-min-evals 2 \
  --wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}" \
  --wandb-name "${WANDB_NAME}" --wandb-job-type stage1
