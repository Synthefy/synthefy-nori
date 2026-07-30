#!/usr/bin/env bash
set -euo pipefail

# Training: tier 1 (from scratch).
#
# Architecture: 16 layers, E=128, H=384, nhead=2, model_v2_lite,
# column_specific_y_aware, ~6M params. Regression via a 999-quantile pinball
# loss with a monotonicity penalty.
#
# Trains to completion on synthetic data: no real-data validation, no early
# stopping, no eval data required. Periodic and final checkpoints are written;
# the curriculum continuation (continue_training.sh, tiers 2-5) seeds from the
# final checkpoint.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh
#   TOTAL_STEPS=100 WANDB_MODE=disabled bash scripts/train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_TAG="${RUN_TAG:-train-$(date +%Y%m%d-%H%M%S)}"
CKPT_DIR="${REPO_ROOT}/checkpoints/${RUN_TAG}/tier1"
mkdir -p "${CKPT_DIR}"

NPROC="${NPROC_PER_NODE:-4}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _VD <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_VD[@]}"
fi

MASTER_PORT="${MASTER_PORT:-29880}"
WANDB_PROJECT="${WANDB_PROJECT:-synthefy}"
WANDB_GROUP="${WANDB_GROUP:-train}"
WANDB_NAME="${WANDB_NAME:-${RUN_TAG}-tier1}"

TOTAL_STEPS="${TOTAL_STEPS:-250000}"
LR="${LR:-2e-4}"
WARMUP="${WARMUP:-4000}"
DECAY_START="${DECAY_START:-60000}"
BATCH_SIZE="${BATCH_SIZE:-24}"

# 999 quantiles: τ_i = (i+1)/1000 for i in 0..998
QUANTILES=$("${REPO_ROOT}/.venv/bin/python" -c "
print(','.join(f'{(i+1)/1000:.4f}' for i in range(999)))
")

SEED_CKPT="${SEED_CKPT:-}"
RESUME_ARGS=()
if [[ -n "${SEED_CKPT}" ]]; then
  RESUME_ARGS=(--resume "${SEED_CKPT}" --resume-model-only)
  echo "Seeding from: ${SEED_CKPT}"
fi

echo
echo "============================================================"
echo "  Tier 1 (from scratch)"
echo "  checkpoint_dir = ${CKPT_DIR}"
echo "  total_steps    = ${TOTAL_STEPS}"
echo "  lr / warmup    = ${LR} / ${WARMUP}"
echo "  decay_start    = ${DECAY_START}"
echo "  batch_size     = ${BATCH_SIZE}"
echo "  nproc          = ${NPROC}"
echo "  quantiles      = 999"
echo "  mono_penalty   = 0.05"
echo "  eval           = none (trains to completion)"
echo "  prefetch       = 4 workers, 6 queued"
echo "  optimizer      = Muon+AdamW"
echo "  column_y_aware = ON"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "  GPUs           = ${CUDA_VISIBLE_DEVICES}"
fi
echo "============================================================"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  -m synthefy_nori.training.cli \
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
  --pinball-monotonicity-weight 0.05 \
  --reg-prior-prob 0.10 --reg-deterministic-prob 0.05 \
  --reg-denoise --reg-dense \
  --synth-v4 --synth-v5 --synth-v5-mixture \
  --no-v4-filter \
  --scm-prior --scm-prior-prob 0.05 \
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
  --max-budget 250000 \
  --min-features 2 --max-features 250 \
  --dim-bias-samples 1.5 --dim-bias-features 1.3 \
  --prefetch-workers 4 --prefetch-count 6 \
  --gradient-checkpointing \
  --save-interval 15000 --log-interval 1000 \
  --wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}" \
  --wandb-name "${WANDB_NAME}" --wandb-job-type tier1
