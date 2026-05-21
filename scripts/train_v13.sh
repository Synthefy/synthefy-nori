#!/usr/bin/env bash
set -euo pipefail

# V13 from-scratch — V11 architecture + audit-driven inference improvements.
#
# Changes vs V11:
#   1. 999 quantile pinball (was 100) — finer τ resolution
#   2. Pinball monotonicity penalty 0.05
#   3. Soft log outlier clip at inference (config-driven)
#   4. Latin square feature shuffling at inference (config-driven)
#   5. column_specific_y_aware — inherited from V11
#
# Architecture: 16 layers, E=128, H=384, nhead=2, model_v2_lite,
# column_specific_y_aware, ~5.5M params.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_v13.sh
#   TOTAL_STEPS=100 WANDB_MODE=disabled bash scripts/train_v13.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_TAG="${RUN_TAG:-v13-scratch-$(date +%Y%m%d-%H%M%S)}"
CKPT_DIR="${REPO_ROOT}/checkpoints/${RUN_TAG}/tier1"
mkdir -p "${CKPT_DIR}"

NPROC="${NPROC_PER_NODE:-4}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _VD <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_VD[@]}"
fi

MASTER_PORT="${MASTER_PORT:-29880}"
WANDB_PROJECT="${WANDB_PROJECT:-synthefy}"
WANDB_GROUP="${WANDB_GROUP:-v13-scratch}"
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
echo "  V13 from-scratch"
echo "  checkpoint_dir = ${CKPT_DIR}"
echo "  total_steps    = ${TOTAL_STEPS}"
echo "  lr / warmup    = ${LR} / ${WARMUP}"
echo "  decay_start    = ${DECAY_START}"
echo "  batch_size     = ${BATCH_SIZE}"
echo "  nproc          = ${NPROC}"
echo "  quantiles      = 999"
echo "  mono_penalty   = 0.05"
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
  -m synthefy_tabular.training.cli \
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
  --max-budget 250000 \
  --min-features 2 --max-features 250 \
  --dim-bias-samples 1.5 --dim-bias-features 1.3 \
  --prefetch-workers 4 --prefetch-count 6 \
  --no-flash-attn \
  --gradient-checkpointing \
  --save-interval 15000 --eval-interval 15000 --log-interval 1000 \
  --early-stop-patience-evals 6 --early-stop-metric mean_r2 \
  --early-stop-min-delta 0.001 --early-stop-min-evals 2 \
  --wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}" \
  --wandb-name "${WANDB_NAME}" --wandb-job-type tier1
