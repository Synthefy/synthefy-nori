#!/usr/bin/env bash
set -euo pipefail

# V13 Tier 2/3 training — extends tier1 with larger-table training.
#
# Tier structure:
#   tier1 — broad mix, small tables (max_samples=2000, max_features=250) [done by train_v13.sh]
#   tier2 — larger tables (max_samples=4096, max_features=384)
#   tier3 — largest tables (max_samples=8144, max_features=768)
#
# Each tier seeds from the previous best with --resume-model-only (fresh LR
# schedule, fresh optimizer).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/continue_training_v13.sh
#
# To skip tier2 and start from tier3:
#   START_TIER=3 bash scripts/continue_training_v13.sh
# To run only tier2:
#   END_TIER=2 bash scripts/continue_training_v13.sh
# To use a specific run root:
#   RUN_ROOT=checkpoints/v13-scratch-YYYYMMDD-HHMMSS bash scripts/continue_training_v13.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Find V13 tier1 run dir
if [[ -z "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="$(ls -dt "${REPO_ROOT}"/checkpoints/v13-scratch-* 2>/dev/null | head -1 || true)"
  if [[ -z "${RUN_ROOT}" ]]; then
    echo "ERROR: no checkpoints/v13-scratch-* directory found" >&2
    exit 1
  fi
fi
if [[ "${RUN_ROOT}" != /* ]]; then
  RUN_ROOT="${REPO_ROOT}/${RUN_ROOT}"
fi

TIER1_BEST="${RUN_ROOT}/tier1/best_reg_r2.pt"
if [[ ! -f "${TIER1_BEST}" ]]; then
  echo "ERROR: tier1 best not found: ${TIER1_BEST}" >&2
  exit 1
fi

TIER2_DIR="${RUN_ROOT}/tier2"
TIER3_DIR="${RUN_ROOT}/tier3"
mkdir -p "${TIER2_DIR}" "${TIER3_DIR}"

NPROC="${NPROC_PER_NODE:-4}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _VD <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_VD[@]}"
fi

MASTER_PORT="${MASTER_PORT:-29890}"
WANDB_PROJECT="${WANDB_PROJECT:-synthefy}"
WANDB_GROUP="${WANDB_GROUP:-v13-tier23}"

# 999 quantiles (V13)
QUANTILES=$("${REPO_ROOT}/.venv/bin/python" -c "
print(','.join(f'{(i+1)/1000:.4f}' for i in range(999)))
")

# ── Tier 2 settings ────────────────────────────────────────────────
TIER2_STEPS="${TIER2_STEPS:-30000}"
TIER2_LR="${TIER2_LR:-1e-4}"
TIER2_WARMUP="${TIER2_WARMUP:-2000}"
TIER2_DECAY_START="${TIER2_DECAY_START:-20000}"
TIER2_BATCH_SIZE="${TIER2_BATCH_SIZE:-8}"
TIER2_GRAD_ACCUM="${TIER2_GRAD_ACCUM:-3}"
TIER2_MAX_BUDGET="${TIER2_MAX_BUDGET:-500000}"
TIER2_MIN_SAMPLES="${TIER2_MIN_SAMPLES:-256}"
TIER2_MAX_SAMPLES="${TIER2_MAX_SAMPLES:-4096}"
TIER2_MIN_FEATURES="${TIER2_MIN_FEATURES:-16}"
TIER2_MAX_FEATURES="${TIER2_MAX_FEATURES:-384}"
TIER2_DIM_BIAS_SAMPLES="${TIER2_DIM_BIAS_SAMPLES:-2.2}"
TIER2_DIM_BIAS_FEATURES="${TIER2_DIM_BIAS_FEATURES:-1.55}"

# ── Tier 3 settings ────────────────────────────────────────────────
TIER3_STEPS="${TIER3_STEPS:-30000}"
TIER3_LR="${TIER3_LR:-5e-5}"
TIER3_WARMUP="${TIER3_WARMUP:-1000}"
TIER3_DECAY_START="${TIER3_DECAY_START:-10000}"
TIER3_BATCH_SIZE="${TIER3_BATCH_SIZE:-4}"
TIER3_GRAD_ACCUM="${TIER3_GRAD_ACCUM:-6}"
TIER3_MAX_BUDGET="${TIER3_MAX_BUDGET:-800000}"
TIER3_MIN_SAMPLES="${TIER3_MIN_SAMPLES:-512}"
TIER3_MAX_SAMPLES="${TIER3_MAX_SAMPLES:-8144}"
TIER3_MIN_FEATURES="${TIER3_MIN_FEATURES:-32}"
TIER3_MAX_FEATURES="${TIER3_MAX_FEATURES:-768}"
TIER3_DIM_BIAS_SAMPLES="${TIER3_DIM_BIAS_SAMPLES:-2.5}"
TIER3_DIM_BIAS_FEATURES="${TIER3_DIM_BIAS_FEATURES:-1.75}"

# ── Shared args (same data recipe + arch as V13 tier1) ─────────────
SHARED_ARGS=(
  --optimizer muon
  --ema-decay 0.999
  --muon-include-embeddings
  --muon-include-nd
  --task-type reg
  --feature-loss-weight 0.0
  --regression-ratio 0.60
  --regression-loss pinball
  --regression-quantiles "${QUANTILES}"
  --pinball-monotonicity-weight 0.05
  --reg-prior-prob 0.10
  --reg-deterministic-prob 0.05
  --reg-denoise
  --reg-dense
  --synth-v4
  --synth-v5
  --synth-v5-mixture
  --no-v4-filter
  --tabicl-prior
  --tabicl-prior-prob 0.05
  --probabilistic-labels
  --nominal-categoricals
  --enhanced-missingness
  --clean-lowdim-prob 0.03
  --tree-prior-prob 0.08
  --lookup-prior-prob 0.02
  --gp-prior-prob 0.12
  --quadratic-surface-prob 0.06
  --sparse-nonlinear-prob 0.08
  --context-missingness-prob 0.5
  --realistic-augmentation-prob 0.5
  --icl-filter-model "${ICL_FILTER_MODEL:-limix}"
  --icl-filter-reg-min-r2 0.05
  --quality-filter-max-retries 5
  --model-v2-lite
  --column-specific-y-aware
  --embed-dim 128
  --hid-dim 384
  --nhead 2
  --nlayers 16
  --prefetch-workers 4
  --prefetch-count 6
  --no-flash-attn
  --gradient-checkpointing
  --save-interval 5000
  --eval-interval 5000
  --log-interval 1000
  --early-stop-patience-evals 4
  --early-stop-metric mean_r2
  --early-stop-min-delta 0.0005
  --early-stop-min-evals 2
)

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

START_TIER="${START_TIER:-2}"
END_TIER="${END_TIER:-3}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"

# ── TIER 2 ─────────────────────────────────────────────────────────
if (( START_TIER <= 2 && END_TIER >= 2 )); then
  echo
  echo "============================================================"
  echo "  V13 Tier 2 — larger tables"
  echo "  seed_ckpt        = ${TIER1_BEST}"
  echo "  tier2_dir        = ${TIER2_DIR}"
  echo "  total_steps      = ${TIER2_STEPS}"
  echo "  lr / warmup      = ${TIER2_LR} / ${TIER2_WARMUP}"
  echo "  decay_start      = ${TIER2_DECAY_START}"
  echo "  batch / grad-acc = ${TIER2_BATCH_SIZE} / ${TIER2_GRAD_ACCUM}"
  echo "  table shapes     = n[${TIER2_MIN_SAMPLES}-${TIER2_MAX_SAMPLES}] f[${TIER2_MIN_FEATURES}-${TIER2_MAX_FEATURES}]"
  echo "  dim_bias         = s=${TIER2_DIM_BIAS_SAMPLES} f=${TIER2_DIM_BIAS_FEATURES}"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "  GPUs             = ${CUDA_VISIBLE_DEVICES}"
  fi
  echo "============================================================"

  torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    -m synthefy_tabular.training.cli \
    "${SHARED_ARGS[@]}" \
    --resume "${TIER1_BEST}" --resume-model-only \
    --checkpoint-dir "${TIER2_DIR}" \
    --total-steps "${TIER2_STEPS}" --run-steps "${TIER2_STEPS}" \
    --lr "${TIER2_LR}" \
    --warmup-steps "${TIER2_WARMUP}" \
    --decay-start-step "${TIER2_DECAY_START}" \
    --gradient-accumulation "${TIER2_GRAD_ACCUM}" \
    --batch-size "${TIER2_BATCH_SIZE}" \
    --max-budget "${TIER2_MAX_BUDGET}" \
    --min-samples "${TIER2_MIN_SAMPLES}" \
    --max-samples "${TIER2_MAX_SAMPLES}" \
    --min-features "${TIER2_MIN_FEATURES}" \
    --max-features "${TIER2_MAX_FEATURES}" \
    --dim-bias-samples "${TIER2_DIM_BIAS_SAMPLES}" \
    --dim-bias-features "${TIER2_DIM_BIAS_FEATURES}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-name "v13-tier2-$(date +%Y%m%d-%H%M%S)" \
    --wandb-job-type tier2
fi

# ── TIER 3 ─────────────────────────────────────────────────────────
if (( START_TIER <= 3 && END_TIER >= 3 )); then
  TIER2_BEST=""
  if [[ -f "${TIER2_DIR}/best_reg_r2.pt" ]]; then
    TIER2_BEST="${TIER2_DIR}/best_reg_r2.pt"
  else
    TIER2_BEST=$(ls -t "${TIER2_DIR}"/checkpoint_step_*.pt 2>/dev/null | head -1)
  fi
  if [[ -z "${TIER2_BEST}" ]]; then
    echo "ERROR: Tier 3 needs tier2 checkpoint but none found in ${TIER2_DIR}" >&2
    exit 1
  fi

  echo
  echo "============================================================"
  echo "  V13 Tier 3 — largest tables"
  echo "  seed_ckpt        = ${TIER2_BEST}"
  echo "  tier3_dir        = ${TIER3_DIR}"
  echo "  total_steps      = ${TIER3_STEPS}"
  echo "  lr / warmup      = ${TIER3_LR} / ${TIER3_WARMUP}"
  echo "  decay_start      = ${TIER3_DECAY_START}"
  echo "  batch / grad-acc = ${TIER3_BATCH_SIZE} / ${TIER3_GRAD_ACCUM}"
  echo "  table shapes     = n[${TIER3_MIN_SAMPLES}-${TIER3_MAX_SAMPLES}] f[${TIER3_MIN_FEATURES}-${TIER3_MAX_FEATURES}]"
  echo "  dim_bias         = s=${TIER3_DIM_BIAS_SAMPLES} f=${TIER3_DIM_BIAS_FEATURES}"
  echo "============================================================"

  torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    -m synthefy_tabular.training.cli \
    "${SHARED_ARGS[@]}" \
    --resume "${TIER2_BEST}" --resume-model-only \
    --checkpoint-dir "${TIER3_DIR}" \
    --total-steps "${TIER3_STEPS}" --run-steps "${TIER3_STEPS}" \
    --lr "${TIER3_LR}" \
    --warmup-steps "${TIER3_WARMUP}" \
    --decay-start-step "${TIER3_DECAY_START}" \
    --gradient-accumulation "${TIER3_GRAD_ACCUM}" \
    --batch-size "${TIER3_BATCH_SIZE}" \
    --max-budget "${TIER3_MAX_BUDGET}" \
    --min-samples "${TIER3_MIN_SAMPLES}" \
    --max-samples "${TIER3_MAX_SAMPLES}" \
    --min-features "${TIER3_MIN_FEATURES}" \
    --max-features "${TIER3_MAX_FEATURES}" \
    --dim-bias-samples "${TIER3_DIM_BIAS_SAMPLES}" \
    --dim-bias-features "${TIER3_DIM_BIAS_FEATURES}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-name "v13-tier3-$(date +%Y%m%d-%H%M%S)" \
    --wandb-job-type tier3
fi

echo
echo "V13 tier2/3 complete."
echo "  Tier 2 dir: ${TIER2_DIR}"
echo "  Tier 3 dir: ${TIER3_DIR}"
