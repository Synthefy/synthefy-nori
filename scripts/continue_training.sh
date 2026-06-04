#!/usr/bin/env bash
set -euo pipefail

# Curriculum continuation: tiers 2 to 5, seeded from the tier-1 checkpoint
# produced by train.sh.
#
# Tier structure (each tier seeds from the previous tier's final checkpoint
# with --resume-model-only: fresh LR schedule, fresh optimizer):
#
#   tier1: broad mix, small tables          (train.sh)
#   tier2: larger tables   n<=4096,  f<=384
#   tier3: largest tables  n<=8144,  f<=768
#   tier4: large-N         n<=56000, f<=96    (large-context specialist)
#   tier5: both-large      n<=33000, f<=1280  (large N and large F, coupled)
#
# All tiers TRAIN TO COMPLETION on synthetic data: no real-data validation,
# no early stopping, no eval data required. Each tier runs its full step budget
# and the next tier seeds from the previous tier's final checkpoint.
#
# ── Tier 4/5 are large-N / both-large and can OOM or NCCL-hang ──────────────
# N up to 56K with dense O(N^2) sample attention forces batch=1 (effective
# batch is recovered via gradient accumulation). Memory at these shapes is
# uncertain on a given GPU/topology. SMOKE-PROBE tiers 4 and 5 before a real
# run, e.g.:
#
#   NCCL_TIMEOUT=300 TIER5_STEPS=12 TIER5_WARMUP=2 TIER5_DECAY_START=8 \
#     START_TIER=5 SAVE_INTERVAL=999 \
#     CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/continue_training.sh
#
# If a tier OOMs or hangs: lower its *_MAX_SAMPLES and/or *_MAX_BUDGET.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/continue_training.sh      # tiers 2 to 5
#   START_TIER=4 bash scripts/continue_training.sh                      # tiers 4 to 5
#   END_TIER=3   bash scripts/continue_training.sh                      # tiers 2 to 3 only
#   RUN_ROOT=checkpoints/train-YYYYMMDD-HHMMSS bash scripts/continue_training.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Find the tier1 run dir (most recent train.sh run by default).
if [[ -z "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="$(ls -dt "${REPO_ROOT}"/checkpoints/train-* 2>/dev/null | head -1 || true)"
  if [[ -z "${RUN_ROOT}" ]]; then
    echo "ERROR: no checkpoints/train-* directory found" >&2
    exit 1
  fi
fi
if [[ "${RUN_ROOT}" != /* ]]; then
  RUN_ROOT="${REPO_ROOT}/${RUN_ROOT}"
fi

# resolve_seed <tier_dir> -> echo the tier's final checkpoint path.
# Training writes periodic and final checkpoint_step_*.pt files; the most
# recent one is the tier's final checkpoint.
resolve_seed() {
  local dir="$1"
  ls -t "${dir}"/checkpoint_step_*.pt 2>/dev/null | head -1
}

TIER1_DIR="${RUN_ROOT}/tier1"
TIER1_SEED="$(resolve_seed "${TIER1_DIR}")"
if [[ -z "${TIER1_SEED}" || ! -f "${TIER1_SEED}" ]]; then
  echo "ERROR: no tier1 checkpoint found in ${TIER1_DIR}" >&2
  echo "  (expected checkpoint_step_*.pt)" >&2
  exit 1
fi

NPROC="${NPROC_PER_NODE:-4}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _VD <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_VD[@]}"
fi

MASTER_PORT="${MASTER_PORT:-29890}"
WANDB_PROJECT="${WANDB_PROJECT:-synthefy}"
WANDB_GROUP="${WANDB_GROUP:-continuation}"

# 999 quantiles
QUANTILES=$("${REPO_ROOT}/.venv/bin/python" -c "
print(','.join(f'{(i+1)/1000:.4f}' for i in range(999)))
")

# Large-N tiers fragment CUDA memory; expandable segments help avoid OOM.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

# Optional global overrides (apply to every tier, e.g. SAVE_INTERVAL=999 for a
# smoke probe). Captured once here so per-tier defaults below don't leak across
# tiers. Empty unless the user sets them in the environment.
SAVE_INTERVAL_OVERRIDE="${SAVE_INTERVAL:-}"
LOG_INTERVAL_OVERRIDE="${LOG_INTERVAL:-}"

# Shared args: data recipe + arch + optimizer, identical across tiers.
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
)

# run_tier <tier_num> <seed_ckpt> <out_dir>
# All per-tier knobs are read from the TIER-prefixed env vars set below.
run_tier() {
  local tier="$1" seed="$2" outdir="$3"
  echo
  echo "============================================================"
  echo "  Tier ${tier}"
  echo "  seed_ckpt        = ${seed}"
  echo "  out_dir          = ${outdir}"
  echo "  total_steps      = ${STEPS}"
  echo "  lr / warmup      = ${LR} / ${WARMUP}"
  echo "  decay_start      = ${DECAY_START}"
  echo "  batch / grad-acc = ${BATCH_SIZE} / ${GRAD_ACCUM}  (effective $((BATCH_SIZE*GRAD_ACCUM*NPROC)))"
  echo "  table shapes     = n[${MIN_SAMPLES}-${MAX_SAMPLES}] f[${MIN_FEATURES}-${MAX_FEATURES}]"
  echo "  dim_bias         = s=${DIM_BIAS_SAMPLES} f=${DIM_BIAS_FEATURES}   budget=${MAX_BUDGET}"
  echo "  eval             = none (trains to completion)"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "  GPUs             = ${CUDA_VISIBLE_DEVICES}"
  fi
  echo "============================================================"

  torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    -m synthefy_tabular.training.cli \
    "${SHARED_ARGS[@]}" \
    --resume "${seed}" --resume-model-only \
    --checkpoint-dir "${outdir}" \
    --total-steps "${STEPS}" --run-steps "${STEPS}" \
    --lr "${LR}" \
    --warmup-steps "${WARMUP}" \
    --decay-start-step "${DECAY_START}" \
    --gradient-accumulation "${GRAD_ACCUM}" \
    --batch-size "${BATCH_SIZE}" \
    --max-budget "${MAX_BUDGET}" \
    --min-samples "${MIN_SAMPLES}" \
    --max-samples "${MAX_SAMPLES}" \
    --min-features "${MIN_FEATURES}" \
    --max-features "${MAX_FEATURES}" \
    --dim-bias-samples "${DIM_BIAS_SAMPLES}" \
    --dim-bias-features "${DIM_BIAS_FEATURES}" \
    --save-interval "${SAVE_INTERVAL}" \
    --log-interval "${LOG_INTERVAL}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-name "tier${tier}-$(date +%Y%m%d-%H%M%S)" \
    --wandb-job-type "tier${tier}"
}

START_TIER="${START_TIER:-2}"
END_TIER="${END_TIER:-5}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"

# ── TIER 2 — larger tables ──────────────────────────────────────────────────
if (( START_TIER <= 2 && END_TIER >= 2 )); then
  STEPS="${TIER2_STEPS:-30000}"
  LR="${TIER2_LR:-1e-4}"
  WARMUP="${TIER2_WARMUP:-2000}"
  DECAY_START="${TIER2_DECAY_START:-20000}"
  BATCH_SIZE="${TIER2_BATCH_SIZE:-8}"
  GRAD_ACCUM="${TIER2_GRAD_ACCUM:-3}"
  MAX_BUDGET="${TIER2_MAX_BUDGET:-500000}"
  MIN_SAMPLES="${TIER2_MIN_SAMPLES:-256}"
  MAX_SAMPLES="${TIER2_MAX_SAMPLES:-4096}"
  MIN_FEATURES="${TIER2_MIN_FEATURES:-16}"
  MAX_FEATURES="${TIER2_MAX_FEATURES:-384}"
  DIM_BIAS_SAMPLES="${TIER2_DIM_BIAS_SAMPLES:-2.2}"
  DIM_BIAS_FEATURES="${TIER2_DIM_BIAS_FEATURES:-1.55}"
  SAVE_INTERVAL="${SAVE_INTERVAL_OVERRIDE:-5000}"
  LOG_INTERVAL="${LOG_INTERVAL_OVERRIDE:-1000}"

  TIER2_DIR="${RUN_ROOT}/tier2"
  mkdir -p "${TIER2_DIR}"
  run_tier 2 "${TIER1_SEED}" "${TIER2_DIR}"
fi

# ── TIER 3 — largest tables ─────────────────────────────────────────────────
if (( START_TIER <= 3 && END_TIER >= 3 )); then
  STEPS="${TIER3_STEPS:-30000}"
  LR="${TIER3_LR:-5e-5}"
  WARMUP="${TIER3_WARMUP:-1000}"
  DECAY_START="${TIER3_DECAY_START:-10000}"
  BATCH_SIZE="${TIER3_BATCH_SIZE:-4}"
  GRAD_ACCUM="${TIER3_GRAD_ACCUM:-6}"
  MAX_BUDGET="${TIER3_MAX_BUDGET:-800000}"
  MIN_SAMPLES="${TIER3_MIN_SAMPLES:-512}"
  MAX_SAMPLES="${TIER3_MAX_SAMPLES:-8144}"
  MIN_FEATURES="${TIER3_MIN_FEATURES:-32}"
  MAX_FEATURES="${TIER3_MAX_FEATURES:-768}"
  DIM_BIAS_SAMPLES="${TIER3_DIM_BIAS_SAMPLES:-2.5}"
  DIM_BIAS_FEATURES="${TIER3_DIM_BIAS_FEATURES:-1.75}"
  SAVE_INTERVAL="${SAVE_INTERVAL_OVERRIDE:-5000}"
  LOG_INTERVAL="${LOG_INTERVAL_OVERRIDE:-1000}"

  TIER3_DIR="${RUN_ROOT}/tier3"
  mkdir -p "${TIER3_DIR}"
  TIER2_SEED="$(resolve_seed "${RUN_ROOT}/tier2")"
  if [[ -z "${TIER2_SEED}" || ! -f "${TIER2_SEED}" ]]; then
    echo "ERROR: tier 3 needs a tier2 checkpoint but none found in ${RUN_ROOT}/tier2" >&2
    exit 1
  fi
  run_tier 3 "${TIER2_SEED}" "${TIER3_DIR}"
fi

# ── TIER 4 — large-N specialist (large N, low F; coupled by budget) ─────────
# At N=49152 dense O(N^2) sample attention forces batch=1; effective batch is
# recovered via gradient accumulation (1 x 16 x NPROC). MAX_SAMPLES=56000 lets
# the bucket round-down reach the top SAMPLE_BUCKETS entry (49152); MAX_BUDGET
# couples N x F so high-N episodes stay low-F (matches real large datasets).
if (( START_TIER <= 4 && END_TIER >= 4 )); then
  STEPS="${TIER4_STEPS:-10000}"
  LR="${TIER4_LR:-5e-5}"
  WARMUP="${TIER4_WARMUP:-300}"
  DECAY_START="${TIER4_DECAY_START:-4000}"
  BATCH_SIZE="${TIER4_BATCH_SIZE:-1}"
  GRAD_ACCUM="${TIER4_GRAD_ACCUM:-16}"
  MAX_BUDGET="${TIER4_MAX_BUDGET:-1600000}"
  MIN_SAMPLES="${TIER4_MIN_SAMPLES:-1024}"
  MAX_SAMPLES="${TIER4_MAX_SAMPLES:-56000}"
  MIN_FEATURES="${TIER4_MIN_FEATURES:-4}"
  MAX_FEATURES="${TIER4_MAX_FEATURES:-96}"
  DIM_BIAS_SAMPLES="${TIER4_DIM_BIAS_SAMPLES:-2.8}"
  DIM_BIAS_FEATURES="${TIER4_DIM_BIAS_FEATURES:-1.3}"
  SAVE_INTERVAL="${SAVE_INTERVAL_OVERRIDE:-1000}"
  LOG_INTERVAL="${LOG_INTERVAL_OVERRIDE:-200}"

  TIER4_DIR="${RUN_ROOT}/tier4"
  mkdir -p "${TIER4_DIR}"
  TIER3_SEED="$(resolve_seed "${RUN_ROOT}/tier3")"
  if [[ -z "${TIER3_SEED}" || ! -f "${TIER3_SEED}" ]]; then
    echo "ERROR: tier 4 needs a tier3 checkpoint but none found in ${RUN_ROOT}/tier3" >&2
    exit 1
  fi
  run_tier 4 "${TIER3_SEED}" "${TIER4_DIR}"
fi

# ── TIER 5 — both-large (large N AND large F in the same episode) ───────────
# Tiers 3/4 trained the edges of the (N,F) grid; tier 5 trains the corner.
# The budget cap PROJECTS over-budget draws onto the frontier N x F = MAX_BUDGET
# (keeps N, shrinks F), so each episode is large-N OR large-F with a both-largish
# middle — 33000 x 1280 at once is physically impossible (it OOMs). This tier
# re-teaches the high-F regime that tier 4's F<=96 cap drops.
if (( START_TIER <= 5 && END_TIER >= 5 )); then
  STEPS="${TIER5_STEPS:-8000}"
  LR="${TIER5_LR:-5e-5}"
  WARMUP="${TIER5_WARMUP:-250}"
  DECAY_START="${TIER5_DECAY_START:-3000}"
  BATCH_SIZE="${TIER5_BATCH_SIZE:-1}"
  GRAD_ACCUM="${TIER5_GRAD_ACCUM:-16}"
  MAX_BUDGET="${TIER5_MAX_BUDGET:-7000000}"
  MIN_SAMPLES="${TIER5_MIN_SAMPLES:-1024}"
  MAX_SAMPLES="${TIER5_MAX_SAMPLES:-33000}"
  MIN_FEATURES="${TIER5_MIN_FEATURES:-16}"
  MAX_FEATURES="${TIER5_MAX_FEATURES:-1280}"
  DIM_BIAS_SAMPLES="${TIER5_DIM_BIAS_SAMPLES:-2.0}"
  DIM_BIAS_FEATURES="${TIER5_DIM_BIAS_FEATURES:-2.0}"
  SAVE_INTERVAL="${SAVE_INTERVAL_OVERRIDE:-1000}"
  LOG_INTERVAL="${LOG_INTERVAL_OVERRIDE:-200}"

  TIER5_DIR="${RUN_ROOT}/tier5"
  mkdir -p "${TIER5_DIR}"
  TIER4_SEED="$(resolve_seed "${RUN_ROOT}/tier4")"
  if [[ -z "${TIER4_SEED}" || ! -f "${TIER4_SEED}" ]]; then
    echo "ERROR: tier 5 needs a tier4 checkpoint but none found in ${RUN_ROOT}/tier4" >&2
    exit 1
  fi
  run_tier 5 "${TIER4_SEED}" "${TIER5_DIR}"
fi

echo
echo "Continuation complete (tiers ${START_TIER} to ${END_TIER})."
echo "  run root: ${RUN_ROOT}"
