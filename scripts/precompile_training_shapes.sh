#!/usr/bin/env bash
set -euo pipefail

# Populate a persistent Inductor/AOTAutograd cache under the same DDP boundary
# used for training. Cached graphs are weight-independent but require the same
# architecture, tensor signatures, compiler flags, and software/hardware stack.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${COMPILE_TEMPLATE_CHECKPOINT:-}" ]]; then
  echo "COMPILE_TEMPLATE_CHECKPOINT must point to a Nori training checkpoint" >&2
  exit 2
fi
if [[ ! -f "${COMPILE_TEMPLATE_CHECKPOINT}" ]]; then
  echo "Checkpoint does not exist: ${COMPILE_TEMPLATE_CHECKPOINT}" >&2
  exit 2
fi

BATCH_SIZE="${BATCH_SIZE:-20}"
COMPILE_SHAPES="${COMPILE_SHAPES:-128x8@0.4,128x8@0.7,128x48@0.4,128x48@0.7,128x128@0.4,128x128@0.7,512x8@0.4,512x8@0.7,512x48@0.4,512x48@0.7,512x128@0.4,512x128@0.7,1536x8@0.4,1536x8@0.7,1536x48@0.4,1536x48@0.7,1536x128@0.4,1536x128@0.7}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${REPO_ROOT}/cache/torchinductor/nori-training-static-b${BATCH_SIZE}}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Match the public training launchers. These import-time settings affect the
# graph and therefore belong to the compiler-cache contract.
export SYNTHEFY_QASS_MODE="${SYNTHEFY_QASS_MODE:-log_only}"
export SYNTHEFY_QASS_SDPA_SCALE="${SYNTHEFY_QASS_SDPA_SCALE:-1}"
export SYNTHEFY_NORI_ALLOW_CUDNN_SDP="${SYNTHEFY_NORI_ALLOW_CUDNN_SDP:-1}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

OUTPUT="${COMPILE_OUTPUT:-${TORCHINDUCTOR_CACHE_DIR}/precompile_manifest.json}"
NPROC="${NPROC_PER_NODE:-4}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _VISIBLE_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_VISIBLE_DEVICES[@]}"
fi
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$("${REPO_ROOT}/.venv/bin/python" -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
fi

NATIVE_RMS_ARGS=()
if [[ "${NATIVE_RMS_NORM:-1}" != "0" ]]; then
  NATIVE_RMS_ARGS+=(--native-rms-norm)
fi
SIGNATURE_COUNT=$(( $(tr -cd ',' <<< "${COMPILE_SHAPES}" | wc -c) + 1 ))

echo "Precompiling ${SIGNATURE_COUNT} training signatures"
echo "  checkpoint = ${COMPILE_TEMPLATE_CHECKPOINT}"
echo "  cache      = ${TORCHINDUCTOR_CACHE_DIR}"
echo "  output     = ${OUTPUT}"
echo "  DDP ranks  = ${NPROC}"

exec "${REPO_ROOT}/.venv/bin/torchrun" \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/benchmarks/bench_training_compile.py" \
  --checkpoint "${COMPILE_TEMPLATE_CHECKPOINT}" \
  --device "${COMPILE_DEVICE:-cuda:0}" \
  --strategy forward-static \
  --compile-mode "${COMPILE_MODE:-default}" \
  --compile-cache-limit "${COMPILE_CACHE_LIMIT:-1024}" \
  --disable-ddp-optimizer \
  "${NATIVE_RMS_ARGS[@]}" \
  --shapes "${COMPILE_SHAPES}" \
  --batch-size "${BATCH_SIZE}" \
  --cycles 1 \
  --shape-order grouped \
  --checkpointing auto \
  --checkpoint-threshold "${GRAD_CKPT_THRESHOLD:-24576}" \
  --output "${OUTPUT}"
