# Training acceleration and optional static-shape compilation

This guide covers Nori's execution-only training speedups and the experimental
static-shape path. The CLI controls remain opt-in. The bundled training
launchers enable the curriculum-neutral subset by default: native RMSNorm,
foreach EMA updates, and regional dynamic compilation. These options preserve
the sampler, shape curriculum, and training hyperparameters; mixed-precision
rounding can still differ at bf16 scale because the kernels reduce in a
different order.

Static compilation remains explicitly opt-in because it requires a bounded
shape palette, which changes the data curriculum and therefore needs a separate
quality validation.

## Motivation

Nori's axial transformer sees many table shapes. A Tier-1 curriculum with 62
valid `(rows, features)` buckets and six context-ratio buckets can expose 372
compiler signatures. An early generic dynamic-compile prototype was 6.4%
slower than eager. The current implementation instead compiles the shared
encoder-layer region and measured 15.6% faster end-to-end on the natural shape
curriculum in one controlled run. Padding every episode to one maximum tensor
would waste quadratic row- and feature-attention work.

Exact static compilation is effective when the curriculum is projected onto a
small, explicitly weighted shape palette. The compiler cache can then be built
before training and reused by fresh models with the same architecture and
execution contract.

## Opt-in controls

| Flag | Effect |
|---|---|
| `--shape-palette ROWSxFEATURES:WEIGHT,...` | Samples from an explicit weighted physical-shape set. |
| `--context-ratio-palette RATIO,...` | Samples only the listed context fractions. |
| `--compile-encoder-layers static` | Regionally compiles one shared encoder-layer forward with exact static signatures. |
| `--compile-encoder-layers dynamic` | Regionally compiles shape-generic kernels. Needs **no** palette, so the data curriculum is unchanged. |
| `--compile-mode MODE` | Selects the `torch.compile` mode. `default` is the tested choice. |
| `--compile-cache-limit N` | Raises Dynamo's live variant limits for the bounded signature set. |
| `--compile-disable-ddp-optimizer` | Disables DDPOptimizer graph splitting for the already-small regional compile unit. |
| `--native-rms-norm` | Uses PyTorch's native RMSNorm kernel. |
| `--skip-zero-feature-decoder` | Omits feature-decoder work when feature loss remains zero. |
| `--ema-foreach` | Groups EMA updates into foreach kernels by device and dtype. |

All flags are off by default **in the CLI**. The public training launchers
(`scripts/train.sh` and `scripts/continue_training.sh`) turn on the
curriculum-neutral subset — `--native-rms-norm`, `--ema-foreach`, and
`--compile-encoder-layers dynamic` — through a `SPEEDUP_ARGS` block. Disable
them per run with `NATIVE_RMS_NORM=0`, `EMA_FOREACH=0`, or
`COMPILE_ENCODER_LAYERS=none`. The palette flags are never enabled there,
because they change the data curriculum.

`--skip-zero-feature-decoder` requires the feature
loss to remain zero and unused-head freezing to remain enabled. It is
deliberately NOT enabled by the scripts: `feature_loss_weight` defaults to 0.5,
so a blanket enable would make the CLI reject the run.
Static encoder compilation requires either `--shape-palette` or `--fixed-size`
and also requires `--context-ratio-palette`; the CLI rejects an unbounded static
compile configuration before model execution.

## Validated 18-signature profile

The initial profile uses three row counts, three feature counts, and two context
fractions:

```text
rows:             128, 512, 1536
features:         8, 48, 128
context ratios:   0.4, 0.7
```

The weighted physical palette projects the measured Tier-1 sampler onto those
nine shapes while keeping its cell-count distribution close to the control:

```text
128x8:0.240,128x48:0.135,128x128:0.126,
512x8:0.115,512x48:0.064,512x128:0.061,
1536x8:0.124,1536x48:0.079,1536x128:0.056
```

These weights are a tested experiment profile, not a universal default.

## Choosing `static` vs `dynamic`

`static` specializes on exact tensor shapes, so it can only run against a
bounded signature set -- hence the palette requirement. `dynamic` emits
shape-generic kernels and handles the natural sampler as-is.

Measured on a separate 4xH200 box (8.5M/24-layer for compute,
5.9M/16-layer for wall-clock):

| | regional compute gain | real training wall-clock | cold build | palette required |
|---|---:|---:|---:|:--:|
| eager | - | 943 ms/step | - | - |
| + native RMSNorm only | +10.2% | 876 ms/step (+7.7%) | - | no |
| shape palette only, eager | - | 856 ms/step (+10.2%) | - | **yes** |
| `dynamic` | +28.6% | 816 ms/step (+15.6%) | ~10 min | no |
| `static` | +40.6% | 734 ms/step (+28.4%) | ~21 min | **yes** |

Compute is roughly 77% of a real training step -- both the RMSNorm and the
static-compile arms imply that independently -- so compute-side gains arrive at
wall-clock discounted by about a quarter.

End to end `static` is ahead, 734 vs 816 ms/step. But almost none of that gap is
the compiler. Isolating each effect:

| effect | gain |
|---|---:|
| `dynamic` compiler alone (natural shapes, A -> E) | +15.6% |
| `static` compiler alone (curriculum held fixed, C -> D) | +16.6% |
| shape palette alone, no compiler change (A -> C) | +10.2% |

**The two compilers are within about one point of each other.** `static`'s
12.8-point lead over `dynamic` is the palette sampling cheaper tables, which the
eager palette-only arm demonstrates on its own.

So the real choice is not which compiler but whether to accept the curriculum
change. **`dynamic` is the default in `scripts/*.sh`**: it delivers essentially
the whole compiler win with no change to what the model trains on. Enable
`static` plus the palette when you have validated that the nine-shape curriculum
does not cost model quality -- that is worth roughly another 10 points, and it is
a data decision, not a performance one.

Caveat: the wall-clock arms are single runs, not a multi-seed performance
study. An earlier pass used the trainer's default `{:.1f}` throughput logging,
which quantizes to ~+/-4.5% at 1.1 steps/s;
those numbers were withdrawn.

## Build the cache

The cache builder loads a checkpoint only to recover architecture and parameter
shapes. Compiled graphs do not contain those particular weights and can be used
by a newly initialized model with the same architecture.

```bash
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor/nori-training-static-b20"

CUDA_VISIBLE_DEVICES=4,5,6,7 \
COMPILE_TEMPLATE_CHECKPOINT=/path/to/checkpoint.pt \
BATCH_SIZE=20 \
GRAD_CKPT_THRESHOLD=24576 \
bash scripts/precompile_training_shapes.sh
```

Override `COMPILE_SHAPES` for another signature set:

```bash
COMPILE_SHAPES='128x16@0.4,128x16@0.7,512x64@0.4,512x64@0.7' \
bash scripts/precompile_training_shapes.sh
```

The precompiler runs under DDP because the cache must match the training graph
boundary. It writes `precompile_manifest.json` into the cache directory.
The wrapper
delegates to `benchmarks/bench_training_compile.py`, which can also be run directly
to compare eager, regional dynamic, and regional static execution.

## Train with the cache

Use the same cache directory, batch size, gradient-checkpoint threshold, native
RMSNorm setting, architecture, PyTorch/CUDA stack, and GPU architecture. The
snippet below prepares the cache environment and static-compile arguments;
append the resulting arguments to your existing `synthefy-nori-train` or
`torchrun` invocation. The public shell launchers deliberately do not enable a
shape palette for you:

```bash
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor/nori-training-static-b20"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1

SHAPE_PALETTE='128x8:0.240,128x48:0.135,128x128:0.126,512x8:0.115,512x48:0.064,512x128:0.061,1536x8:0.124,1536x48:0.079,1536x128:0.056'

STATIC_COMPILE_ARGS=(
  --shape-palette "$SHAPE_PALETTE"
  --context-ratio-palette 0.4,0.7
  --native-rms-norm
  --ema-foreach
  --compile-encoder-layers static
  --compile-mode default
  --compile-cache-limit 1024
  --compile-disable-ddp-optimizer
)
```

Add `--skip-zero-feature-decoder` only when feature loss remains zero and
unused-head freezing remains enabled.

The first use in a new process still loads and attaches cached artifacts. It
should not repeat cold code generation and autotuning.

## Additional implementation measurements

These earlier measurements used a 9.8M-parameter, 28-layer model with batch 20
on four H200s. They characterize individual components and prototypes; use the
matched wall-clock table above to choose the current regional compile mode.

| Intervention | Measured result |
|---|---:|
| Native RMSNorm | 8.8% higher four-GPU throughput in the matched phase benchmark. |
| Skip zero-loss feature decoder | About 1.6% end-to-end improvement. |
| Exact static compile, fixed `256x64` | Model forward/backward improved from 359.9 ms to 291.3 ms (23.5% compute throughput). |
| Exact static compile, fixed `256x64` | Full step improved from 542.9 ms to 472.6 ms (14.9% end-to-end throughput). |
| Early generic dynamic-compile prototype | 6.4% slower than eager; superseded by the current regional dynamic path. |
| Whole-stack static compile under DDP | Slower than regional compilation; not exposed by the training CLI. |

The complete 18-signature cold build produced:

```text
18 successful signatures
504 AOTAutograd entries (18 signatures x 28 layers)
1.6 GiB persistent cache
24.5 minutes total cold build time
96.1 GiB maximum allocated GPU memory
```

Disabling Dynamo DDPOptimizer for regional compilation was about 10% faster in
the exact DDP benchmark and allowed AOTAutograd disk-cache hits across fresh
processes. Each regional layer is already smaller than a DDP bucket, so the
extra graph splitting did not improve communication overlap in this model.

## Scientific rollout

The shape palette changes the data curriculum. Compilation and curriculum
quality must therefore be evaluated separately:

1. Compare eager and compiled runs using the identical explicit palette, seeds,
   architecture, optimizer, and evaluation checkpoints. This isolates execution.
2. Compare the best palette run against the unrestricted sampler. This measures
   any quality impact from shape discretization.
3. Require exact-step raw-evaluation parity before enabling the profile broadly.
4. Retain eager fallback until multiple models, GPU types, and resume paths have
   completed successfully.

Do not interpret a faster compiled palette run as a pure systems ablation against
an unrestricted-shape control.

## Cache contract and limitations

Treat compiled artifacts as disposable build products. Rebuild when any of the
following changes:

- model architecture or compiled layer code;
- batch size, table shape, context split, or checkpointing state;
- dtype or compiler flags;
- QASS mode and attention-backend environment settings;
- PyTorch, Triton, CUDA, driver, or GPU architecture.

The cache is not committed to Git. If an expected signature is missing, Dynamo
can compile it lazily, which may pause training for a minute or more. Monitor the
first occurrence of every signature and keep the generated manifest with run
provenance.
