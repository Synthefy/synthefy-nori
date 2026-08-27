# Training acceleration and optional static-shape compilation

This guide covers Nori's execution-only training speedups and the experimental
static-shape path. The underlying CLI controls remain opt-in. The reviewed 10M
V6 production recipe enables the curriculum-neutral subset: native RMSNorm,
foreach EMA updates, regional dynamic compilation, and omission of the
permanently unused feature decoder.
These options preserve the sampler, shape curriculum, and training
hyperparameters; mixed-precision rounding can still differ at bf16 scale
because the kernels reduce in a different order.

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

All flags are off by default **in the underlying CLI**. Reviewed YAML recipes
pin each value explicitly through `scripts/train`; there are no environment
switches that can mutate a production recipe. Static palettes belong in the
reviewed recipe because they change the data curriculum.

`--skip-zero-feature-decoder` requires the feature
loss to remain zero and unused-head freezing to remain enabled. It is
never inferred by the launcher: a recipe must enable it together with a
permanently zero feature-loss schedule, or the underlying CLI rejects the run.
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
change. A reviewed production YAML must explicitly pin dynamic or static mode;
there is no shell-script default. Enable `static` plus a palette only after
validating that the discretized curriculum does not cost model quality -- that
is worth roughly another 10 points, and it is a data decision, not a performance
one.

Caveat: the wall-clock arms are single runs, not a multi-seed performance
study. An earlier pass used the trainer's default `{:.1f}` throughput logging,
which quantizes to ~+/-4.5% at 1.1 steps/s;
those numbers were withdrawn.

## Cache policy

The old generic precompile shell helper was removed. It hard-coded architecture
and import-time environment settings independently from the training recipe,
which made it possible to warm a cache under a different QASS, SDPA, native
RMSNorm, dtype, or DDP contract than the run that consumed it.

Compiled recipes pin a persistent TorchInductor cache directory, graph-cache
controls, and compiler-worker count under `environment`. These are operational
settings, but pinning them makes `plan` and `launch` share the same artifacts
instead of relying on PyTorch's volatile `/tmp` default. The YAML also pins the
settings that affect compilation semantics: shape/context palettes,
microbatching policy, checkpointing, native RMSNorm, compiler mode/cache limit,
and DDP-optimizer setting.

When `scripts/train plan CONFIG.yaml` must profile planned microbatches, it does
so on exactly one GPU and also populates the compiler cache. A matching bundled
memory plan may skip that profile, in which case launch builds whichever
regional dynamic guard variants its encountered shapes require. The variant
count is workload-dependent; it is not a fixed part of the recipe contract. A
bounded four-rank empty-cache smoke completed in about 55 seconds with the
four-worker-per-rank cap, but that fixed-shape smoke does not bound cold-start
time for the natural-shape curriculum. Do not reconstruct a raw multi-rank
static precompile or `torchrun` command.

### Why the old multi-rank static warmup hung

A reproduced four-rank warmup completed the small and medium signatures, then
stopped for roughly three hours. NCCL showed ranks 0, 2, and 3 waiting in
all-reduce sequence 40 while rank 1 had only enqueued sequence 39. Rank 1 was
still compiling or executing its local graph while the peers had entered the
backward collective. The watchdog eventually terminated the job. At the same
time, each rank could spawn PyTorch's default compiler-worker pool and contend
for the same filesystem cache.

Offline multi-signature cache generation is local code generation; it does not
need DDP and should use one process. The standalone compile benchmark now
rejects compiled multi-rank runs by default for this reason. Its explicit
`--allow-distributed-compile` escape hatch is for deliberate DDP timing and
cold-start qualification. The production path is different: it uses one
regional dynamic function, a small workload-dependent set of guard variants, a
persistent cache, and four compiler workers per rank. The bounded four-rank
empty-cache path completed successfully. An independent eight-rank
natural-shape run also completed 160 optimizer updates and checkpointed, but
first-time code generation recurred as new guards were encountered. Replaying
the populated cache shortened those pauses without eliminating process-local
AOTAutograd setup. Treat these as finite cold-start pauses, not steady-state
throughput.

The old fixed-batch warmup was also incompatible with planned microbatching. A
hard-coded batch of 20 exceeded the safe memory plan on the largest signatures
and triggered checkpoint retries. The planner now measures the physical
microbatch for each shape and compiled qualification keeps checkpointing and
reactive OOM recovery disabled.

Do not confuse a slow single-GPU `plan` with this deadlock. A cold plan may be
quiet for several minutes while it initializes the configured LimiX filter and
builds the first dynamic graph. It then profiles every physical shape across
all configured context ratios and candidate microbatches; the complete
62-shape Tier-1 memory plan is intentionally a long, one-time hardware
qualification. Healthy output advances through `[MEMORY-PLAN] i/62` on one
GPU. A failed distributed warmup instead leaves several ranks in an NCCL
collective with another rank on an earlier sequence number.

### Dynamic signature behavior on current main

The regional dynamic compile was tested across changing row count, feature-token
count, context split, and planned microbatch size. A guard-level test with the
current H200 plan's microbatches, which are all at least two, recorded one
regional-forward graph across those dimensions; batch size one formed one
additional guard variant. A full single-GPU forward/backward smoke test over
`128x8@0.4`, `512x48@0.7`, and `1536x128@0.4` produced two total Dynamo graphs,
spent about 59 seconds on cold compilation, and then ran the three cache-hit
steps in 0.24--0.35 seconds each. That count is not universal: a separate exact
final-checkpoint benchmark over five representative shapes recorded three
Dynamo graphs, and the natural-shape trainer encountered cached signatures
progressively. Do not hard-code a graph count or assume that a fresh process
will be pause-free merely because its FX cache is populated. These tests show
that dynamic compilation can reuse graphs across shapes instead of necessarily
creating the full Cartesian product of the 62-shape curriculum and context
ratios.

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
