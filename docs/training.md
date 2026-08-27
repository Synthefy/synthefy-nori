# Training

## The supported interface

All reviewed training runs start from a committed YAML recipe through one
entrypoint:

```bash
scripts/train plan configs/training/production/<recipe>.yaml
scripts/train launch configs/training/production/<recipe>.yaml
```

For a recipe with `batching.policy: planned`, `plan` is an optional GPU
preflight: it builds the configured model, profiles every table-shape bucket on
the lowest-memory requested device, and writes one shape-memory table named for
the recipe. It never starts W&B or training. `--reprofile-memory` deliberately
replaces that table. `--no-profile-memory` only inspects static resolution.

`launch` first uses a structurally valid plan from the machine-local cache. If
there is no local plan, it can use a committed hardware plan when every requested
GPU has the recorded accelerator name and at least the recorded memory capacity.
For example, the 10M production recipe ships its measured H200 table, so an H200
machine can launch immediately after pulling the reviewed commit. Other GPU
models remain supported: if no matching committed or local plan exists, launch
uses the configured per-rank logical capacity as the physical batch. There is no
code/config/software fingerprint and no requirement to run `plan`; the
researcher launching an unprofiled or stale recipe owns the OOM risk.

Do not launch production training by constructing a raw `torchrun` command.
Agents may edit a recipe, run `plan`, inspect manifests, and execute the reviewed
launcher after a researcher approves the resolved launch summary. After that
approval, `launch --yes` records the same manifest and starts without requiring
the researcher to open a terminal or type a digest token.

`launch` refuses to run when any of these conditions is false:

- the YAML is strict and matches the pinned trainer-CLI contract;
- its declared architecture builds the exact expected parameter count;
- the recipe has `status: ready` and no blockers;
- the config is tracked, the worktree is clean, `HEAD` is attached to a branch,
  and it exactly matches that same-named branch's current tip on `origin`
  (verified without credential prompts);
- the core project environment matches the frozen `uv.lock` resolution;
- no CI environment marker is present; an agent marker is accepted only with
  `--yes`, which represents approval collected outside the terminal;
- CUDA is available and at least the requested number of devices is visible.
  The launcher preserves a scheduler/user `CUDA_VISIBLE_DEVICES` allocation
  instead of selecting or widening it, and it does not require a GPU model;
- either a human types the first 12 characters of the resolved manifest digest,
  or `--yes` records that explicit approval was collected elsewhere.

There is no raw-arguments escape hatch. `--yes` skips only the terminal and
agent gates; it does not bypass recipe status, pushed-clean-tree provenance,
environment-lock, CUDA-capacity, memory-plan, or validation-data checks. The
underlying public `synthefy-nori-train` CLI remains available for code-level
debugging, but it is not a production recipe interface.

## Why the YAML is reproducible

A recipe separates the parts researchers need to reason about:

- Git branch is launch provenance, not a scientific recipe setting. The launcher
  derives the checked-out branch, records its full `refs/heads/...` ref and exact
  commit in the manifest, and accepts `main` or any research branch whose clean
  `HEAD` matches its same-named branch on `origin`.
- `architecture` pins the architecture explicitly. A label such as “10M” never
  selects dimensions implicitly. The launcher builds the real model on CPU and
  derives its parameter count for `plan` and the launch manifest; recipe authors
  do not calculate or maintain a second expected-count field.
- `training` contains reviewed trainer options. Unknown keys fail. Trainer
  defaults are covered by `cli_contract_digest`, so a changed default forces a
  deliberate recipe review instead of silently changing a run.
- `batching` pins the logical global optimizer batch independently of physical
  GPU execution. Its optional planning limits define the largest microbatch to
  probe and the HBM headroom to retain; researchers do not hand-author a shape
  table. Reviewed hardware plans live under
  `configs/training/production/memory-plans/<recipe>/` and are selected only for
  matching accelerators.
- `launcher` pins a default world size, output root, and rendezvous port. The
  world size is an operational override; the launcher accepts any CUDA GPU
  model and records the actual runtime inventory in the manifest.
- `environment` pins semantic environment inputs and scrubs undeclared
  `SYNTHEFY_*`, NCCL, and W&B overrides. It preserves `CUDA_VISIBLE_DEVICES`
  from the scheduler or user while pinning the other CUDA behavior used by the
  recipe. Import-time attention settings are explicit too; the legacy
  `--no-flash-attn` flag alone does not control FlashAttention discovery.
- QASS mode and scaling live under `architecture`, even though the current model
  implementation receives them through import-time environment variables.
- `training.icl_filter_model` records the selected ICL-filter path or alias. The
  supported production contract does not require content-versioning the shared
  `limix` alias.
- `validation` is either `null` or one opaque name from the eval-owned suite
  registry. It can never contain datasets, folds, paths, preprocessing, or
  metrics. `validation_interval_steps` pins when production training pauses at
  an optimizer boundary to checkpoint and invoke that suite; zero disables the
  automatic callback while retaining the named suite for manual checkpoint
  evaluation.
- `overrides.allowed` is an explicit per-recipe scalar allowlist. Every applied
  override is written into the manifest. The launcher globally limits this to
  GPU process count, output root, rendezvous port, W&B mode, and disabling the
  automatic validation interval; a recipe cannot allow ad hoc LR, model, data,
  schedule, or named-suite selection overrides.
- online W&B runs pin the shared entity, project, group, job type, and generated
  run name. The creating user is the owner of the `WANDB_API_KEY` in the shell
  that launches (`wandb login`); Aditya's login creates an Aditya-authored run
  and Billybob's login creates a Billybob-authored run in the same team project.

The launch directory contains read-only `launch-manifest.json`, `resolved.yaml`,
and `command.txt` before `torchrun` starts. When a memory plan is selected, it
also contains a read-only `memory-plan.json`. The manifest records the source
and recipe digests, exact argv, git state, dependency-lock digest, parameter
count, global batch, LR reference/scale/optimizer value, suite digest, redacted
environment, runtime versions, and the selected plan's content digest. Resume
reuses that exact run-local plan. The W&B run config records the resolved
shape-to-microbatch mapping, so throughput curves can be interpreted without
access to the training host.

The recipe authors `batching.target_global_batch_size`; this is the number of
valid tables contributing to every optimizer update regardless of table shape
or world size. At runtime, the trainer generates one logical local batch, looks
up that shape's safe physical microbatch, and performs sample-weighted
backwards with DDP `no_sync()` until the logical target is complete. Data
preparation, permutations, and masking are sampled once at the logical-batch
boundary before slicing, so changing hardware does not change the recipe's RNG
stream. A global batch of 160 on eight ranks has local capacity 20: a shape
planned at microbatch 3 executes `3+3+3+3+3+3+2` locally and still produces
exactly one 160-table optimizer update. Non-divisible world sizes use
zero-weight dummy slots on the final collective so every rank performs the
same number of backwards.

`training.lr` is calibrated at `training.lr_reference_batch_size`, and the
trainer uses:

```text
optimizer_lr = lr × global_batch / lr_reference_batch_size
```

Production recipes must pin the reference batch. The underlying debug CLI does
not apply any implicit DDP scaling when the reference is omitted. Changing the
rank count no longer changes either the logical global batch or LR; it changes
the generated local capacity without forcing a new plan.

The profiler measures complete steady-state training work: forward, loss,
backward, optimizer state, and EMA, across every task/context variant declared
by the recipe. A binary search verifies the largest safe microbatch instead of
assuming memory is perfectly linear. Successful probes above the explicit HBM
ceiling are rejected, preserving the configured headroom. Planning uses
deterministic synthetic tensors in an isolated process, so it cannot consume
the eventual run's data or RNG state. The production trainer disables reactive
checkpoint retry and shape blacklisting under this policy. Planning is an
operator-controlled performance tool, not a launch compatibility gate.

Inspect the complete record without launching:

```bash
scripts/train plan configs/training/production/<recipe>.yaml --json
```

Operational overrides use dotted scalar paths and only work when the recipe
allows them:

```bash
scripts/train plan CONFIG.yaml \
  --set launcher.nproc_per_node=4 \
  --set launcher.master_port=29501 \
  --set launcher.output_root=/mnt/checkpoints
```

For the 10M production recipe, automatic validation can be disabled without
changing checkpoint cadence or the pinned suite:

```bash
scripts/train launch configs/training/production/10m_v6.yaml \
  --set validation_interval_steps=0
```

For a phone-approved agent launch, first review the agent's resolved summary,
then approve it in the conversation. The agent runs the same command with
`--yes`; no terminal token is needed:

```bash
scripts/train launch configs/training/production/10m_v6.yaml --yes
```

`nproc_per_node` may be any positive count that fits the visible CUDA
allocation. If `CUDA_VISIBLE_DEVICES` is set by a scheduler or the user, it is
passed through unchanged and `torchrun` uses the first requested number of
logical devices from that allocation. If it is unset, `torchrun` uses the first
requested GPUs visible to CUDA. The launcher checks capacity but never checks
whether those devices are H200s, H100s, A100s, or another CUDA model. Changing
the rank count preserves the reviewed global optimizer batch and LR while
changing the local logical capacity. It does not force a new plan.

Changing model, data, optimizer, schedule, or named-suite semantics should be a
reviewed YAML change, not an override.

## Named checkpoint validation

Real data and scoring stay in the eval module, not the gradient loop. At each
positive `validation_interval_steps`, every DDP rank pauses, rank 0 writes the scheduled
immutable checkpoint, releases cached training activations, and invokes
`scripts/train validate` synchronously. Training continues only after the full
named suite succeeds and its metrics are acknowledged by the original W&B run;
an incomplete suite or upload failure aborts the launch rather than silently
skipping validation.

The same command is the idempotent manual retry/debug interface. It takes the
manifested run directory—not a bare config—so it can verify checkpoint
ownership and attach metrics to the exact W&B run:

```bash
scripts/train validate checkpoints/production/<run> --checkpoint checkpoints/production/<run>/checkpoint_step_1000.pt
```

The training side passes only the suite name and checkpoint to
`synthefy_nori.evaluation.named_suites`. The suite definition owns exact OpenML
task IDs, exact fold IDs, caps, imputation, inference preprocessing, metrics,
and fold/dataset aggregation. To compare researchers fairly, add a new reviewed
file under `src/synthefy_nori/evaluation/suite_defs/` and give it a versioned
name such as `billybob_val_50_v1`; never add a task/fold override to the train
config.

Only `tabarena_fold0_v1` and `openml_ctr23_fold0_v1` are declared initially
because those are the reviewed subsets needed for the first production recipe
and its immediate comparison. The named-suite adapter also composes the existing
TALENT, BeyondArena, ScoringBench, and manifest-backed directory loaders. The
directory form covers customer/POC benchmarks such as Augury, GoodRx, and
MegaFood without moving their manifests, split preparation, or metrics into a
training config. A RamanBench definition can use that adapter if it has the
standard eval manifest; otherwise its canonical loader must be added to the eval
framework first. We deliberately do not invent dataset memberships for any of
these: each desired subset becomes a separately reviewed, versioned suite YAML.

Each output directory is bound to the suite digest, checkpoint hash, recipe
digest, inference policy, and eval code state before crash-resume is enabled.
Reusing that directory for a different checkpoint or code state fails instead
of returning stale rows. The suite withholds its primary score unless every
declared task/fold succeeds; partial results retain a clearly labeled
`available_score` for diagnosis only and `scripts/train validate` exits nonzero.
The default output path is deterministic for a run/checkpoint/recipe, so
rerunning the same command resumes it safely.

For online runs, the trainer is the primary writer to its deterministic W&B
run ID and validators join as non-primary shared writers. A validator cannot
finish the training run or replace its config/system metadata. Complete suites
log the macro score and per-dataset scores under `validation/`, using
`validation/checkpoint_step` as their custom x-axis. That step comes from the
checkpoint's serialized `optimizer_step`, so a manual retry remains plotted at
the checkpoint it measured. The validator never publishes the partial
`available_score`.

An online launcher-managed run aborts if its primary W&B writer cannot
initialize; it never spends a training allocation while silently dropping
telemetry. Choose `wandb.mode: disabled` explicitly for an intentionally local
run.

Shared W&B writers cannot use W&B's implicit global step safely. Trainer curves
therefore use `train/optimizer_step` as their custom x-axis as well. The trainer
and each validator write independent history rows to one run without either
writer changing the other's x-axis.

The trainer also publishes explicit progress counters so the logical batch is
not confused with its physical slices:

- `batching/global_optimizer_batch_size` — reviewed logical batch size;
- `progress/global_optimizer_batches_completed` — cumulative optimizer updates;
- `progress/global_tables_processed` — completed updates times global batch;
- `throughput/global_optimizer_batches_per_sec` — full logical batches per second.

After W&B acknowledges the point, the output directory receives a deterministic
`wandb-log.json` receipt. Ordinary reruns skip an identical completed upload;
an upload failure exits nonzero while retaining the resumable eval files. Runs
launched with W&B offline or disabled retain the same local eval artifacts but
do not attempt an online upload.

List or inspect the committed suites without downloading data:

```bash
scripts/train suites
scripts/train suites tabarena_fold0_v1
```

The existing `synthefy-nori-eval` CLI remains supported for its documented
public evaluation workflow. Named suites are an additive checkpoint-validation
contract; they do not replace or change that CLI.

## Resume

Resume from the latest periodic checkpoint in a manifested run:

```bash
scripts/train resume checkpoints/production/<run>
```

Or select an exact checkpoint and bound only this invocation:

```bash
scripts/train resume checkpoints/production/<run> \
  --checkpoint checkpoints/production/<run>/checkpoint_step_210400.pt \
  --run-steps 190000
```

Resume requires the original git commit, a lock-synchronized environment,
the clean-tree/approval/CUDA-capacity production gates, and the original
run-local memory-plan digest. `resume --yes` may reuse the original approval for
an exact continuation of the same manifested run; changing the run, GPU scope,
or training bound requires a new approval.
It deliberately does not require that historical commit to remain the current
tip of its original branch; otherwise a long run would become unresumable as
soon as that branch advanced. It hashes the checkpoint and writes a separate
read-only resume manifest; it never edits the original launch record.
The initial manifest also pins a W&B run ID. Resume reuses that ID with
`WANDB_RESUME=must`, so a continuation cannot silently create a second curve
with the same display name.

## Current 10M production recipe

`configs/training/production/10m_v6.yaml` is the ready main-line 10M recipe. It
pins the 9,736,585-parameter architecture, the current Synthetic-v6 curriculum,
the natural Tier-1 shape distribution, regional dynamic encoder compilation,
and a default of eight processes to reproduce the reference 8xH200 setup. This
is not a hardware gate: any positive visible CUDA GPU count and any CUDA GPU
model may be used.
The recipe pins a logical global batch of 160 and a maximum probed local
microbatch of 24; `plan` derives the safe value for each Tier-1 shape on the
actual allocation. Main v6's historical LR calibration (`8e-4` at global batch
96) scales to optimizer LR `1.3333333e-3`.

The execution-only Speed3 options are enabled: regional dynamic TorchInductor
compilation, native RMSNorm, foreach EMA, and omission of the permanently
unused feature decoder. The persistent compiler cache and four-worker-per-rank
cap avoid the multi-rank static-precompile failure described in the
acceleration guide. Gradient checkpointing, reactive OOM retry, and OOM shape
blacklisting remain disabled; planned physical batching is this recipe's only
normal activation-memory policy.

The old descendant-only synthetic options and W&B command reconstruction are
not part of the production suite. Comparisons with historical checkpoints use
the same named suite and W&B namespace; retained checkpoints can still be
evaluated manually.

## Execution details

The configuration exposes compiler mode, shape/context palettes, mixed
precision, checkpointing, prefetching, Muon routing, and logging cadence as
ordinary reviewed fields. Static compilation is a data-recipe decision because
it requires a fixed shape palette; it is not interchangeable with dynamic
compilation. See [the acceleration guide](training_static_compile.md) for the
kernel/cache measurements and caveats.

Terminal `log_interval` and W&B `wandb.log_interval` are independent. The
launcher records both. W&B receives the timestamped launch name and stable
recipe group, so repeated launches remain distinguishable without changing the
scientific recipe digest. `training.curriculum_tier` is also pinned separately:
the legacy trainer can infer it from a tier-like W&B job type, but disabling
W&B through an operational override must not change the data curriculum.
