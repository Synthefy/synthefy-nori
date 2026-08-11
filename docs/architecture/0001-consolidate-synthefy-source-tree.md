# ADR 0001: Build `synthefy` and `synthefy-nori` from one source tree

- **Status:** Accepted for internal implementation
- **Date:** 2026-08-11
- **Issue:** [Synthefy/nori-monorepo#216](https://github.com/Synthefy/nori-monorepo/issues/216)
- **Inputs:** [`0001-consolidate-synthefy-source-tree.json`](0001-consolidate-synthefy-source-tree.json)

## Context

Nori's local runtime and the lightweight `synthefy` client are developed in
separate repositories. Features such as forecasting, text preparation, and
explainability cross that boundary: a change can require coordinated client,
local-runtime, serving-contract, and documentation work. Contributors currently
reconstruct the combined workspace with editable installs or candidate Git SHAs,
and CI has cross-repository lanes to detect drift after the fact.

The repository boundary should change without changing what users install. A
hosted user must still be able to install a lightweight SDK without Torch or model
weights, while a local user can install the full Nori runtime.

## Decision

### One repository, two distributions

The Nori source repository will contain two independently built and versioned
Python projects:

```text
synthefy-nori/
├── pyproject.toml
├── src/synthefy_nori/
└── libs/synthefy/
    ├── pyproject.toml
    ├── src/synthefy/
    ├── tests/
    ├── README.md
    ├── CHANGELOG.md
    └── LICENSE
```

The root project continues to produce `synthefy-nori`. The project under
`libs/synthefy/` produces `synthefy`. Their wheels own disjoint namespaces and
must contain no overlapping files.

Published dependency metadata points in one direction:

```text
synthefy-nori  ->  synthefy
```

Base `synthefy` must neither depend on nor import `synthefy_nori`. The single
allowed runtime edge in the other direction is an explicit `mode="local"` call,
which may lazily import the installed local runtime after normal `synthefy`
imports have completed.

### Preserve the public Nori contract

`SynthefyNoriClient` remains the public Nori-specific facade. A future async
counterpart may be named `AsyncSynthefyNoriClient`; broader `SynthefyClient` or
`SynthefyEnterpriseClient` names remain available for a future multi-service
client.

The migration preserves the existing `NoriPredictRequest` and
`NoriPredictResponse` names. Refactoring or relocating those types is not an API
rename. The hosted request remains the numeric `X_train`, `y_train`, and `X_test`
contract.

Backend selection is explicit: the supported modes are `remote`, `sagemaker`,
and `local`. `mode="auto"` is removed in the `synthefy` 7.0 cutover rather than
retained as an alias. The initial consolidated client remains one-shot; bound
sessions, context reuse, and their warning semantics require a separate design.

### Packaging and compatibility

The target Python floor for both distributions is 3.9. Before either project
declares that floor, the combined dependency graph, builds, clean installs,
imports, and core tests must pass on Python 3.9. A concrete blocker must be
recorded and explicitly approved before raising either floor.

`pip install synthefy` must remain Torch-free. Optional text dependencies may
bring Torch transitively, but base dependencies and base imports may not. Local
installation is expressed by installing `synthefy-nori`, avoiding a published
dependency cycle through a `synthefy[local]` extra.

### Source import and provenance

The standalone client is imported from the immutable commit and tree recorded in
the adjacent JSON manifest. The import is one reviewable snapshot commit with the
source repository and commit in its message. It is not a subtree merge or history
graft: internal, staging, and public must retain a linear promotion history.

The source declares MIT metadata and includes `LICENSE` in its sdist definition,
but the pinned tree has no matching license file. Importing the snapshot does not
resolve that discrepancy. Publishing either consolidated artifact is blocked
until the grants, notices, and per-wheel license contents have been reviewed and
made complete.

### Development and release order

All consolidated source-tree and serving implementation is authored, integrated,
and validated in `Synthefy/synthefy-nori-internal` first. The same public-ready
patch is then promoted in order:

```text
internal  ->  staging  ->  public
```

Staging and public are promotion gates, not development branches. A defect found
there is fixed in internal and re-promoted. Only the public repository may publish
the two distributions, selected by package-specific tags such as
`synthefy-v7.0.0` and `synthefy-nori-v0.17.0`.

Containment, publisher, PyPI, and customer-documentation changes happen in the
repository or service that owns each surface, at the gates below. They do not
authorize separate implementations of the consolidated source tree.

This is an L1 package/source cutover that preserves the existing L2 hosted API.
It does not change checkpoints or Hugging Face weights, model slugs, gateway
bindings, pricing, usage dimensions, billing allowlists, key limits,
entitlements, console pricing, or IaC behavior.

## Foundation gates

Before the source snapshot is imported:

1. Freeze standalone-client feature work and mark overlapping pull requests as
   blocked by #216. Only an approved critical production fix may change the
   recorded source commit before import.
2. Disable and delete the `synthefy-package` subtree-sync workflow.
3. Quarantine its stale vendored client source so it cannot sync, publish, or be
   treated as an editable source of the public SDK.
4. Revoke or rotate `SYNTHEFY_PUBLIC_REPO_TOKEN` after removing the workflow.
5. Approve the source and license treatment before copying files; do not infer a
   license from package metadata. Artifact publication remains blocked until the
   per-distribution grants and notices are complete.

Before production cutover:

1. Inventory every unbounded `synthefy` consumer; migrate or retire it, or pin it
   explicitly to `synthefy<7`.
2. Rehearse `synthefy` alone, `synthefy` 7 with the currently released
   `synthefy-nori`, and both candidate wheels together.
3. Verify the correct grants and notices in both wheels and sdists.
4. Configure distinct TestPyPI and PyPI trusted-publisher workflow/environment
   pairs for both new namespaced public workflows. At cutover, revoke the
   standalone `Synthefy/synthefy` `publish.yaml` / `pypi` identity and the public
   Nori repository's generic `publish.yml` / `testpypi` and `pypi` identities;
   archiving or renaming a workflow is not revocation.
5. Publish and verify `synthefy` 7 first. If verification fails, stop and fix
   forward with a new SDK version; do not publish `synthefy-nori` 0.17. Publish
   and verify the heavy distribution only after its `synthefy>=7,<8` dependency
   is resolvable.
6. Verify the existing gateway binding and billing allowlist, and prove that the
   unchanged `usage` response is still metered.
7. Use an ephemeral dev key for hosted smoke tests and always delete it. Any
   production key or production smoke test requires explicit approval.
8. Merge customer documentation only after the published install commands
   resolve from PyPI.

## Consequences

- One checkout and one source SHA can validate the lightweight workflow and
  heavy local/runtime halves together.
- Users still choose between lightweight and local installations.
- Release automation becomes more explicit because one repository publishes two
  independently versioned distributions.
- The standalone client repository remains active until the published pair and
  hosted smoke tests pass, then becomes a migration notice and archive.
- Bound sessions, async APIs, hosted artifact metadata, and other post-foundation
  features are not prerequisites for this migration.

## Alternatives not chosen

- **Keep two source repositories:** preserves isolation but keeps coordinated
  development, candidate-SHA CI, and atomic validation as contributor-owned
  setup.
- **Move only workflows to the standalone client:** leaves local runtime,
  serving, and release-contract changes across the same repository boundary.
- **Use overlapping wheel namespaces:** makes file ownership and uninstall
  behavior ambiguous. The two distributions instead own disjoint namespaces.
- **Merge or graft the standalone history:** conflicts with the linear promotion
  cascade. Immutable snapshot provenance provides traceability without merge
  commits.
