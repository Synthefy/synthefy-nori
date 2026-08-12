# Releasing `synthefy` and `synthefy-nori`

This repository builds two independently versioned Python distributions. Source
work is completed in `Synthefy/synthefy-nori-internal` first, then the same
public-ready patch is promoted through staging to public:

```text
internal  ->  staging  ->  public
```

Internal and staging build and test both artifacts, but cannot upload either one.
Only `Synthefy/synthefy-nori` is a release authority.

## Release streams

| Distribution | Project root | Version files | Tag | Workflow |
|---|---|---|---|---|
| `synthefy` | `libs/synthefy/` | `libs/synthefy/pyproject.toml`, `libs/synthefy/src/synthefy/__init__.py`, changelog | `synthefy-vX.Y.Z` | `publish-synthefy.yml` |
| `synthefy-nori` | repository root | `pyproject.toml`, `src/synthefy_nori/__init__.py` | `synthefy-nori-vX.Y.Z` | `publish-synthefy-nori.yml` |

A GitHub Release for one namespaced tag starts both workflows, but only the
matching workflow builds. The old unnamespaced `vX.Y.Z` publisher is removed.
Neither workflow consults the repository-wide "latest release."

## Required order

Publish the lightweight SDK first. The heavy package declares
`synthefy>=7,<8`, so publishing it before a compatible SDK exists on PyPI would
create an unresolvable release.

1. Publish and verify `synthefy 7.0.0` (or the selected fix-forward version).
2. Publish and verify `synthefy-nori 0.17.0`.
3. Merge customer documentation only after both documented install commands
   resolve from PyPI.

Do not upload new Hugging Face weights, change gateway slugs, pricing, billing,
limits, or entitlements for this package-source cutover.

## Before promotion

- Complete and validate the candidate in internal.
- Run the offline package, namespace, candidate/released-client, OpenAPI, and
  packaged-container gates.
- Complete the explicitly assigned final Baseten-dev and SageMaker validation;
  record any human handoff rather than treating a skipped live test as passed.
- Confirm the existing gateway binding, billing allowlist, and unchanged
  `usage` response remain valid.
- Keep the customer docs PR staged, not merged.

Then promote the identical public-ready patch to staging, run every gate there,
and promote to public last. Wait for rebase-sync/drop-check before tagging.
Defects found in staging or public are fixed in internal and re-promoted.

## Prepare a version

For `synthefy`:

1. Set the exact version in `libs/synthefy/pyproject.toml` and
   `libs/synthefy/src/synthefy/__init__.py`.
2. Add the matching section to `libs/synthefy/CHANGELOG.md`.
3. Use tag `synthefy-vX.Y.Z`.

For `synthefy-nori`:

1. Set the exact version in `pyproject.toml` and
   `src/synthefy_nori/__init__.py`.
2. Use tag `synthefy-nori-vX.Y.Z`; GitHub Release notes are its changelog.

The workflows verify tag, project metadata, module `__version__`, and built wheel
metadata agree. They also require the peeled tag commit to be an ancestor of
public `main`, so a tag from an unpromoted branch cannot publish.

## Rehearse on TestPyPI

Manual dispatch is deliberately tag-bound. Supply distribution, version, target,
and tag explicitly, and select that same existing tag as the workflow ref:

```bash
gh workflow run publish-synthefy.yml \\
  --repo Synthefy/synthefy-nori \\
  --ref synthefy-v7.0.0 \\
  -f distribution=synthefy \\
  -f version=7.0.0 \\
  -f target=testpypi \\
  -f tag=synthefy-v7.0.0

gh workflow run publish-synthefy-nori.yml \\
  --repo Synthefy/synthefy-nori \\
  --ref synthefy-nori-v0.17.0 \\
  -f distribution=synthefy-nori \\
  -f version=0.17.0 \\
  -f target=testpypi \\
  -f tag=synthefy-nori-v0.17.0
```

A branch ref, mismatched tag/version, wrong distribution, or tag not contained in
public `main` fails before building. TestPyPI uploads use distinct environments
and trusted-publisher identities from production.

Before production, rehearse three clean environments:

1. the candidate `synthefy` wheel by itself;
2. candidate `synthefy 7` with released `synthefy-nori 0.16.0`;
3. both candidate wheels together.

The mixed released-heavy/candidate-light environment must either pass its
supported smoke test or fail immediately with explicit version guidance. These
candidate-wheel rehearsals do not depend on either package index. Because the
heavy wheel resolves `synthefy>=7,<8`, dispatch its TestPyPI publisher only after
the selected lightweight SDK version has been verified on PyPI.

## Publish to PyPI

After public CI and the rehearsals pass, create one GitHub Release at a time:

```bash
gh release create synthefy-v7.0.0 \\
  --repo Synthefy/synthefy-nori \\
  --target main \\
  --title "synthefy 7.0.0" \\
  --generate-notes
```

Approve the `synthefy-pypi` deployment when the workflow parks at its protected
environment. Verify in a clean environment:

```bash
uv venv --no-project /tmp/synthefy-release-check
uv pip install --python /tmp/synthefy-release-check/bin/python \\
  "synthefy==7.0.0"
uv pip check --python /tmp/synthefy-release-check/bin/python
/tmp/synthefy-release-check/bin/python -c \\
  'import synthefy; print(synthefy.__version__)'
```

Only after that succeeds, create the heavy release:

```bash
gh release create synthefy-nori-v0.17.0 \\
  --repo Synthefy/synthefy-nori \\
  --target main \\
  --title "synthefy-nori 0.17.0" \\
  --generate-notes
```

Approve `synthefy-nori-pypi`, then verify `synthefy-nori==0.17.0` resolves
`synthefy>=7,<8` and that local regression plus
`synthefy-nori[forecasting]` work in clean environments.

Each workflow builds only its own project root, runs strict metadata checks,
clean-installs the wheel, and runs `uv pip check` before upload. Every TestPyPI
and PyPI upload job is additionally gated by:

```text
github.repository == 'Synthefy/synthefy-nori'
```

That makes the same workflow files safe to validate in internal and staging.

## Trusted-publisher setup

Create four exact workflow/environment pairs in both GitHub and the matching
package index:

| Package | Index | Workflow | GitHub environment |
|---|---|---|---|
| `synthefy` | TestPyPI | `publish-synthefy.yml` | `synthefy-testpypi` |
| `synthefy` | PyPI | `publish-synthefy.yml` | `synthefy-pypi` |
| `synthefy-nori` | TestPyPI | `publish-synthefy-nori.yml` | `synthefy-nori-testpypi` |
| `synthefy-nori` | PyPI | `publish-synthefy-nori.yml` | `synthefy-nori-pypi` |

Require reviewers on both production environments. At cutover, revoke the
standalone `Synthefy/synthefy` `publish.yaml` identity and the public Nori
repository's old `publish.yml` identities. Renaming a workflow or archiving a
repository does not revoke an OIDC publisher.

Do not keep a local API-token publishing fallback. The public workflows and their
protected OIDC environments are the only release path.

## Post-publish verification

- Install base `synthefy`, `synthefy[aws]`, and
  `synthefy[forecasting]` from PyPI.
- Run hosted regression with the released client against Baseten dev.
- Install `synthefy-nori` and `synthefy-nori[forecasting]`; run local regression
  and forecasting.
- Run released-client SageMaker validation as assigned in the final serving
  handoff.
- Confirm prediction responses retain
  `prompt_tokens`, `completion_tokens`, and `total_tokens` and that the event is
  metered.
- Merge the staged customer docs only after the install commands work.
- Observe the released pair before archiving the standalone client repository.

A production smoke test requires explicit approval. Use ephemeral dev
credentials and delete them after testing.

## Failure and rollback

PyPI artifacts are immutable. If `synthefy 7.0.0` fails verification, do not
publish `synthefy-nori 0.17.0`; fix forward with `7.0.1`. If the heavy package
fails after publication, fix forward with `0.17.1`. Never reuse a version or
re-enable the old publisher. Existing `synthefy 6.3.0` and
`synthefy-nori 0.16.0` remain installable during validation.
