# Releasing `synthefy` and `synthefy-nori`

This repository builds two independently versioned Python distributions.
Upstream development and staging repositories build and test candidates but
cannot upload either distribution:

```text
upstream  ->  staging  ->  public
```

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
2. Publish and verify `synthefy-nori 0.17.3`.
3. Merge customer documentation only after both documented install commands
   resolve from PyPI.

Do not upload new Hugging Face weights, change gateway slugs, pricing, billing,
limits, or entitlements for this package-source cutover.

## Before promotion

- Complete and validate the candidate upstream.
- Run the offline package, namespace, candidate/released-client, OpenAPI, and
  packaged-container gates.
- Complete the explicitly assigned final Baseten-dev and SageMaker validation;
  record any human handoff rather than treating a skipped live test as passed.
- Confirm the existing gateway binding, billing allowlist, and unchanged
  `usage` response remain valid.
- Keep the customer docs PR staged, not merged.

Then promote the identical public-ready patch to staging, run every gate there,
and promote to public last. Wait for rebase-sync/drop-check before tagging.
Defects found in staging or public are fixed upstream and re-promoted.

## Prepare a version

A version is written down in more places than its own project files. Three of them
are shared by both distributions and none is next to the code it describes, so a
bump that misses one fails CI rather than the release:

| Location | Applies to | Enforced by |
|---|---|---|
| `pyproject.toml` / `libs/synthefy/pyproject.toml` | its own project | publish workflow |
| `src/synthefy_nori/__init__.py` / `libs/synthefy/src/synthefy/__init__.py` | its own project | publish workflow |
| `uv.lock` | **both** — one entry per workspace member | `uv sync --locked` in `ci.yml` |
| `.github/workflows/ci.yml` | **both** — an inline `__version__` assertion | itself |
| `tests/test_synthefy_workspace.py` | **both** — asserts each `project.version` | `pytest` |
| the git tag | its own project | publish workflow |

`uv.lock` is the one most often missed: `uv sync` rewrites it as a side effect of
any local run, so it looks incidental, but `ci.yml` runs `uv sync --locked` and
fails on a stale lock. Commit it with the bump.

For `synthefy`:

1. Set the exact version in `libs/synthefy/pyproject.toml` and
   `libs/synthefy/src/synthefy/__init__.py`.
2. Add the matching section to `libs/synthefy/CHANGELOG.md`.
3. Update the three shared locations above.
4. Use tag `synthefy-vX.Y.Z`.

For `synthefy-nori`:

1. Set the exact version in `pyproject.toml` and
   `src/synthefy_nori/__init__.py`.
2. Update the three shared locations above.
3. Use tag `synthefy-nori-vX.Y.Z`; GitHub Release notes are its changelog.

Then confirm nothing was missed, rather than trusting the table:

```bash
grep -rn "<previous version>" --include="*.toml" --include="*.py" \
  --include="*.yml" --include="*.lock" . | grep -v "\.venv\|CHANGELOG"
```

Historical `CHANGELOG.md` entries and `RELEASING.md` examples legitimately name old
versions; everything else the grep finds is a miss.

The workflows verify tag, project metadata, module `__version__`, and built wheel
metadata agree. They also require the peeled tag commit to be an ancestor of
public `main`, so a tag from an unpromoted branch cannot publish.

## Expect to repair the sync chain after publishing

`public` defines the version, but `synthefy-nori-internal` carries its own copies of
every shared location above. So a release reliably conflicts on the `rebase-sync`
internal hop, on `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`, and
`tests/test_synthefy_workspace.py`.

That is expected, not a sign the release went wrong. The conflict aborts the hop and
alerts; nothing is force-pushed broken, but internal `main` stays stuck and every
later push to it re-fails until someone repairs it by hand. Resolve by taking
`public`'s new version and keeping internal-only lines around it (its
`requires-python`, its own comments), then force-push internal — a PR cannot land a
rebase that rewrites `main`, because `main` requires linear history. Archive-tag the
old head first so the force-push is reversible.

Re-run `rebase-sync` and `drop-check` afterwards; an internal push does not trigger
them. `drop-check` will then flag every PR merged before the rewrite, because
patch-ids do not survive one — verify their content really is on `main`, then bump
`NOT_BEFORE` in `scripts/sync/check_dropped_prs.sh` past the rewrite.

## Rehearse the lightweight SDK on TestPyPI

The lightweight `synthefy` SDK retains a TestPyPI rehearsal path. Manual
dispatch is deliberately tag-bound: supply distribution, version, target, and
tag explicitly, and select that same existing tag as the workflow ref:

```bash
gh workflow run publish-synthefy.yml \\
  --repo Synthefy/synthefy-nori \\
  --ref synthefy-v7.0.0 \\
  -f distribution=synthefy \\
  -f version=7.0.0 \\
  -f target=testpypi \\
  -f tag=synthefy-v7.0.0
```

A branch ref, mismatched tag/version, wrong distribution, or tag not contained in
public `main` fails before building. The SDK TestPyPI upload uses a distinct
environment and trusted-publisher identity from production. `synthefy-nori`
publishes only to production PyPI; its workflow still performs the same strict
artifact metadata and clean-install validation before the protected upload.

Before production, rehearse three clean environments:

1. the candidate `synthefy` wheel by itself;
2. candidate `synthefy 7` with released `synthefy-nori 0.16.0`;
3. both candidate wheels together.

The mixed released-heavy/candidate-light environment must either pass its
supported smoke test or fail immediately with explicit version guidance. These
candidate-wheel rehearsals do not depend on either package index. The heavy
wheel resolves `synthefy>=7,<8`, so publish it only after the selected
lightweight SDK version has been verified on PyPI.

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
environment.

Nobody signs in to a package index to do this: upload is OIDC trusted publishing,
so there are no index credentials. The only gate is the GitHub environment, and its
reviewer list is a small set that may **not** include whoever cut the release. Check
before you plan the timing, so the release does not sit parked waiting on a person
who was never asked:

```bash
gh api repos/Synthefy/synthefy-nori/actions/runs/<run-id>/pending_deployments \
  -q '.[] | {env: .environment.name, canApprove: .current_user_can_approve,
             reviewers: [.reviewers[]?.reviewer.login]}'
```

Verify in a clean environment:

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
gh release create synthefy-nori-v0.17.3 \\
  --repo Synthefy/synthefy-nori \\
  --target main \\
  --title "synthefy-nori 0.17.3" \\
  --generate-notes
```

Approve `synthefy-nori-pypi`, then verify `synthefy-nori==0.17.3` resolves
`synthefy>=7,<8` and that local regression plus
`synthefy-nori[forecasting]` work in clean environments.

Each workflow builds only its own project root, runs strict metadata checks,
clean-installs the wheel, and runs `uv pip check` before upload. Every package
index upload job is additionally gated by:

```text
github.repository == 'Synthefy/synthefy-nori'
```

That makes the same workflow files safe to validate in internal and staging.

## Trusted-publisher setup

Create three exact workflow/environment pairs in both GitHub and the matching
package index:

| Package | Index | Workflow | GitHub environment |
|---|---|---|---|
| `synthefy` | TestPyPI | `publish-synthefy.yml` | `synthefy-testpypi` |
| `synthefy` | PyPI | `publish-synthefy.yml` | `synthefy-pypi` |
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
publish `synthefy-nori 0.17.3`; fix forward with `7.0.1`. If the heavy package
fails after publication, fix forward with `0.17.4`. Never reuse a version or
re-enable the old publisher. Existing `synthefy 6.3.0` and
`synthefy-nori 0.16.0` remain installable during validation.
