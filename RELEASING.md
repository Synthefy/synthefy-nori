# Releasing `synthefy-tabular` to PyPI

This package can be published either through the automated GitHub Actions
workflow (recommended) or manually from a developer machine.

## TL;DR — ship version `X.Y.Z`

1. Set the same `X.Y.Z` in three places: `pyproject.toml` (`version =`), `src/synthefy_tabular/__init__.py` (`__version__ =`), and the git tag you're about to cut.
2. Commit and push to `main`, then wait for the `ci` workflow to go green on that commit.
3. (Optional) Rehearse on TestPyPI: `gh workflow run publish.yml --ref main -f target=testpypi`, then `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ synthefy-tabular==X.Y.Z` in a clean venv.
4. Cut the release: `gh release create vX.Y.Z --target main --title "vX.Y.Z" --generate-notes`.
5. This triggers `publish.yml`, which builds the sdist + wheel, runs `twine check --strict`, and verifies the wheel version matches the `vX.Y.Z` tag.
6. The upload step parks at the `pypi` environment's reviewer gate — go to Actions → the running `publish` run → **Review deployments** → check `pypi` → **Approve**.
7. After approval, `pypa/gh-action-pypi-publish` uploads to PyPI over OIDC (no tokens) in ~30 seconds.
8. Sanity-check: `pip install synthefy-tabular==X.Y.Z` in a fresh venv and run `python -c "import synthefy_tabular; print(synthefy_tabular.__version__)"`.
9. If the build fails on version mismatch, fix the version files locally, push, then delete + recreate the release with `gh release delete vX.Y.Z --cleanup-tag --yes && gh release create vX.Y.Z ...`.
10. PyPI never allows re-uploading the same version — if an upload itself partially fails, bump to `X.Y.Z+1` and start over rather than reusing the tag.

## One-time setup

### 1. Reserve the name on PyPI

Make sure `synthefy-tabular` is available (or already owned by Synthefy):
<https://pypi.org/project/synthefy-tabular/>. If taken by someone else, pick a
different `name =` in `pyproject.toml` before going further.

### 2. Configure PyPI trusted publishing (no API tokens needed)

Trusted publishing lets GitHub Actions upload to PyPI over OIDC, with no
long-lived secrets stored in the repo.

Do this **twice** — once for TestPyPI, once for PyPI — once the project page
exists. For brand-new project names, use the "pending publisher" form on each
site to authorize the first upload.

For each of <https://pypi.org/manage/account/publishing/> and
<https://test.pypi.org/manage/account/publishing/>, add a publisher with:

- **PyPI Project Name:** `synthefy-tabular`
- **Owner:** `Synthefy`
- **Repository name:** `synthefy-tabular`
- **Workflow name:** `publish.yml`
- **Environment name:** `pypi` (for PyPI) or `testpypi` (for TestPyPI)

### 3. Create matching GitHub environments

In the GitHub repo settings → Environments, create two environments named
`pypi` and `testpypi`. Optionally require a manual approval reviewer on `pypi`
so a real human has to click through every production release.

## Automated release (recommended)

1. Bump the version in **both** places (keep them in sync):
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `src/synthefy_tabular/__init__.py` → `__version__ = "X.Y.Z"`
2. Commit and merge to `main`.
3. Optional rehearsal on TestPyPI:
   - GitHub Actions → **publish** → "Run workflow" → target `testpypi`.
   - Install from TestPyPI to sanity-check:
     ```bash
     uv pip install --index-url https://test.pypi.org/simple/ \
         --extra-index-url https://pypi.org/simple/ synthefy-tabular==X.Y.Z
     ```
4. Cut a GitHub Release with tag `vX.Y.Z`. The `publish` workflow builds,
   verifies the tag matches the wheel version, and uploads to PyPI.

## Manual release (fallback)

```bash
# Clean previous builds
rm -rf dist/

# Build sdist + wheel
uv build

# Validate the README renders on PyPI and metadata is sane
uvx twine check --strict dist/*

# Upload to TestPyPI first
uvx twine upload --repository testpypi dist/*

# Then the real thing
uvx twine upload dist/*
```

Credentials for manual uploads come from a PyPI API token. Put it in
`~/.pypirc` or export `TWINE_USERNAME=__token__` and
`TWINE_PASSWORD=pypi-…`. Scope the token to the `synthefy-tabular` project
after the first upload so the blast radius is limited.

## Version policy

- Pre-1.0, breaking changes are allowed on minor bumps (`0.X.0`).
- `__version__` in code and `version` in `pyproject.toml` MUST match the git
  tag (`vX.Y.Z`). The publish workflow enforces this.
- Once a version is uploaded to PyPI it cannot be re-uploaded — bump the
  version and try again rather than deleting.
