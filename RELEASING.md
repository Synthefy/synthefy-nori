# Releasing `synthefy-tabular` to PyPI

This package can be published either through the automated GitHub Actions
workflow (recommended) or manually from a developer machine.

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
