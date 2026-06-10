# Private distribution of `synthefy-tabular`

`synthefy-tabular` is a **private** package. It is **not** published to PyPI, and
it is **not** in GitHub Packages (GitHub Packages has no Python/PyPI registry —
see [Why not GitHub Packages / PyPI](#why-not-github-packages--pypi)).

Instead, the **private GitHub repo is the registry**. There are two install
paths, both private and both requiring no external infrastructure:

1. **git-install pinned to a version tag** (primary — zero infra).
2. **a prebuilt wheel attached to a private GitHub Release** (faster, no build
   step; produced by [`private-release.yml`](../.github/workflows/private-release.yml)).

A version is "published" simply by pushing a `vX.Y.Z` tag. See
[Publishing a private version](#publishing-a-private-version-maintainers).

---

## Install (consumers)

Prerequisites: **read access** to `Synthefy/synthefy-tabular`, and GitHub auth
configured locally (SSH key, or `gh auth login`, or a PAT — see
[Authentication](#authentication)).

### With `uv`

One-off into the current environment:

```bash
uv pip install \
  "synthefy-tabular @ git+https://github.com/Synthefy/synthefy-tabular.git@v0.2.0"
```

As a pinned dependency of a project — declare a normal dependency and point `uv`
at the private git source:

```toml
# consumer pyproject.toml
[project]
dependencies = ["synthefy-tabular"]

[tool.uv.sources]
synthefy-tabular = { git = "https://github.com/Synthefy/synthefy-tabular.git", tag = "v0.2.0" }
```

> `[tool.uv.sources]` is a uv-only, dev-time mechanism. It is **not** written
> into your package's published metadata, so it is safe to use even in a project
> you later upload to PyPI — but it only takes effect for people building with
> uv from your repo, not for people who `pip install` your package. For the
> public-SDK case see [Consuming from the Synthefy SDK](#consuming-from-the-synthefy-sdk).

### With `pip`

```bash
pip install \
  "synthefy-tabular @ git+https://github.com/Synthefy/synthefy-tabular.git@v0.2.0"
```

### Over SSH (instead of HTTPS)

```bash
pip install \
  "synthefy-tabular @ git+ssh://git@github.com/Synthefy/synthefy-tabular.git@v0.2.0"
```

### Prebuilt wheel from a private Release (no build step)

Each `vX.Y.Z` tag attaches a wheel + sdist to a private GitHub Release. Pull the
wheel directly — handy in CI or images where you don't want a git clone + build:

```bash
gh release download v0.2.0 -R Synthefy/synthefy-tabular -p '*.whl'
pip install synthefy_tabular-0.2.0-py3-none-any.whl
```

---

## Authentication

The repo is private, so every install path needs a credential that proves repo
read access.

### Local development

Easiest is one of:

- **SSH** — use the `git+ssh://` install form with your existing GitHub SSH key.
- **`gh`** — run `gh auth login` once; its git credential helper then satisfies
  the `git+https://` form transparently.

### CI / Docker

Use a token with **read-only `contents`** scope on `Synthefy/synthefy-tabular`
(a fine-grained PAT, or a GitHub App / deploy token). Expose it as a secret and
let git use it for the clone:

```yaml
# GitHub Actions example
- name: Install synthefy-tabular (private)
  env:
    GH_TOKEN: ${{ secrets.SYNTHEFY_TABULAR_READ_TOKEN }}
  run: |
    git config --global \
      url."https://x-access-token:${GH_TOKEN}@github.com/".insteadOf \
      "https://github.com/"
    pip install \
      "synthefy-tabular @ git+https://github.com/Synthefy/synthefy-tabular.git@v0.2.0"
```

Or, to grab the prebuilt wheel instead of building from source:

```yaml
- name: Install synthefy-tabular wheel (private Release)
  env:
    GH_TOKEN: ${{ secrets.SYNTHEFY_TABULAR_READ_TOKEN }}
  run: |
    gh release download v0.2.0 -R Synthefy/synthefy-tabular -p '*.whl'
    pip install synthefy_tabular-*.whl
```

Prefer **fine-grained PATs** (or short-lived GitHub App tokens) scoped to this
single repo with read-only contents, so the blast radius stays minimal. Never
bake a token into the published metadata of any package.

---

## Publishing a private version (maintainers)

"Publishing privately" = cutting a version tag. The
[`private-release.yml`](../.github/workflows/private-release.yml) workflow does
the rest.

1. Bump the version in **both** places, kept in sync:
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `src/synthefy_tabular/__init__.py` → `__version__ = "X.Y.Z"`
2. Commit and merge to `main`; wait for `ci` to go green.
3. Tag and push:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

4. On the tag push, `private-release.yml`:
   - builds the sdist + wheel (`uv build`),
   - runs `twine check --strict`,
   - verifies the tag matches the built package version (mirrors `publish.yml`),
   - creates a **private GitHub Release** `vX.Y.Z` with both artifacts attached.

Consumers can then pin `@vX.Y.Z` (git-install) or download the wheel from the
Release. This path is independent of the PyPI `publish.yml` workflow, which is
left intact and untouched.

---

## Consuming from the Synthefy SDK

The SDK (public, on PyPI) supports **hosted inference by default** and **optional
local inference** for authorized users.

### Hosted (default) — no dependency on this package

The hosted path runs the model on Baseten; the SDK is a thin HTTP client and
does **not** import or depend on `synthefy-tabular` at all. The private model
code and weights never ship. The request/response contract is the Baseten
endpoint, specified in [`baseten/openapi.yaml`](../baseten/openapi.yaml)
(`X_train` / `y_train` / `X_test` → `predictions`). Base install pulls only
public deps (e.g. `httpx`).

### Local (`[local]` extra) — private install, out of band

A package uploaded to **public PyPI cannot carry a private dependency**: PyPI
rejects direct-reference (URL/git) requirements in metadata, and a bare
`synthefy-tabular` name would be unresolvable for anyone without repo access. So
the `[local]` extra must **not** list `synthefy-tabular` as a git URL.

The working pattern:

1. The SDK's `local` extra installs only **public** heavy deps it needs directly
   (often none — `synthefy-tabular` brings its own torch/numpy/etc.):

   ```toml
   # SDK pyproject.toml
   [project.optional-dependencies]
   local = []  # heavy public deps only, if any; NOT synthefy-tabular
   ```

2. Authorized users install the private package **out of band** first (any
   install path above), then the SDK with the extra:

   ```bash
   pip install \
     "synthefy-tabular @ git+https://github.com/Synthefy/synthefy-tabular.git@v0.2.0"
   pip install "synthefy-sdk[local]"
   ```

3. The SDK imports it **lazily**, with a clear error when it's absent:

   ```python
   def _load_local_regressor():
       try:
           from synthefy_tabular import SynthefyTabularRegressor
       except ImportError as exc:  # pragma: no cover
           raise ImportError(
               "Local inference requires the private 'synthefy-tabular' package. "
               "Install it with repo access:\n"
               "  pip install 'synthefy-tabular @ "
               "git+https://github.com/Synthefy/synthefy-tabular.git@v0.2.0'\n"
               "Or use the hosted client (no install needed)."
           ) from exc
       return SynthefyTabularRegressor()
   ```

This keeps the public SDK installable by anyone (hosted path) while gating local
inference behind a private install only authorized users can complete.

---

## Why not GitHub Packages / PyPI

- **GitHub Packages** hosts npm, Maven, Gradle, NuGet, RubyGems, and container
  images — but **not Python/PyPI**. Native Python support is not on the roadmap
  ([github/roadmap#94](https://github.com/github/roadmap/issues/94)). The
  GitHub-native way to host Python artifacts is therefore **GitHub Releases**,
  which is what `private-release.yml` uses.
- **Public PyPI** can't keep this package private: a published package may not
  declare direct-URL/git dependencies, and a private dependency would be
  unresolvable for external installers. So the SDK uses the hosted-API default
  and an out-of-band private install for the `[local]` extra (above).

## Upgrade path: a real private index

If git-install + Releases becomes limiting (e.g. you want `[local]` to resolve
`synthefy-tabular` by bare name, or you need a true resolver/cache), stand up a
private PyPI-compatible index — **AWS CodeArtifact**, **GCP Artifact Registry**,
**Cloudsmith**, or **Gemfury**. Then publish wheels there and consumers add it
via `--extra-index-url https://<token>@<host>/simple/`. That restores normal
`pip install synthefy-tabular` resolution at the cost of running the index.
