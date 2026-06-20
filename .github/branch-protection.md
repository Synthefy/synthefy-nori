# Protecting the default branch

This repo uses [`CODEOWNERS`](CODEOWNERS) so that changes can only land on the
default branch (`main`) after an approving review from an authorized maintainer.
It auto-requests review from the authorized maintainers and satisfies GitHub's
"Require review from Code Owners" rule.

**Authorized reviewers** — a PR needs an approving review from at least one of:

- `sagarwal-atg`
- `d31003`
- `yogabbagabb`
- `raimishah`
- `adityanarayanan03`

## One-time setup (requires repo admin)

`CODEOWNERS` describes the rule, but only a branch protection rule (or ruleset)
can actually *block* the merge button. A repo admin must enable this once, under
**Settings → Branches → Add branch ruleset** (or **Add rule**) targeting the
default branch:

1. **Require a pull request before merging**, with **Required approvals** set to
   at least `1`.
2. **Require review from Code Owners** — ties the required approval to the
   `CODEOWNERS` list above.
3. *(Recommended)* **Dismiss stale pull request approvals when new commits are
   pushed**, and **Do not allow bypassing the above settings** (include
   administrators), so the rule cannot be sidestepped by accounts outside any
   explicitly configured bypass list.
4. **Require status checks to pass before merging**, and add these two jobs
   (from [`workflows/ci.yml`](workflows/ci.yml)) to the required checks. This is
   what actually blocks the merge button on a broken model:
   - **`CI test inference`** — runs a real inference forward pass (downloads the
     public `Synthefy/Nori` checkpoint).
   - **`CI test train step`** — runs one real training step from scratch on CPU.

   *(Add the fast `test` job too if you want lint / unit tests / build to gate
   merges as well.)*

   **Optional GPU gate** — [`workflows/gpu-ci.yml`](workflows/gpu-ci.yml) adds a
   **`GPU test inference + train step`** job that runs the same two checks on a
   CUDA device, catching CUDA-only / dtype / autocast regressions the CPU gate
   cannot. It needs a runner labelled `gpu` (a GitHub-hosted GPU larger runner,
   provisioned by an org admin under **Settings → Actions → Runners → New
   runner**). **Do not mark this job required until that runner exists** — with
   no runner the check sits queued forever and would block every merge. Note the
   hosted GPU runner is an NVIDIA T4 (sm75), so it cannot run FlashAttention-2
   (needs Ampere+); the flash-attn step in that workflow is left disabled.

> A workflow only *reports* status — only a required status check *blocks* the
> merge. The check name GitHub matches is the job's display name (`CI test
> inference` / `CI test train step`); each appears in the list once the workflow
> has run at least once on a PR.

> Code owners must have **write** access to the repo, or GitHub ignores their
> `CODEOWNERS` entries.

To change the authorized reviewer set, edit `CODEOWNERS`.
