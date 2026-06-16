# Protecting the default branch

This repo ships two complementary pieces of scaffolding so that changes can only
land on the default branch (`main`) after an approving review from an authorized
maintainer:

| File | Role |
| --- | --- |
| [`CODEOWNERS`](CODEOWNERS) | Auto-requests review from the authorized maintainers and satisfies GitHub's "Require review from Code Owners" rule. |
| [`workflows/require-approval.yml`](workflows/require-approval.yml) | A status check that fails unless at least one authorized maintainer has approved the PR. |

**Authorized reviewers** — a PR needs an approving review from at least one of:

- `sagarwal-atg`
- `d31003`
- `yogabbagabb`
- `raimishah`
- `adityanarayanan03`

## One-time setup (requires repo admin)

The files above describe the rule, but only a branch protection rule (or
ruleset) can actually *block* the merge button. A repo admin must enable this
once, under **Settings → Branches → Add branch ruleset** (or **Add rule**)
targeting the default branch:

1. **Require a pull request before merging**, with **Required approvals** set to
   at least `1`.
2. **Require review from Code Owners** — ties the required approval to the
   `CODEOWNERS` list above.
3. **Require status checks to pass before merging** — add the
   **`require-approval`** check (it shows up in the list once the workflow has
   run at least once).
4. *(Recommended)* **Dismiss stale pull request approvals when new commits are
   pushed**, and **Do not allow bypassing the above settings** (include
   administrators), so the rule cannot be sidestepped.

> `require-approval` and `CODEOWNERS` are belt-and-suspenders: the workflow
> encodes the exact allowlist in version control, while Code Owners provides
> reviewer auto-assignment. Enable either or both. Code owners must have
> **write** access to the repo, or GitHub ignores their `CODEOWNERS` entries.

To change the authorized reviewer set, edit **both** `CODEOWNERS` and the
`AUTHORIZED_REVIEWERS` array in `workflows/require-approval.yml`.
