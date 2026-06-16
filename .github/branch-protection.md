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

> Code owners must have **write** access to the repo, or GitHub ignores their
> `CODEOWNERS` entries.

To change the authorized reviewer set, edit `CODEOWNERS`.
