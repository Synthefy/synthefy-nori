## Summary

<!-- Explain the problem and the approach taken. Keep the change focused. -->

## Related issue

<!-- Link an issue or write "None". Use "Closes #123" when appropriate. -->

## Change classification

Select exactly one primary class:

- [ ] Training or model-internal
- [ ] Local library only
- [ ] Backend-neutral workflow
- [ ] Hosted regression or wire affecting
- [ ] Packaging or release only
- [ ] CI, governance, or documentation only

## Execution contract

- Supported modes (local, remote, sagemaker, or N/A):
- Unsupported modes and their fail-closed behavior:
- NoriPredictRequest / NoriPredictResponse or raw wire impact:
- Hosted artifact or capability impact:

## Package impact

- Distribution(s) changed:
- Base-install size or dependency impact:
- Does pip install synthefy remain Torch-free?
- Python-version impact:

## Verification

- Offline tests and package checks:
- Mocked/local mode coverage:
- Documentation or runnable examples updated:
- Protected live or pre-release validation still required:

## Source and release ownership

- Authoritative source path(s):
- Internal-first / staging / public promotion impact:
- Related serving, client, docs, or release PRs:

<!--
Backend-neutral workflows should cover every supported execution mode. If a
mode is intentionally unsupported, name the capability boundary and the tested,
actionable error. Billable live tests remain protected manual or pre-release
checks; they do not belong in ordinary pull-request CI.
-->
## Validation

<!-- List the exact commands or manual checks you ran and their results. If a
check was not relevant or could not be run, explain why. See CONTRIBUTING.md. -->

- Tests:
- Lint:
- Build:
- Other:

## Checklist

- [ ] The change is focused and contains no unrelated modifications.
- [ ] Tests and documentation were added or updated where needed.
- [ ] Relevant checks from `CONTRIBUTING.md` pass, or exceptions are explained above.
- [ ] No checkpoints, datasets, caches, credentials, or private artifacts are included.
- [ ] Existing license, notice, and third-party attribution requirements are preserved.

<!-- External contributors: the CLA bot will explain how to sign CLA.md after
the pull request is opened. -->
