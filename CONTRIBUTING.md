# Contributing

By participating in the Nori community, you agree to follow our
[Code of Conduct](CODE_OF_CONDUCT.md).

Use focused pull requests and include validation for behavior changes.

Before opening a PR, run:

```bash
uv sync --extra dev
uv run pytest tests
uv run ruff check src scripts tests
uv build
```

Do not commit checkpoints, generated datasets, caches, benchmark results, or
private research artifacts.

External contributors must sign the [Contributor License Agreement](CLA.md).
The CLA bot provides signing instructions when a pull request is opened.
