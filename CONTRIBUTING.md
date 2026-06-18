# Contributing

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
