# Contributing

Use focused pull requests and include validation for behavior changes.

Before opening a PR, run:

```bash
python -m pip install -e ".[dev]"
pytest tests
ruff check src scripts tests
python -m build
```

Do not commit checkpoints, generated datasets, caches, benchmark results, or
private research artifacts.
