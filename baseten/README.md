# Deploying Synthefy Tabular to Baseten

This is a [Truss](https://docs.baseten.co/development/model/custom-model-code)
that serves the Synthefy Tabular in-context learning model on Baseten.

## Layout

```
baseten/
├── config.yaml              # resources (T4 GPU), requirements, HF secret
├── model/
│   └── model.py             # Model class: load() warms the checkpoint, predict() serves regression
└── packages/
    └── synthefy_tabular/    # the bundled package (configs included)
```

The `synthefy_tabular` package is vendored under `packages/` (auto-added to the
container's `PYTHONPATH` by Truss) because it is not yet published to PyPI. The
vendored copy is **gitignored** to avoid a second, drifting copy of `src/`, so
you must generate it before the first push and re-sync it after any source change:

```bash
rm -rf packages/synthefy_tabular && cp -R ../src/synthefy_tabular packages/synthefy_tabular
```

## One-time setup

1. Install the CLI and log in:

   ```bash
   uv tool install truss        # or: pip install truss
   truss login                  # paste a Baseten API key from https://app.baseten.co/settings/api_keys
   ```

2. The default checkpoint (`Synthefy/synthefy-tabular`) is **public on Hugging
   Face**, so no token is required to download it. If you want authenticated
   downloads anyway (e.g. to avoid anonymous rate limits), add a Baseten secret
   named `hf_access_token` under **Settings → Secrets** set to a HF read token.
   (`config.yaml` already declares this secret.)

## Deploy

From this directory:

```bash
cd baseten
truss push --watch     # build + deploy, stream logs, hot-reload on file changes
```

The first request to a fresh deployment triggers `load()`, which downloads the
checkpoint and warms up the model — so cold starts take a bit longer.

## Invoke

The endpoint takes the in-context training rows and the query rows in one call.
The deployment is **regression-only**: the `task` field is optional and, if
given, must be `"regression"` (or `"reg"`).

```bash
curl -X POST https://model-{MODEL_ID}.api.baseten.co/development/predict \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -d '{
    "task": "regression",
    "X_train": [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]],
    "y_train": [0.1, 0.9, 0.5, 0.3],
    "X_test":  [[0.3, 0.7], [0.8, 0.2]]
  }'
# -> {"task": "regression", "predictions": [0.34, 0.81]}
```

`X_train`/`X_test` are `n_rows × n_features`; `y_train` aligns with `X_train`.

## Notes

- **Hardware**: T4 GPU (`resources` in `config.yaml`). Bump to `A10G` for larger
  tables / higher throughput, or set `use_gpu: false` + `accelerator: null` for a
  CPU-only deployment.
- **Concurrency**: inference is serialized with a lock in `model.py` because the
  predictor's preprocessing keeps per-call RNG state. Scale out with replicas
  rather than in-process concurrency.
