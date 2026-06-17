# Inference

Use the public wrappers:

```python
from synthefy_nori import NoriRegressor

reg = NoriRegressor(model_path="checkpoints/best_reg_r2.pt")
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

If `model_path` is omitted, the default checkpoint is resolved from Hugging
Face through `synthefy_nori.hf.download_checkpoint()`.

## Output types

`predict` mirrors the `TabPFNRegressor.predict` contract via `output_type`:

| `output_type` | Returns | Shape |
| --- | --- | --- |
| `"mean"` (default) | distribution mean | `(n_samples,)` |
| `"median"` | distribution median (τ=0.5) | `(n_samples,)` |
| `"mode"` | distribution mode | `(n_samples,)` |
| `"quantiles"` | quantiles at `quantiles=[...]` levels | `(n_levels, n_samples)` |
| `"full"` | dict `{"quantiles", "taus", "mean"}` | quantiles `(n_samples, K)` |

```python
# Prediction intervals from quantiles
lo, mid, hi = reg.predict(X_test, output_type="quantiles", quantiles=[0.05, 0.5, 0.95])

# Full predictive distribution (e.g. for CRPS / calibration)
dist = reg.predict(X_test, output_type="full")
Q, taus = dist["quantiles"], dist["taus"]   # Q: (n_samples, 999), ascending per row
```

The default checkpoint exposes a 999-quantile pinball head; quantiles come back
in original-`y` units, sorted per row. `"quantiles"`/`"full"` require the
pinball checkpoint — a `bar_distribution` checkpoint raises `NotImplementedError`.
