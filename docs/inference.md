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
