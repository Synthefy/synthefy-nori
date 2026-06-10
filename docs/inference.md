# Inference

Use the public wrappers:

```python
from synthefy_tabular import SynthefyTabularRegressor

reg = SynthefyTabularRegressor(model_path="checkpoints/best_reg_r2.pt")
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

If `model_path` is omitted, the default checkpoint is resolved from Hugging
Face through `synthefy_tabular.hf.download_checkpoint()`.
