# Nori 1.5 weights for offline use

This folder is laid out like a Hugging Face cache, so `NoriRegressor(model="nori-1-5")` finds the weights here without a network connection. From the repository root:

```bash
export HF_HOME="$PWD/hf-home"
export HF_HUB_OFFLINE=1
```

```python
from synthefy_nori import NoriRegressor
model = NoriRegressor(model="nori-1-5")
```

Needs `synthefy-nori>=0.24.0`.
