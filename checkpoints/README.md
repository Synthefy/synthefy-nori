# Nori 1.5 checkpoint, v9.6.4 tier 4

`nori-1-5-v9.6.4-tier4.pt` is the Nori 1.5 weights retrained to recipe v9.6.4 tier 4 (EMA, optimizer step 297,000), 20,757,832 parameters, with the 4-member `model_v3_inference.json` stored in the file. It is stored with Git LFS.

- sha256: `8cd053a411fc7e7b2ad348e1d37e09d865b422be05d41ef3c41167d7839d6ab6`
- Needs `synthefy-nori>=0.24.0`. Earlier releases cannot load it.
- The same file is on HuggingFace (`Synthefy/Nori-1-5`, `nori.pt`) and attached to the GitHub Release `synthefy-nori-v0.24.0`.

```bash
git lfs install
git clone --branch customer-nori-1-5-v964-tier4-lfs https://github.com/Synthefy/synthefy-nori.git
```

```python
from synthefy_nori import NoriRegressor
model = NoriRegressor(model_path="synthefy-nori/checkpoints/nori-1-5-v9.6.4-tier4.pt")
```

This branch exists to hand the file to a customer through git. It is not meant to merge into `main`.
