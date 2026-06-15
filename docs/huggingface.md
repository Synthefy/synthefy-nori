# Hugging Face

Download the default checkpoint:

```bash
synthefy-nori-download
```

Upload a checkpoint:

```bash
synthefy-nori-upload checkpoints/best_reg_r2.pt \
  --repo-id Synthefy/synthefy-nori
```

Python API:

```python
from synthefy_nori.hf import download_checkpoint, push_checkpoint
```
