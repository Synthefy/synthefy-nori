# Hugging Face

Download the default checkpoint:

```bash
synthefy-tabular-download
```

Upload a checkpoint:

```bash
synthefy-tabular-upload checkpoints/best_reg_r2.pt \
  --repo-id Synthefy/synthefy-tabular
```

Python API:

```python
from synthefy_tabular.hf import download_checkpoint, push_checkpoint
```
