# Hugging Face

Download a checkpoint by name (`nori-6m`, `nori-1-5`, `nori-30m`, or `nori-100m`):

```bash
synthefy-nori-download --model nori-30m
```

With no `--model` (or `--repo-id`), it downloads Nori 1.5 (`nori-1-5`) from
`Synthefy/Nori-1-5`. The command prints the local path of the cached file.

Upload a checkpoint:

```bash
synthefy-nori-upload checkpoints/best_reg_r2.pt \
  --repo-id Synthefy/Nori
```

Python API:

```python
from synthefy_nori.hf import download_checkpoint, push_checkpoint
```
