# Training

`scripts/train.sh` is the default full training recipe. It calls
`synthefy_tabular.training.cli` through `torchrun` and writes checkpoints under
`checkpoints/`.

Use environment variables to shorten or resize a run:

```bash
TOTAL_STEPS=2 NPROC_PER_NODE=1 WANDB_MODE=disabled bash scripts/train.sh
```

`scripts/continue_training.sh` resumes from `stage1/best_reg_r2.pt` and expands
the row and feature ranges for large-table continuation.
