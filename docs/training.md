# Training

All training runs on synthetic data and **trains to completion**: there is no
real-data validation in the loop, so no benchmark data is needed to train and no
eval signal influences checkpoint selection. Runs write periodic
`checkpoint_step_*.pt` files plus a final checkpoint, and each curriculum tier
seeds from the previous tier's final checkpoint.

`scripts/train.sh` calls `synthefy_tabular.training.cli` through `torchrun` and
writes checkpoints under `checkpoints/`. Use environment variables to shorten or
resize a run:

```bash
TOTAL_STEPS=2 NPROC_PER_NODE=1 WANDB_MODE=disabled bash scripts/train.sh
```

## Curriculum

`scripts/train.sh` trains tier 1 from scratch. `scripts/continue_training.sh`
runs tiers 2 to 5, each seeding from the previous tier's final checkpoint:

| Tier | Table shapes (N x F) | Focus |
|---|---|---|
| 1 | N ≤ 8K, F ≤ 250 | from scratch, broad shape mix |
| 2 | N ≤ 4K, F ≤ 384 | larger tables |
| 3 | N ≤ 8K, F ≤ 768 | largest tables |
| 4 | N ≤ 56K, F ≤ 96 | large-N / long-context specialist |
| 5 | N ≤ 33K, F ≤ 1280 | both-large corner (N and F coupled by a cell budget) |

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh                  # tier 1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/continue_training.sh      # tiers 2 to 5
```

Select a subset with `START_TIER` / `END_TIER` (e.g. `END_TIER=3`), or point at a
specific run with `RUN_ROOT=checkpoints/<run>`. Each tier's knobs are overridable
via `TIER{2..5}_*` environment variables (steps, LR, batch size, shape ranges);
see the script header.

Tiers 4 and 5 reach N = 56K rows, where dense O(N²) sample attention forces
`batch=1` with large gradient accumulation. They can OOM or hang on smaller GPUs;
smoke-probe them first (the script header shows a short-run probe).
