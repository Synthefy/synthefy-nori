# Evaluation

Run:

```bash
synthefy-tabular-eval --checkpoint "Synthefy:checkpoints/best_reg_r2.pt"
```

The CLI loads local TabArena-style CSV caches by default:

```text
cache/tabarena_cls/
cache/tabarena_reg/
```

Use `--custom-cls-dir` or `--custom-reg-dir` for local custom datasets.
