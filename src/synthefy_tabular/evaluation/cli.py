"""Command-line evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthefy_tabular.api import config_path


def _parse_checkpoint(raw: str) -> tuple[str, str]:
    if ":" in raw:
        label, path = raw.split(":", 1)
        return label, path
    return Path(raw).stem, raw


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Synthefy Tabular checkpoints")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="Checkpoint path or label:path. Repeatable.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/eval")
    parser.add_argument("--tabarena-cls-dir", default="cache/tabarena_cls")
    parser.add_argument("--tabarena-reg-dir", default="cache/tabarena_reg")
    parser.add_argument("--custom-cls-dir", default=None)
    parser.add_argument("--custom-reg-dir", default=None)
    parser.add_argument("--cls-config", default=config_path("cls_default_noretrieval.json"))
    parser.add_argument("--reg-config", default=config_path("reg_allordinal_poly10_noretrieval.json"))
    parser.add_argument("--task-types", nargs="+", default=None,
                        choices=["classification", "regression"])
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--max-predict-samples", type=int, default=50000)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-dir", default="cache/eval_cache")
    args = parser.parse_args(argv)

    from synthefy_tabular.evaluation.datasets import DatasetRegistry
    from synthefy_tabular.evaluation.models import ModelRegistry
    from synthefy_tabular.evaluation.runner import EvalRunner
    from synthefy_tabular.hf import download_checkpoint

    datasets = DatasetRegistry(max_train_samples=args.max_train_samples)
    datasets.load_tabarena(cls_dir=args.tabarena_cls_dir, reg_dir=args.tabarena_reg_dir)
    if args.custom_cls_dir:
        datasets.load_custom_dir(args.custom_cls_dir, task_type="classification")
    if args.custom_reg_dir:
        datasets.load_custom_dir(args.custom_reg_dir, task_type="regression")

    models = ModelRegistry(device=args.device)
    checkpoints = args.checkpoint or [f"Synthefy:{download_checkpoint()}"]
    for raw in checkpoints:
        label, path = _parse_checkpoint(raw)
        models.add_limix(
            label,
            path,
            device=args.device,
            cls_config=args.cls_config,
            reg_config=args.reg_config,
        )

    runner = EvalRunner(
        models,
        datasets,
        output_dir=args.output_dir,
        warmup_runs=args.warmup,
        max_samples=args.max_predict_samples,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
    )
    runner.run(sources=args.sources, task_types=args.task_types)


if __name__ == "__main__":
    main()
