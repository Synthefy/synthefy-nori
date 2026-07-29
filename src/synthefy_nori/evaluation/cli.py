"""Command-line evaluation entry point (regression only)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..configs import DEFAULT_CONFIG_NAME, config_path


def _parse_checkpoint(raw: str) -> tuple[str, str]:
    if ":" in raw:
        label, path = raw.split(":", 1)
        return label, path
    return Path(raw).stem, raw


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Nori checkpoints")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="Checkpoint path or label:path. Repeatable.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/eval")
    parser.add_argument("--tabarena-reg-dir", default="cache/tabarena_reg")
    parser.add_argument("--talent-reg-dir", default="cache/talent_reg")
    parser.add_argument("--openml-reg", action="store_true",
                        help="Include the curated OpenML regression suite (downloads from OpenML)")
    parser.add_argument("--download-benchmarks", action="store_true",
                        help="Download the TabArena and TALENT regression CSV caches from OpenML first")
    parser.add_argument("--custom-reg-dir", default=None)
    parser.add_argument("--reg-config", default=config_path(DEFAULT_CONFIG_NAME))
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--max-predict-samples", type=int, default=50000)
    parser.add_argument("--max-elements-budget", type=int, default=8_000_000,
                        help="SYNTHEFY_MAX_ELEMENTS_BUDGET for inference chunking/subsampling. "
                             "The 8M default targets large GPUs (>=80GB) and stays under CUDA "
                             "kernel grid limits on the largest test sets; lower it (e.g. "
                             "2000000) on smaller GPUs. An explicit env var wins.")
    parser.add_argument("--gpu-mem-gb", type=float, default=None,
                        help="Enable the memory-model train-row cap for smaller GPUs (e.g. 24). "
                             "Default: uncapped — train rows bounded only by --max-train-samples.")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-dir", default="cache/eval_cache")
    args = parser.parse_args(argv)

    import os
    os.environ.setdefault("SYNTHEFY_MAX_ELEMENTS_BUDGET", str(args.max_elements_budget))

    from synthefy_nori.evaluation.datasets import DatasetRegistry
    from synthefy_nori.evaluation.models import ModelRegistry
    from synthefy_nori.evaluation.runner import EvalRunner
    from synthefy_nori.hf import download_checkpoint

    datasets = DatasetRegistry(max_train_samples=args.max_train_samples)
    if args.download_benchmarks:
        datasets.download_tabarena(reg_dir=args.tabarena_reg_dir)
        datasets.download_talent(reg_dir=args.talent_reg_dir)
    datasets.load_tabarena(reg_dir=args.tabarena_reg_dir)
    datasets.load_talent(reg_dir=args.talent_reg_dir)
    if args.openml_reg:
        datasets.load_openml_regression()
    if args.custom_reg_dir:
        datasets.load_custom_dir(args.custom_reg_dir)

    models = ModelRegistry(device=args.device)
    checkpoints = args.checkpoint or [f"Synthefy:{download_checkpoint()}"]
    for raw in checkpoints:
        label, path = _parse_checkpoint(raw)
        models.add_checkpoint(
            label,
            path,
            device=args.device,
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
        gpu_mem_gb=args.gpu_mem_gb,
    )
    df = runner.run(sources=args.sources)
    if df is not None and len(df) and "r2" in df.columns:
        reg = df.dropna(subset=["r2"])
        if len(reg):
            print("\nMean R^2 by source (regression):")
            for src, g in reg.groupby("source"):
                print(f"  {src:12s} N={len(g):3d}  R2={g['r2'].mean():.4f}")
            print(f"  {'ALL':12s} N={len(reg):3d}  R2={reg['r2'].mean():.4f}")


if __name__ == "__main__":
    main()
