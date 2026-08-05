#!/usr/bin/env python3
"""Build or benchmark exact-shape compiler artifacts for Nori training."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
import types
from pathlib import Path

import torch

from synthefy_nori.model.layer import RMSNorm
from synthefy_nori.utils.loading import _safe_torch_load, build_model


def parse_shapes(raw: str) -> list[tuple[int, int, float]]:
    shapes = []
    seen = set()
    for item in raw.split(","):
        try:
            dims, ratio_raw = item.strip().split("@", maxsplit=1)
            rows_raw, features_raw = dims.lower().split("x", maxsplit=1)
            rows = int(rows_raw)
            features = int(features_raw)
            ratio = float(ratio_raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Shapes must use ROWSxFEATURES@CONTEXT_RATIO"
            ) from exc
        signature = (rows, features, ratio)
        if rows <= 1 or features <= 0 or not 0 < ratio < 1:
            raise argparse.ArgumentTypeError(f"Invalid shape signature: {item!r}")
        if signature in seen:
            raise argparse.ArgumentTypeError(f"Duplicate shape signature: {item!r}")
        seen.add(signature)
        shapes.append(signature)
    if not shapes:
        raise argparse.ArgumentTypeError("At least one shape signature is required")
    return shapes


def build_exact_model(
    checkpoint: str,
    device: torch.device,
    native_rms_norm: bool,
):
    state = _safe_torch_load(checkpoint)
    model = build_model(dict(state["model_config"]))
    model.load_state_dict(state["model_state_dict"])
    for name in (
        "feature_decoder",
        "cls_y_encoder",
        "cls_y_decoder",
        "cls_target_aware_embedding",
    ):
        module = getattr(model, name, None)
        if module is not None:
            module.requires_grad_(False)
    model._skip_feature_decoder = True
    if native_rms_norm:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.use_native = True
    return model.to(device).train()


def compile_model(model: torch.nn.Module, strategy: str, mode: str):
    if strategy == "eager":
        return model
    compile_mode = None if mode == "default" else mode
    if strategy in ("forward-dynamic", "forward-static"):
        layers = model.transformer_encoder.layers
        compiled_forward = torch.compile(
            type(layers[0]).forward,
            mode=compile_mode,
            dynamic=strategy == "forward-dynamic",
            fullgraph=False,
        )
        for layer in layers:
            layer.forward = types.MethodType(compiled_forward, layer)
        return model
    if strategy in ("stack-dynamic", "stack-static"):
        model.transformer_encoder = torch.compile(
            model.transformer_encoder,
            mode=compile_mode,
            dynamic=strategy == "stack-dynamic",
            fullgraph=False,
        )
        return model
    raise ValueError(strategy)


def make_batch(
    batch: int,
    rows: int,
    features: int,
    eval_pos: int,
    device: torch.device,
    seed: int,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch, rows, features, generator=generator)
    y = torch.randn(batch, rows, generator=generator)
    query_mask = (
        torch.rand(batch, rows - eval_pos, features, generator=generator) < 0.15
    )
    x[:, eval_pos:][query_mask] = float("nan")
    context = y[:, :eval_pos]
    y = (
        (y - context.mean(-1, keepdim=True))
        / context.std(-1, keepdim=True).clamp_min(1e-8)
    )
    return x.to(device), y.to(device)


def clear_grads(model):
    for parameter in model.parameters():
        parameter.grad = None


def counters_snapshot():
    from torch._dynamo.utils import counters

    return {
        str(category): {
            str(key): int(value)
            for key, value in values.items()
            if isinstance(value, int) and value
        }
        for category, values in counters.items()
        if values
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--native-rms-norm", action="store_true")
    parser.add_argument(
        "--strategy",
        choices=(
            "eager",
            "forward-dynamic",
            "forward-static",
            "stack-dynamic",
            "stack-static",
        ),
        default="eager",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument("--compile-cache-limit", type=int, default=1024)
    parser.add_argument("--disable-ddp-optimizer", action="store_true")
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=parse_shapes("256x64@0.5"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument(
        "--shape-order",
        choices=("interleaved", "grouped"),
        default="interleaved",
    )
    parser.add_argument("--checkpoint-threshold", type=int, default=24576)
    parser.add_argument(
        "--checkpointing", choices=("auto", "on", "off"), default="auto"
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    is_main = not distributed or local_rank == 0
    torch.cuda.set_device(device)
    torch.manual_seed(1234)
    torch._dynamo.reset()
    cache_limit = max(8, args.compile_cache_limit)
    torch._dynamo.config.cache_size_limit = cache_limit
    torch._dynamo.config.recompile_limit = cache_limit
    torch._dynamo.config.accumulated_cache_size_limit = cache_limit
    torch._dynamo.config.accumulated_recompile_limit = cache_limit
    if args.disable_ddp_optimizer:
        torch._dynamo.config.optimize_ddp = False

    model = build_exact_model(args.checkpoint, device, args.native_rms_norm)
    model = compile_model(model, args.strategy, args.compile_mode)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    records = []
    started_all = time.perf_counter()
    schedule = [
        (cycle, shape_index)
        for cycle in range(args.cycles)
        for shape_index in range(len(args.shapes))
    ]
    if args.shape_order == "grouped":
        schedule.sort(key=lambda item: (item[1], item[0]))

    for cycle, shape_index in schedule:
        rows, features, ratio = args.shapes[shape_index]
        eval_pos = max(1, min(rows - 1, int(rows * ratio)))
        x, y = make_batch(
            args.batch_size,
            rows,
            features,
            eval_pos,
            device,
            seed=cycle * 1000 + shape_index,
        )
        bare = model.module if distributed else model
        bare = getattr(bare, "_orig_mod", bare)
        if args.checkpointing == "auto":
            checkpointed = rows * features >= args.checkpoint_threshold
        else:
            checkpointed = args.checkpointing == "on"
        bare.transformer_encoder.gradient_checkpointing = checkpointed
        clear_grads(model)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(x=x, y=y, eval_pos=eval_pos, task_type="reg")
            target = y[:, eval_pos:].unsqueeze(-1)
            loss = (output["reg_output"].float() - target).square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        record = {
            "cycle": cycle,
            "rows": rows,
            "features": features,
            "ratio": ratio,
            "eval_pos": eval_pos,
            "checkpointed": checkpointed,
            "batch_size": args.batch_size,
            "seconds": time.perf_counter() - started,
            "peak_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "loss": float(loss.detach()),
        }
        records.append(record)
        if is_main:
            print(json.dumps(record), flush=True)
        del x, y, output, loss

    steady = [record["seconds"] for record in records if record["cycle"] > 0]
    summary = {
        "strategy": args.strategy,
        "native_rms_norm": args.native_rms_norm,
        "batch_size": args.batch_size,
        "shape_count": len(args.shapes),
        "shape_order": args.shape_order,
        "total_seconds": time.perf_counter() - started_all,
        "steady_mean_seconds": statistics.mean(steady) if steady else None,
        "counters": counters_snapshot(),
        "records": records,
    }
    if is_main:
        print("SUMMARY " + json.dumps(summary), flush=True)
    if args.output and is_main:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
    gc.collect()
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
