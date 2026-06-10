#!/usr/bin/env python3
"""Training entry point for Synthefy Tabular CCMM training.

Usage:
    # Single GPU
    synthefy-tabular-train --device cuda:2
    synthefy-tabular-train --device cuda:2 --total-steps 10 --no-wandb

    # Multi-GPU DDP
    torchrun --nproc_per_node=4 -m synthefy_tabular.training.cli --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from datetime import datetime

# Must set before model imports which check this env var at module load time
if '--no-flash-attn' in sys.argv:
    os.environ["SYNTHEFY_TABULAR_NO_FLASH_ATTN"] = "1"

import torch

from synthefy_tabular.utils.loading import build_model
from synthefy_tabular.training.config import TrainingConfig, package_config_path
from synthefy_tabular.training.trainer import SynthefyTabularTrainer


def load_model_config(source: str | None) -> dict:
    """Load model architecture config from a checkpoint or JSON file.

    When *source* is ``None`` the bundled ``model_base.json`` is used, so
    training can start from scratch without an external checkpoint.
    """
    if source is None:
        with open(package_config_path("model_base.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    if source.endswith(".json"):
        with open(source, "r", encoding="utf-8") as f:
            return json.load(f)
    state = torch.load(source, map_location='cpu', weights_only=False)
    if 'model_config' in state:
        return state['model_config']
    return state['config']


def parse_quantiles(raw: str) -> tuple[float, ...]:
    vals = [float(part.strip()) for part in raw.split(',') if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated quantile")
    vals = sorted(set(vals))
    if any(v <= 0 or v >= 1 for v in vals):
        raise argparse.ArgumentTypeError("Quantiles must satisfy 0 < q < 1")
    return tuple(vals)


def parse_tags(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(',') if part.strip())


def main():
    parser = argparse.ArgumentParser(description='Train Synthefy Tabular from scratch')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint (.ckpt/.pt) or JSON file to load model architecture config from. '
                             'Defaults to the bundled model_base.json.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Training checkpoint to resume from')
    parser.add_argument('--resume-model-only', action='store_true',
                        help='Load only model weights from --resume and reset optimizer/scheduler/step counters')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'muon'],
                        help='Optimizer to use (default: adamw)')
    parser.add_argument('--muon-include-embeddings', action='store_true',
                        help='Route 2D embedding tables to Muon instead of AdamW. '
                             'Default behavior excludes embeddings; this flag turns them on.')
    parser.add_argument('--muon-include-nd', action='store_true',
                        help='Route higher-rank (3D/4D) attention qkv/out_proj weights to '
                             'Muon via reshape-to-2D. Switches backend to MuonND (stock '
                             'torch.optim.Muon rejects ndim != 2). Captures the bulk of '
                             'attention parameters — the largest weight tensors in the model.')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--total-steps', type=int, default=200_000)
    parser.add_argument('--run-steps', type=int, default=None,
                        help='If set, run only this many optimizer steps in the current invocation while keeping total_steps as the global LR-schedule horizon')
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--decay-start-step', type=int, default=0,
                        help='Step to begin cosine decay (0 = right after warmup)')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default=os.environ.get('WANDB_PROJECT', 'synthefy-tabular'),
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=os.environ.get('WANDB_ENTITY'),
                        help='Optional Weights & Biases entity/team')
    parser.add_argument('--wandb-name', type=str, default=os.environ.get('WANDB_NAME'),
                        help='Optional Weights & Biases run name')
    parser.add_argument('--wandb-group', type=str, default=os.environ.get('WANDB_RUN_GROUP'),
                        help='Optional Weights & Biases run group')
    parser.add_argument('--wandb-job-type', type=str, default=os.environ.get('WANDB_JOB_TYPE'),
                        help='Optional Weights & Biases job type')
    parser.add_argument('--wandb-tags', type=parse_tags, default=parse_tags(os.environ.get('WANDB_TAGS')),
                        help='Comma-separated Weights & Biases tags')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Checkpoint directory (default: ./checkpoints/<timestamp>)')
    parser.add_argument('--ema-decay', type=float, default=0.0,
                        help='EMA decay for validation/checkpoint averaging (0 disables EMA)')
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=5000)
    parser.add_argument('--feature-loss-weight', type=float, default=0.5)
    parser.add_argument('--feature-loss-weight-end', type=float, default=None,
                        help='Optional end value for linear feature-loss decay')
    parser.add_argument('--feature-loss-decay-start-step', type=int, default=0,
                        help='Optimizer step to begin feature-loss decay')
    parser.add_argument('--feature-loss-decay-end-step', type=int, default=0,
                        help='Optimizer step to end feature-loss decay')
    parser.add_argument('--no-mixed-precision', action='store_true')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for kernel fusion speedup')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Checkpoint inter-layer activations to reduce VRAM')
    parser.add_argument('--max-budget', type=int, default=None,
                        help='Max n_samples*n_features per batch (OOM protection)')
    parser.add_argument('--min-samples', type=int, default=None,
                        help='Min n_samples per synthetic table (default: 50)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max n_samples per synthetic table (default: 2000)')
    parser.add_argument('--min-features', type=int, default=None,
                        help='Min n_features per synthetic table (default: 2)')
    parser.add_argument('--max-features', type=int, default=None,
                        help='Max n_features per synthetic table (default: 250)')
    parser.add_argument('--fixed-size', type=str, default=None,
                        help='Fixed NxF size (e.g. "1024x64"). Overrides random sampling.')
    parser.add_argument('--no-synth-v2', action='store_true',
                        help='Disable synth_v2 data augmentations (revert to Run 8 behavior)')
    parser.add_argument('--no-synth-v3', action='store_true',
                        help='Disable synth_v3 data augmentations (true categoricals, interactions, skewed targets)')
    parser.add_argument('--no-rich-reg-targets', action='store_true',
                        help='Disable rich regression targets (multi-feature deps + interactions in y)')
    parser.add_argument('--no-scale-variation', action='store_true',
                        help='Disable random target scale variation for regression')
    parser.add_argument('--synth-v4', action='store_true',
                        help='Enable synth_v4 data augmentations (TabICLv2-inspired diversity)')
    parser.add_argument('--no-v4-filter', action='store_true',
                        help='Disable ExtraTrees filtering in synth_v4')
    parser.add_argument('--learnability-filter', action='store_true',
                        help='Enable ExtraTrees learnability filter for all synthetic data (independent of synth_v4)')
    parser.add_argument('--learnability-filter-cls-min-score', type=float, default=0.60,
                        help='Minimum cls ExtraTrees OOB score to keep a dataset')
    parser.add_argument('--learnability-filter-cls-margin', type=float, default=0.10,
                        help='Minimum cls ExtraTrees margin above chance to keep a dataset')
    parser.add_argument('--learnability-filter-reg-min-score', type=float, default=0.10,
                        help='Minimum reg ExtraTrees OOB R2 to keep a dataset')
    parser.add_argument('--icl-filter-model', type=str, default='',
                        help='GPU model for ICL learnability filtering. Options: '
                             '"limix" to auto-download LimiX-2M from HuggingFace, '
                             '"hf" to auto-download the Synthefy checkpoint, '
                             'path to local checkpoint (.pt/.ckpt), '
                             'or empty string to disable. '
                             'Runs on the training GPU after each batch.')
    parser.add_argument('--icl-filter-cls-min-auc', type=float, default=0.55,
                        help='Minimum classification accuracy for ICL filter (default: 0.55)')
    parser.add_argument('--icl-filter-reg-min-r2', type=float, default=0.05,
                        help='Minimum regression R2 for ICL filter (default: 0.05)')
    parser.add_argument('--icl-scaling-filter', action='store_true',
                        help='Enable ICL scaling filter: reject datasets where more context does not help')
    parser.add_argument('--icl-scaling-min-improvement', type=float, default=0.03,
                        help='Minimum R2 improvement from small to large context (default: 0.03)')
    parser.add_argument('--v4-keep-edge-noise', action='store_true',
                        help='Keep Gaussian edge noise with synth_v4 (default: removed)')
    parser.add_argument('--task-type', type=str, default='both',
                        choices=['both', 'cls', 'reg'],
                        help='Task specialization: both (50/50), cls-only, or reg-only')
    parser.add_argument('--regression-ratio', type=float, default=0.5,
                        help='Fraction of steps that are regression when task_type=both (default: 0.5)')
    parser.add_argument('--regression-loss', type=str, default='mse',
                        choices=['mse', 'smooth_l1', 'huber', 'pinball', 'bar_distribution'],
                        help='Regression loss function (default: mse)')
    parser.add_argument('--regression-loss-beta', type=float, default=1.0,
                        help='Beta for smooth_l1/huber loss (default: 1.0)')
    parser.add_argument('--regression-quantiles', type=parse_quantiles,
                        default=parse_quantiles("0.1,0.25,0.5,0.75,0.9"),
                        help='Comma-separated quantiles for pinball loss (default: 0.1,0.25,0.5,0.75,0.9)')
    parser.add_argument('--pinball-tail-weight', type=float, default=0.0,
                        help='Upweight extreme quantiles in pinball loss. 0.0=uniform, '
                             '1.0=~2x weight at extremes (τ=0.01, τ=0.99). Targets bounded-'
                             'skewed regression datasets (sulfur, Goodreads). Default 0.0.')
    parser.add_argument('--pinball-monotonicity-weight', type=float, default=0.0,
                        help='Soft penalty for τ-crossing in pinball loss. Adds '
                             'relu(q[i] - q[i+1])^2 across adjacent τ. Improves smoothness '
                             'of the τ-mean point estimate (helps houses, space_ga, '
                             'physiochemical_protein). Try 0.05.')
    parser.add_argument('--pinball-mse-weight', type=float, default=0.0,
                        help='Auxiliary MSE on the τ-mean of pinball quantile predictions. '
                             'Forces unbiased point estimate; targets the QSAR-TID-11 '
                             '"prediction compression" failure where extreme-target rows '
                             'are averaged toward the bulk. Try 0.1.')
    parser.add_argument('--num-bars', type=int, default=5000,
                        help='Number of bins for --regression-loss bar_distribution. '
                             'The reg head is sized to output this many logits per row. '
                             'Default 5000; bin width = (bar_borders_high - bar_borders_low)/num_bars.')
    parser.add_argument('--bar-borders-low', type=float, default=-10.0,
                        help='Lower edge of bar_distribution bins on context-normalized y '
                             '(default -10.0 = 10 std below context mean).')
    parser.add_argument('--bar-borders-high', type=float, default=10.0,
                        help='Upper edge of bar_distribution bins on context-normalized y '
                             '(default +10.0 = 10 std above context mean).')
    parser.add_argument('--bar-target-sigma', type=float, default=0.0,
                        help='Soft-CE width for bar_distribution loss, in BIN units. '
                             '0.0 (default) = hard CE (one-hot target). >0 = Gaussian-smoothed '
                             'target with sigma=this many bins, giving partial credit to '
                             'nearby bins and smoothing the categorical cliffs in CE loss '
                             'landscape. Recommended: 5-20 for 5000 bins.')
    parser.add_argument('--reg-prior-prob', type=float, default=0.4,
                        help='Probability of using regression prior generator for reg episodes (default: 0.4)')
    parser.add_argument('--reg-denoise', action='store_true',
                        help='Reduce noise for regression episodes (noise features, missingness, Gaussian noise, target transforms)')
    parser.add_argument('--reg-deterministic-prob', type=float, default=0.20,
                        help='Probability that a regression-prior episode is zero-noise / deterministic (default: 0.20)')
    parser.add_argument('--reg-dense', action='store_true',
                        help='Dense-signal regression: flatter importances, fewer noise features, more parents, and more smooth multivariate priors')
    parser.add_argument('--probabilistic-labels', action='store_true',
                        help='Enable probabilistic classification labelers (logistic, tree, Gaussian, threshold)')
    parser.add_argument('--nominal-categoricals', action='store_true',
                        help='Enable nominal (non-ordinal) categorical feature generation')
    parser.add_argument('--enhanced-missingness', action='store_true',
                        help='Enable enhanced missingness patterns (row-level, block, target-dependent)')
    parser.add_argument('--clean-lowdim-prob', type=float, default=0.0,
                        help='Probability of clean low-dim regime episodes (default: 0.0)')
    parser.add_argument('--tree-prior-prob', type=float, default=0.0,
                        help='Probability of tree-ensemble prior episodes (default: 0.0)')
    parser.add_argument('--lookup-prior-prob', type=float, default=0.0,
                        help='Probability of categorical lookup prior episodes (default: 0.0)')
    parser.add_argument('--quadratic-surface-prob', type=float, default=0.0,
                        help='Probability of quadratic response surface episodes (default: 0.0)')
    parser.add_argument('--sparse-nonlinear-prob', type=float, default=0.0,
                        help='Probability of sparse nonlinear high-dim episodes (default: 0.0)')
    parser.add_argument('--gp-prior-prob', type=float, default=0.0,
                        help='Probability of GP smooth function episodes (default: 0.0)')
    parser.add_argument('--context-missingness-prob', type=float, default=0.0,
                        help='Probability of injecting NaN cells in context rows (default: 0.0)')
    parser.add_argument('--realistic-augmentation-prob', type=float, default=0.0,
                        help='Probability of applying heavy-tail/correlation/skew transforms (default: 0.0)')
    parser.add_argument('--y-transform-prob', type=float, default=0.0,
                        help='Probability of applying realistic y-target transforms per episode. '
                             'Simulates integer counts, censored values, ordinal ratings, bounded averages, '
                             'zero-inflated counts. Targets datasets we underperform on '
                             '(stock_fardamento02, boston, sensory, Goodreads, etc). Default 0.0 (off).')
    parser.add_argument('--cap-injection-prob', type=float, default=0.0,
                        help='Probability of injecting upper/lower quantile cap (saturation) per '
                             'regression episode. Minimal y-distortion — just clips tails to a '
                             'single value to teach the model that caps exist. Fixes models failure '
                             'to predict near timeout/cap values (MIP-2016, boston MEDV). Default 0.0.')
    parser.add_argument('--heavy-tail-prior-prob', type=float, default=0.0,
                        help='Probability of applying heavy-tail y transforms per regression episode. '
                             'Continuous transforms only: log-normal (exp), Pareto additive noise, '
                             'strong outlier injection (5-10%% @ 3-15x scale). No rounding — safe vs '
                             'failed y_transform experiment. Targets weak spots on skewed data '
                             '(stock_fardamento02, CPS1988, sulfur, Food_Delivery_Time). Default 0.0.')
    parser.add_argument('--pareto-importance-prob', type=float, default=0.0,
                        help='Probability per regression-prior episode of multiplying the per-feature '
                             'beta/coefficient vector by a Pareto-distributed importance vector so '
                             'a few features dominate while many are weak distractors. Targets the '
                             'high-dim "many distractors" failure (QSAR-TID-11, Allstate). Default 0.0.')
    parser.add_argument('--latent-factor-prob', type=float, default=0.0,
                        help='Probability per regression-prior episode of using a latent-factor '
                             'target: y = f(X @ V) where V is a d×k random projection (k ≪ d). '
                             'Models datasets with shared low-dim latent structure. Default 0.0.')
    parser.add_argument('--high-cap-prob', type=float, default=0.0,
                        help='Probability per cap-injected episode of using a higher cap fraction '
                             '(20-50%% at cap, percentile drawn from [50, 80] instead of [80, 97]). '
                             'Models timeout/runtime censoring patterns (MIP-2016, SAT11). '
                             'Composes with --cap-injection-prob (only applies when cap fires). '
                             'Default 0.0.')
    parser.add_argument('--low-unique-y-prob', type=float, default=0.0,
                        help='Probability per regression episode of rounding/binning y to '
                             '5-15 unique levels. Models discrete-regression datasets (Wine_Quality '
                             '6 levels, sensory 1-10 ratings). Default 0.0.')
    parser.add_argument(
        '--quality-filter-rules',
        type=str,
        default=None,
        help='Path to synthetic quality rules JSON (from eval_synthetic_benchmark.py)',
    )
    parser.add_argument(
        '--quality-filter-max-retries',
        type=int,
        default=3,
        help='Max retries when generated data fails quality rules (default: 3)',
    )
    parser.add_argument('--scm-prior', action='store_true',
                        help='Enable TabICL prior generator (MLP/Tree SCM + Reg2Cls)')
    parser.add_argument('--scm-prior-prob', type=float, default=0.5,
                        help='Per-dataset probability of using TabICL prior (default: 0.5)')
    parser.add_argument('--synth-v5', action='store_true',
                        help='Enable synth_v5 SCM improvements (informative categoricals, concat aggregation, multi-dim nodes)')
    parser.add_argument('--synth-v5-mixture', action='store_true',
                        help='Enable synth_v5 mixture prior (25%% v4, 35%% v5c, 40%% v5a per dataset)')
    parser.add_argument('--model-v2', action='store_true',
                        help='v2 arch: SwiGLU + RMSNorm + pre-norm/DeepNorm + PBLD')
    parser.add_argument('--model-v2-lite', action='store_true',
                        help='v2-lite arch: SwiGLU + RMSNorm + pre-norm/DeepNorm, keep RBF (no PBLD)')
    parser.add_argument('--embed-dim', type=int, default=None,
                        help='Override embedding dimension (default: from checkpoint)')
    parser.add_argument('--hid-dim', type=int, default=None,
                        help='Override MLP hidden dimension (default: from checkpoint)')
    parser.add_argument('--nhead', type=int, default=None,
                        help='Override number of attention heads (default: from checkpoint)')
    parser.add_argument('--nlayers', type=int, default=None,
                        help='Override number of transformer layers (default: from checkpoint)')
    parser.add_argument('--features-per-group', type=int, default=None,
                        help='Override features_per_group (default: from checkpoint, TabPFN uses 3)')
    parser.add_argument('--feature-positional-embedding-type', type=str, default=None,
                        choices=['subortho', 'learned', 'none'],
                        help='Feature positional embedding type. '
                             'subortho=random orthogonal each fwd (default). '
                             'learned=nn.Embedding table with permuted slots (TabPFN-2.6 style). '
                             'none=no feature positional embedding.')
    parser.add_argument('--feature-positional-embedding-num-slots', type=int, default=1000,
                        help='Number of learned positional slots when type=learned (default: 1000)')
    parser.add_argument('--no-qassmax', action='store_true',
                        help='Disable query-aware scalable softmax in sequence attention')
    parser.add_argument('--no-target-aware-embedding', action='store_true',
                        help='Disable target-aware label injection into context feature tokens')
    parser.add_argument('--column-specific-y-aware', action='store_true',
                        help='Add column-specific gating to target-aware embedding so each '
                             'feature column gets a y-derived bias scaled by its alignment '
                             'with the y embedding direction. Closes the gap with TabICL on '
                             'high-dim datasets where most features are noise (QSAR-TID-11). '
                             'Backward-compatible — alpha init makes new path contribute ~0.7%% '
                             'at FT step 0, so V8old behavior is preserved at start.')
    parser.add_argument('--freeze-column-y-alpha', action='store_true',
                        help='With --column-specific-y-aware, freeze column_y_aware_alpha at '
                             'its init value so it cannot grow during training. Control '
                             'experiment to isolate whether V10s gains come from the new '
                             'gating mechanism vs. just continued V8old FT.')
    parser.add_argument('--regression-only', action='store_true',
                        help='Regression-only forward path (strips cls branch for cleaner compile)')
    parser.add_argument('--target-aware-init-scale', type=float, default=1.0,
                        help='Initial scale for target-aware embedding during warm-up')
    parser.add_argument('--target-aware-warmup-steps', type=int, default=0,
                        help='Warm target-aware embedding to full scale over this many optimizer steps')
    parser.add_argument('--dim-bias-samples', type=float, default=1.5,
                        help='Exponent to bias n_samples toward larger tables (1.0=uniform, 1.5=40%% > 1000)')
    parser.add_argument('--dim-bias-features', type=float, default=1.3,
                        help='Exponent to bias n_features toward larger feature counts (1.0=uniform)')
    parser.add_argument('--no-flash-attn', action='store_true',
                        help='Disable flash attention (use PyTorch SDPA fallback)')
    parser.add_argument('--prefetch-workers', type=int, default=4,
                        help='Number of async data generation workers (0 to disable prefetching)')
    parser.add_argument('--prefetch-count', type=int, default=4,
                        help='Number of batches to prefetch ahead')
    parser.add_argument('--no-prefetch', action='store_true',
                        help='Disable async data prefetching')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data generation and training. Seeds Python, '
                             'NumPy, and torch (CPU + CUDA) so same-seed runs are '
                             'reproducible (default: 42)')
    parser.add_argument('--debug-dump-dir', type=str, default='',
                        help='Dump per-step NPZ files (data, gradients, '
                             'weights, loss) for the first --debug-dump-steps '
                             'optimizer steps into this directory. Setting this '
                             'also enables deterministic mode (seeded torch + '
                             'cudnn.deterministic).')
    parser.add_argument('--debug-dump-steps', type=int, default=5,
                        help='Number of optimizer steps to dump (default: 5)')
    args = parser.parse_args()

    # Seed Python/NumPy/torch from --seed so same-seed runs are reproducible.
    # Model init, the subortho feature embeddings, and the tabicl prior all draw
    # from the global torch RNG, so without this a run is only reproducible in
    # debug-dump mode. Seeding here is cheap and does not affect training speed.
    import random as _random
    import numpy as _np_seed
    _random.seed(args.seed)
    _np_seed.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.debug_dump_dir:
        # cudnn determinism disables fast nondeterministic kernels (slower), so it
        # is reserved for byte-exact debug dumps, not normal training.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[debug-dump] deterministic mode ON: seed={args.seed}, "
              f"cudnn.deterministic=True")

    # Auto-generate checkpoint directory with timestamp
    if args.checkpoint_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        args.checkpoint_dir = f'./checkpoints/{timestamp}'

    if args.run_steps is not None and args.run_steps <= 0:
        parser.error("--run-steps must be a positive integer")
    if args.feature_loss_weight_end is not None and args.feature_loss_weight_end < 0:
        parser.error("--feature-loss-weight-end must be >= 0")
    if args.feature_loss_decay_start_step < 0 or args.feature_loss_decay_end_step < 0:
        parser.error("--feature-loss-decay-start-step and --feature-loss-decay-end-step must be >= 0")
    if args.target_aware_warmup_steps < 0:
        parser.error("--target-aware-warmup-steps must be >= 0")
    if not 0.0 <= args.target_aware_init_scale <= 1.0:
        parser.error("--target-aware-init-scale must be in [0, 1]")
    # Load model architecture config
    config_source = args.checkpoint or "bundled model_base.json"
    print(f"Loading model config from {config_source}")
    model_config = load_model_config(args.checkpoint)

    # Enable mask prediction for training
    model_config['mask_prediction'] = True
    model_config.setdefault('decoder_config', {})
    # Regression head width depends on loss type:
    #   pinball          -> one output per quantile
    #   bar_distribution -> one output per bin (num_bars, fixed borders)
    #   other (mse/huber/smooth_l1) -> single scalar
    if args.regression_loss == 'pinball':
        _reg_head_width = len(args.regression_quantiles)
    elif args.regression_loss == 'bar_distribution':
        _reg_head_width = int(args.num_bars)
    else:
        _reg_head_width = 1
    model_config['decoder_config']['num_reg_quantiles'] = _reg_head_width
    # Persist bar-distribution config into the model_config so inference can
    # reconstruct borders without a separate flag.
    if args.regression_loss == 'bar_distribution':
        model_config['decoder_config']['regression_loss'] = 'bar_distribution'
        model_config['decoder_config']['num_bars'] = int(args.num_bars)
        model_config['decoder_config']['bar_borders_low'] = float(args.bar_borders_low)
        model_config['decoder_config']['bar_borders_high'] = float(args.bar_borders_high)
    model_config['use_qassmax'] = not args.no_qassmax
    model_config['use_target_aware_embedding'] = not args.no_target_aware_embedding
    model_config['use_column_specific_y_aware'] = bool(args.column_specific_y_aware)
    if args.regression_only:
        model_config['regression_only'] = True

    # --- Distributed setup ---
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1

    if distributed:
        # Per-layer compile can take >10min per new shape on 14 layers.
        # Default 600s NCCL timeout causes rank desync during compilation.
        import datetime as _dt
        nccl_timeout = int(os.environ.get('NCCL_TIMEOUT', '3600'))
        torch.distributed.init_process_group(
            backend='nccl',
            timeout=_dt.timedelta(seconds=nccl_timeout))
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        effective_lr = args.lr * world_size  # Linear scaling rule
    else:
        device = args.device
        # Set default CUDA device so that torch.compile, empty_cache(), etc.
        # target the correct GPU instead of defaulting to cuda:0.
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
        effective_lr = args.lr

    # Apply dimension overrides (before v2 arch changes since deepnorm_alpha depends on nlayers)
    for key, arg_val in [('embed_dim', args.embed_dim), ('hid_dim', args.hid_dim),
                         ('nhead', args.nhead), ('nlayers', args.nlayers),
                         ('features_per_group', args.features_per_group)]:
        if arg_val is not None:
            old_val = model_config.get(key)
            model_config[key] = arg_val
            if local_rank == 0:
                print(f"Model override: {key} {old_val} -> {arg_val}")

    # Feature positional embedding type / slots
    if args.feature_positional_embedding_type is not None:
        old_val = model_config.get('feature_positional_embedding_type', 'subortho')
        model_config['feature_positional_embedding_type'] = args.feature_positional_embedding_type
        model_config['feature_positional_embedding_num_slots'] = args.feature_positional_embedding_num_slots
        if local_rank == 0:
            print(
                f"Model override: feature_positional_embedding_type {old_val} -> "
                f"{args.feature_positional_embedding_type} "
                f"(num_slots={args.feature_positional_embedding_num_slots})"
            )

    # Propagate embed_dim into encoder sub-configs so that encoder output
    # dimensions match the transformer backbone.
    if args.embed_dim is not None:
        new_dim = args.embed_dim
        for sub_key in ('encoder_config_x', 'encoder_config_y'):
            sub = model_config.get(sub_key, {})
            for field in ('embedding_size', 'mask_embedding_size'):
                if field in sub:
                    sub[field] = new_dim

    # Propagate features_per_group into preprocess_config_x and encoder_config_x.
    # ValidFeatureEncoder pads input groups to this size, so it must match.
    if args.features_per_group is not None:
        fpg = args.features_per_group
        if 'preprocess_config_x' in model_config and 'num_features' in model_config['preprocess_config_x']:
            model_config['preprocess_config_x']['num_features'] = fpg
        if 'encoder_config_x' in model_config and 'num_features' in model_config['encoder_config_x']:
            model_config['encoder_config_x']['num_features'] = fpg
        if local_rank == 0:
            print(f"Propagated features_per_group={fpg} to preprocess_config_x and encoder_config_x")

    # Apply v2 architecture overrides
    if args.model_v2:
        model_config['activation'] = 'swiglu'
        model_config['norm_type'] = 'rmsnorm'
        model_config['pre_norm'] = True
        model_config['deepnorm_alpha'] = model_config['nlayers'] ** (-0.5)
        model_config['encoder_config_x']['numeric_embed_type'] = 'PBLD'
        model_config['encoder_config_x']['PBLD_config'] = {'n_frequencies': 48}
        if local_rank == 0:
            print("Model v2 enabled: SwiGLU + RMSNorm + pre-norm/DeepNorm + PBLD")
    elif args.model_v2_lite:
        model_config['activation'] = 'swiglu'
        model_config['norm_type'] = 'rmsnorm'
        model_config['pre_norm'] = True
        model_config['deepnorm_alpha'] = model_config['nlayers'] ** (-0.5)
        # Keep RBF numeric embedding (no PBLD) for speed
        if local_rank == 0:
            print("Model v2-lite enabled: SwiGLU + RMSNorm + pre-norm/DeepNorm (RBF kept)")
    if local_rank == 0:
        print(
            f"Architecture extras: QASSMax={'on' if model_config['use_qassmax'] else 'off'}, "
            f"TAE={'on' if model_config['use_target_aware_embedding'] else 'off'}"
        )

    # Build model from scratch (random init)
    if local_rank == 0:
        print("Building model from scratch (random initialization)")
    model = build_model(model_config)

    # Freeze column_y_aware_alpha if requested (V10c control experiment).
    if args.freeze_column_y_alpha and getattr(model, 'column_y_aware_alpha', None) is not None:
        model.column_y_aware_alpha.requires_grad_(False)
        if local_rank == 0:
            print(f"  [freeze] column_y_aware_alpha grad disabled "
                  f"(value held at {model.column_y_aware_alpha.item():.4f})")

    if local_rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    model.to(device)

    if args.gradient_checkpointing:
        model.transformer_encoder.gradient_checkpointing = True
        if local_rank == 0:
            print("Gradient checkpointing enabled on transformer encoder layers")

    # torch.compile (before DDP wrapping)
    # Note: dynamic=False with shape bucketing (trainer.SAMPLE_BUCKETS /
    # FEATURE_BUCKETS / CONTEXT_RATIO_BUCKETS) gives a bounded set of
    # compiled graphs. First encounter of each (shape, task) combo is slow
    # (~60-90s compilation) but subsequent hits are fast. For short runs,
    # use --fixed-size to avoid recompilation overhead.
    if args.compile:
        # Disable FX graph cache to avoid pickle errors with DDP + pybind11 objects
        os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '0'
        import torch._inductor.config as _inductor_cfg
        _inductor_cfg.fx_graph_cache = False
        _inductor_cfg.fx_graph_remote_cache = False
        import torch._dynamo.config as _dynamo_cfg
        _dynamo_cfg.optimize_ddp = False
        # Allow many more cached graphs — with shape bucketing we visit
        # 50-200 unique combos. Default 8 forces eviction + recompilation.
        _dynamo_cfg.cache_size_limit = 512
        _dynamo_cfg.accumulated_cache_size_limit = 512
        # Static shapes per bucket: skips symbolic shape analysis (the
        # pow_by_natural warning), compiles each bucket independently as a
        # static graph. Faster compile per bucket, no DDP rank desync from
        # symbolic-trace divergence.
        _dynamo_cfg.assume_static_by_default = True
        _dynamo_cfg.automatic_dynamic_shapes = False

        if local_rank == 0:
            print("Compiling model with torch.compile (FX cache disabled, DDP optimize off, "
                  "cache_size=512, static shapes)...")
        model = torch.compile(model, dynamic=False)

    # DDP wrapping (after compile)
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False,
                    gradient_as_bucket_view=True)
        if local_rank == 0:
            print(f"DDP enabled: {world_size} GPUs, effective LR={effective_lr:.2e}")

    # OOM budget: default 200K for single GPU, 100K with compile/DDP
    if args.max_budget is not None:
        max_budget = args.max_budget
    elif args.compile or distributed:
        max_budget = 100_000
    else:
        max_budget = 200_000

    # Parse fixed-size override
    fixed_n_samples = None
    fixed_n_features = None
    if args.fixed_size:
        parts = args.fixed_size.lower().split('x')
        if len(parts) != 2:
            parser.error(f"--fixed-size must be NxF (e.g. '1024x64'), got '{args.fixed_size}'")
        fixed_n_samples, fixed_n_features = int(parts[0]), int(parts[1])
        features_per_group = model_config.get('features_per_group', 2)
        if fixed_n_features % features_per_group != 0:
            fixed_n_features += features_per_group - (fixed_n_features % features_per_group)
        if local_rank == 0:
            print(f"Fixed size: {fixed_n_samples} samples x {fixed_n_features} features")

    # Training config
    train_config = TrainingConfig(
        device=device,
        optimizer=args.optimizer,
        muon_include_embeddings=args.muon_include_embeddings,
        muon_include_nd=args.muon_include_nd,
        lr=effective_lr,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        run_steps=args.run_steps,
        warmup_steps=args.warmup_steps,
        decay_start_step=args.decay_start_step,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity or None,
        wandb_name=args.wandb_name or None,
        wandb_group=args.wandb_group or None,
        wandb_job_type=args.wandb_job_type or None,
        wandb_tags=args.wandb_tags,
        checkpoint_dir=args.checkpoint_dir,
        ema_decay=args.ema_decay,
        gradient_accumulation=args.gradient_accumulation,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        feature_loss_weight=args.feature_loss_weight,
        feature_loss_weight_end=args.feature_loss_weight_end,
        feature_loss_decay_start_step=args.feature_loss_decay_start_step,
        feature_loss_decay_end_step=args.feature_loss_decay_end_step,
        mixed_precision=not args.no_mixed_precision,
        checkpoint_path=args.checkpoint or "",
        features_per_group=model_config.get('features_per_group', 2),
        target_aware_init_scale=args.target_aware_init_scale,
        target_aware_warmup_steps=args.target_aware_warmup_steps,
        compile=args.compile,
        distributed=distributed,
        local_rank=local_rank,
        world_size=world_size,
        max_sample_feature_budget=max_budget,
        **({"min_samples": args.min_samples} if args.min_samples is not None else {}),
        **({"max_samples": args.max_samples} if args.max_samples is not None else {}),
        **({"min_features": args.min_features} if args.min_features is not None else {}),
        **({"max_features": args.max_features} if args.max_features is not None else {}),
        fixed_n_samples=fixed_n_samples,
        fixed_n_features=fixed_n_features,
        synth_v2=not args.no_synth_v2,
        synth_v3=not args.no_synth_v3,
        rich_reg_targets=not args.no_rich_reg_targets,
        scale_variation=not args.no_scale_variation,
        synth_v4=args.synth_v4,
        v4_filter=not args.no_v4_filter,
        learnability_filter=args.learnability_filter,
        learnability_filter_cls_min_score=args.learnability_filter_cls_min_score,
        learnability_filter_cls_margin=args.learnability_filter_cls_margin,
        learnability_filter_reg_min_score=args.learnability_filter_reg_min_score,
        icl_filter_model=args.icl_filter_model,
        icl_filter_cls_min_auc=args.icl_filter_cls_min_auc,
        icl_filter_reg_min_r2=args.icl_filter_reg_min_r2,
        icl_scaling_filter=args.icl_scaling_filter,
        icl_scaling_min_improvement=args.icl_scaling_min_improvement,
        v4_no_edge_noise=not args.v4_keep_edge_noise,
        synth_v5=args.synth_v5,
        synth_v5_mixture=args.synth_v5_mixture,
        task_type=args.task_type,
        regression_ratio=args.regression_ratio,
        regression_loss=args.regression_loss,
        regression_loss_beta=args.regression_loss_beta,
        regression_quantiles=args.regression_quantiles,
        pinball_tail_weight=args.pinball_tail_weight,
        pinball_monotonicity_weight=args.pinball_monotonicity_weight,
        pinball_mse_weight=args.pinball_mse_weight,
        num_bars=args.num_bars,
        bar_borders_low=args.bar_borders_low,
        bar_borders_high=args.bar_borders_high,
        bar_target_sigma=args.bar_target_sigma,
        reg_prior_prob=args.reg_prior_prob,
        reg_denoise=args.reg_denoise,
        reg_deterministic_prob=args.reg_deterministic_prob,
        reg_dense=args.reg_dense,
        probabilistic_labels=args.probabilistic_labels,
        nominal_categoricals=args.nominal_categoricals,
        enhanced_missingness=args.enhanced_missingness,
        clean_lowdim_prob=args.clean_lowdim_prob,
        tree_prior_prob=args.tree_prior_prob,
        lookup_prior_prob=args.lookup_prior_prob,
        quadratic_surface_prob=args.quadratic_surface_prob,
        sparse_nonlinear_prob=args.sparse_nonlinear_prob,
        gp_prior_prob=args.gp_prior_prob,
        context_missingness_prob=args.context_missingness_prob,
        realistic_augmentation_prob=args.realistic_augmentation_prob,
        y_transform_prob=args.y_transform_prob,
        cap_injection_prob=args.cap_injection_prob,
        heavy_tail_prior_prob=args.heavy_tail_prior_prob,
        pareto_importance_prob=args.pareto_importance_prob,
        latent_factor_prob=args.latent_factor_prob,
        high_cap_prob=args.high_cap_prob,
        low_unique_y_prob=args.low_unique_y_prob,
        quality_filter_rules_path=args.quality_filter_rules,
        quality_filter_max_retries=args.quality_filter_max_retries,
        dim_bias_samples=args.dim_bias_samples,
        dim_bias_features=args.dim_bias_features,
        scm_prior=args.scm_prior,
        scm_prior_prob=args.scm_prior_prob,
        prefetch_workers=0 if args.no_prefetch else args.prefetch_workers,
        prefetch_count=args.prefetch_count,
        seed=args.seed,
        debug_dump_dir=args.debug_dump_dir,
        debug_dump_steps=args.debug_dump_steps,
    )

    # Create trainer (model already on device, skip internal .to())
    trainer = SynthefyTabularTrainer(model, train_config, model_config=model_config)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume, model_only=args.resume_model_only)

    # Start training
    trainer.train()

    # Cleanup distributed
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
