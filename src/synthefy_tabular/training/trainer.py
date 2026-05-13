"""Main training loop for LimiX CCMM training."""

import json
import os
import time
import math
import warnings
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from torch.amp import GradScaler

from synthefy_tabular.training.config import TrainingConfig
from synthefy_tabular.training.data_generator import generate_batch
from synthefy_tabular.training.masking import create_masks, random_mask_type, random_mask_ratio
from synthefy_tabular.training.loss import compute_ccmm_loss
from synthefy_tabular.training.optim import build_optimizer
from synthefy_tabular.training.prefetch import DataPrefetcher


def get_wcd_schedule(optimizer, warmup_steps, decay_start_step, total_steps, lr_min):
    """Warmup → Constant → Cosine Decay schedule.

    When decay_start_step <= warmup_steps (default 0), decay starts right
    after warmup — identical to the old cosine-with-warmup behavior.
    """
    # If decay_start_step is before warmup ends, start decay right after warmup
    effective_decay_start = max(decay_start_step, warmup_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step < effective_decay_start:
            return 1.0
        progress = (step - effective_decay_start) / max(total_steps - effective_decay_start, 1)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return max(lr_min / optimizer.defaults['lr'], cosine_decay)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LimiXTrainer:
    """Training loop for LimiX with CCMM objective."""

    def __init__(self, model, config: TrainingConfig, model_config: dict = None,
                 on_checkpoint_saved=None):
        self.model = model
        self.config = config
        self.model_config = model_config
        self.on_checkpoint_saved = on_checkpoint_saved
        self.device = torch.device(config.device)

        # In DDP mode, model is already on device from train_limix.py
        if not config.distributed:
            self.model.to(self.device)

        # Rank-0 flag for logging/checkpointing
        self.is_main = config.local_rank == 0

        # Optimizer
        self.optimizer, optimizer_stats = build_optimizer(
            self.model.named_parameters(),
            optimizer_name=config.optimizer,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            muon_momentum=config.muon_momentum,
            muon_nesterov=config.muon_nesterov,
            muon_adjust_lr_fn=config.muon_adjust_lr_fn,
            muon_include_embeddings=config.muon_include_embeddings,
            muon_include_nd=config.muon_include_nd,
        )
        if self.is_main:
            print(
                f"Optimizer: {config.optimizer} "
                f"(muon_tensors={optimizer_stats['muon_tensors']}, "
                f"adamw_tensors={optimizer_stats['adamw_tensors']})"
            )

        # Scheduler
        self.scheduler = get_wcd_schedule(
            self.optimizer,
            config.warmup_steps,
            config.decay_start_step,
            config.total_steps,
            config.lr_min,
        )

        # Mixed precision — use bf16 (not fp16).
        # fp16 max is 65504; the MLP normal_(std=1) init + post-norm cause
        # gradient overflow in fp16.  bf16 has the same exponent range as fp32.
        # GradScaler is unnecessary for bf16 (no overflow/underflow risk).
        self.scaler = GradScaler('cuda', enabled=False)

        # RNG — shared RNG for data shape/task/eval_pos (identical across ranks
        # so all GPUs get the same n_samples, n_features, task_type, n_classes,
        # eval_pos). Per-rank RNG for actual data generation and masking.
        # IMPORTANT: all shared_rng consumption must happen at the TOP of
        # train_step, before any error-prone code, to stay in sync across ranks.
        self.shared_rng = np.random.default_rng(config.seed)
        self.rng = np.random.default_rng(config.seed + config.local_rank)

        # State
        self.global_step = 0
        self.optimizer_step = 0
        self.accumulated_micro_steps = 0
        self.best_loss = float('inf')
        self.best_auc = -float('inf')
        self.best_r2 = -float('inf')
        self.best_early_stop_metric = -float('inf')
        self.best_early_stop_step = 0
        self.early_stop_bad_evals = 0
        self.early_stop_eval_count = 0
        self.should_stop_early = False

        # Loss spike detection (EMA-based)
        self._loss_ema = None
        self._loss_ema_alpha = 0.01  # slow-moving average
        self._loss_spike_threshold = 10.0  # skip if loss > threshold * EMA
        self.ema_decay = float(config.ema_decay)
        self.ema_state_dict = None
        if 0.0 < self.ema_decay < 1.0:
            self.ema_state_dict = {
                name: tensor.detach().clone()
                for name, tensor in self._get_bare_model().state_dict().items()
            }
            if self.is_main:
                print(f"EMA enabled: decay={self.ema_decay}")

        # Optional mined synthetic quality rules.
        self.quality_rules = None
        if config.quality_filter_rules_path:
            try:
                with open(config.quality_filter_rules_path, "r", encoding="utf-8") as f:
                    self.quality_rules = json.load(f)
                if self.is_main:
                    print(f"Loaded quality filter rules: {config.quality_filter_rules_path}")
            except Exception as e:
                if self.is_main:
                    print(
                        f"Warning: failed to load quality filter rules "
                        f"({config.quality_filter_rules_path}): {e}"
                    )
                self.quality_rules = None

        # Async data prefetching
        self.prefetcher = None
        if config.prefetch_workers > 0:
            self.prefetcher = DataPrefetcher(
                num_workers=config.prefetch_workers,
                prefetch_count=config.prefetch_count,
            )

        # GPU-batched ICL learnability filter: load a frozen LimiX checkpoint
        # onto the training GPU and run batched forward passes (~31ms for 8
        # episodes) instead of per-episode CPU inference in prefetch workers.
        self._icl_filter_model = None
        if config.icl_filter_model:
            self._init_gpu_icl_filter(config.icl_filter_model)

        self._set_target_aware_scale(self._get_current_target_aware_scale())

        # Wandb
        self.wandb_run = None

    # Shape buckets for torch.compile. These keep recompiles bounded while still
    # allowing larger late-curriculum tables.
    SAMPLE_BUCKETS = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
    FEATURE_BUCKETS = [4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 320, 384, 512, 768, 1024]
    CONTEXT_RATIO_BUCKETS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    def _sample_data_params(self):
        """Sample data generation parameters for one training step.

        Uses shared_rng so all DDP ranks get identical shape/task params.
        Also samples context_ratio here (before error-prone code) to keep
        shared_rng in sync even if generate_batch fails on one rank.
        """
        cfg = self.config
        rng = self.shared_rng  # Same across all ranks

        if cfg.fixed_n_samples is not None and cfg.fixed_n_features is not None:
            # Fixed size mode: skip random sampling but still consume rng
            # to keep shared_rng in sync with code that expects these draws.
            rng.uniform(0, 1)  # consume log_n_samples draw
            rng.uniform(0, 1)  # consume log_n_features draw
            n_samples = cfg.fixed_n_samples
            n_features = cfg.fixed_n_features
        else:
            # Biased log-uniform sampling: sample u ~ Uniform(0,1), raise to
            # power alpha (>1 biases toward max), then map to log-space range.
            # alpha=1.0 recovers standard log-uniform.
            log_min_s, log_max_s = np.log(cfg.min_samples), np.log(cfg.max_samples)
            u_s = rng.uniform(0, 1) ** (1.0 / cfg.dim_bias_samples)
            n_samples = int(np.exp(log_min_s + u_s * (log_max_s - log_min_s)))

            log_min_f, log_max_f = np.log(cfg.min_features), np.log(cfg.max_features)
            u_f = rng.uniform(0, 1) ** (1.0 / cfg.dim_bias_features)
            n_features = int(np.exp(log_min_f + u_f * (log_max_f - log_min_f)))
            # Must be even (features_per_group=2)
            if n_features % cfg.features_per_group != 0:
                n_features += cfg.features_per_group - (n_features % cfg.features_per_group)
            n_features = max(cfg.features_per_group, n_features)

        # OOM protection: cap dimensions to stay within budget.
        # Always respect min_samples — shrink features first.
        budget = cfg.max_sample_feature_budget
        if n_samples * n_features > budget:
            # Shrink features to fit n_samples within budget
            max_feat = budget // n_samples
            max_feat -= max_feat % cfg.features_per_group
            max_feat = max(cfg.features_per_group, max_feat)

            if max_feat >= cfg.features_per_group:
                n_features = min(n_features, max_feat)
            else:
                # Even min features won't fit — shrink samples too
                n_features = cfg.features_per_group
                n_samples = max(cfg.min_samples, budget // n_features)

        # Shape bucketing AFTER budget cap: round down to largest bucket
        # that fits within the budget-capped dimensions. Always bucket —
        # the model was trained on bucketed shapes and the offline pool
        # stores data in these exact buckets.
        valid_s = [b for b in self.SAMPLE_BUCKETS
                   if b <= n_samples and b >= cfg.min_samples]
        valid_f = [b for b in self.FEATURE_BUCKETS if b <= n_features]
        n_samples = max(valid_s) if valid_s else max(
            cfg.min_samples, self.SAMPLE_BUCKETS[0])
        n_features = max(valid_f) if valid_f else self.FEATURE_BUCKETS[0]

        # Task type selection
        if cfg.task_type == 'both':
            task_type = 'reg' if rng.random() < cfg.regression_ratio else 'cls'
        else:
            # Still consume the rng value to keep shared_rng in sync for DDP
            rng.random()
            task_type = cfg.task_type

        # Number of classes
        n_classes = None
        if task_type == 'cls':
            n_classes = int(rng.integers(cfg.min_classes, cfg.max_classes + 1))

        # Context/query split ratio — sampled here (shared_rng) so eval_pos
        # is identical across all ranks and shared_rng stays in sync.
        context_ratio = rng.uniform(cfg.context_ratio_min, cfg.context_ratio_max)

        # Quantize context_ratio to nearest bucket — keeps eval_pos bounded
        # and matches the distribution the model was trained on.
        valid_cr = [cr for cr in self.CONTEXT_RATIO_BUCKETS
                    if cr >= cfg.context_ratio_min and cr <= cfg.context_ratio_max]
        if valid_cr:
            context_ratio = min(valid_cr, key=lambda cr: abs(cr - context_ratio))

        return n_samples, n_features, task_type, n_classes, context_ratio

    def _build_gen_kwargs(self, n_samples, n_features, task_type, n_classes):
        """Build keyword arguments dict for generate_batch from config."""
        cfg = self.config
        return {
            'batch_size': cfg.batch_size,
            'n_samples': n_samples,
            'n_features': n_features,
            'task_type': task_type,
            'n_classes': n_classes,
            'augment': cfg.synth_v2,
            'augment_v3': cfg.synth_v3,
            'rich_reg_targets': cfg.rich_reg_targets,
            'scale_variation': cfg.scale_variation,
            'augment_v4': cfg.synth_v4,
            'v4_filter': cfg.v4_filter,
            'learnability_filter': cfg.learnability_filter,
            'v4_no_edge_noise': cfg.v4_no_edge_noise,
            'synth_v5': cfg.synth_v5,
            'synth_v5_mixture': cfg.synth_v5_mixture,
            'reg_prior_prob': cfg.reg_prior_prob,
            'reg_denoise': cfg.reg_denoise,
            'reg_deterministic_prob': cfg.reg_deterministic_prob,
            'reg_dense': cfg.reg_dense,
            'tabicl_prior': cfg.tabicl_prior,
            'tabicl_prior_prob': cfg.tabicl_prior_prob,
            'probabilistic_labels': cfg.probabilistic_labels,
            'nominal_categoricals': cfg.nominal_categoricals,
            'enhanced_missingness': cfg.enhanced_missingness,
            'clean_lowdim_prob': cfg.clean_lowdim_prob,
            'tree_prior_prob': cfg.tree_prior_prob,
            'lookup_prior_prob': cfg.lookup_prior_prob,
            'quadratic_surface_prob': cfg.quadratic_surface_prob,
            'sparse_nonlinear_prob': cfg.sparse_nonlinear_prob,
            'gp_prior_prob': cfg.gp_prior_prob,
            'context_missingness_prob': cfg.context_missingness_prob,
            'realistic_augmentation_prob': cfg.realistic_augmentation_prob,
            'y_transform_prob': cfg.y_transform_prob,
            'cap_injection_prob': cfg.cap_injection_prob,
            'heavy_tail_prior_prob': cfg.heavy_tail_prior_prob,
            'pareto_importance_prob': cfg.pareto_importance_prob,
            'latent_factor_prob': cfg.latent_factor_prob,
            'high_cap_prob': cfg.high_cap_prob,
            'low_unique_y_prob': cfg.low_unique_y_prob,
            'learnability_filter_cls_min_score': cfg.learnability_filter_cls_min_score,
            'learnability_filter_cls_margin': cfg.learnability_filter_cls_margin,
            'learnability_filter_reg_min_score': cfg.learnability_filter_reg_min_score,
            'icl_scaling_filter': cfg.icl_scaling_filter,
            'icl_scaling_min_improvement': cfg.icl_scaling_min_improvement,
            'quality_rules': self.quality_rules,
            'filter_max_retries': cfg.quality_filter_max_retries,
        }

    def _get_current_feature_loss_weight(self) -> float:
        """Feature loss schedule in optimizer-step units."""
        cfg = self.config
        start = float(cfg.feature_loss_weight)
        end = cfg.feature_loss_weight_end
        if end is None or cfg.feature_loss_decay_end_step <= cfg.feature_loss_decay_start_step:
            return start

        step = int(self.optimizer_step)
        start_step = int(cfg.feature_loss_decay_start_step)
        end_step = int(cfg.feature_loss_decay_end_step)
        if step <= start_step:
            return start
        if step >= end_step:
            return float(end)

        alpha = (step - start_step) / max(end_step - start_step, 1)
        return float((1.0 - alpha) * start + alpha * float(end))

    def _get_current_target_aware_scale(self) -> float:
        """Warm up target-aware embedding from init_scale to 1.0."""
        cfg = self.config
        init_scale = float(cfg.target_aware_init_scale)
        warmup_steps = int(cfg.target_aware_warmup_steps)
        if warmup_steps <= 0:
            return 1.0

        step = min(max(int(self.optimizer_step), 0), warmup_steps)
        alpha = step / max(warmup_steps, 1)
        return float((1.0 - alpha) * init_scale + alpha * 1.0)

    def _set_target_aware_scale(self, scale: float) -> None:
        bare_model = self._get_bare_model()
        if hasattr(bare_model, 'target_aware_scale'):
            bare_model.target_aware_scale = float(scale)

    def _submit_prefetch(self, n_samples, n_features, task_type, n_classes):
        """Submit one batch to the prefetcher with a deterministic seed."""
        seed = int(self.rng.integers(0, 2**63))
        gen_kwargs = self._build_gen_kwargs(
            n_samples, n_features, task_type, n_classes)
        self.prefetcher.submit(seed=seed, gen_kwargs=gen_kwargs)

    # ── GPU-batched ICL learnability filter ──────────────────────────────

    def _init_gpu_icl_filter(self, model_path):
        """Load a frozen model for GPU-batched learnability filtering.

        Supports:
          - LimiX checkpoints (.pt/.ckpt) — uses native forward pass
          - 'tabicl' — loads TabICLv2 regressor/classifier
          - 'tabpfn' — loads TabPFN-2.5 regressor/classifier

        The model_path can be a file path (LimiX) or a string identifier
        ('tabicl', 'tabpfn') for library models.
        """
        self._icl_filter_type = 'limix'  # default

        if model_path == 'tabicl':
            try:
                from tabicl import TabICLRegressor
                m = TabICLRegressor(device=str(self.device))
                self._icl_filter_model = m
                self._icl_filter_type = 'tabicl'
                if self.is_main:
                    print(f"GPU ICL filter: TabICLv2 -> {self.device}")
                return
            except ImportError:
                raise ImportError("tabicl not installed. Run: pip install tabicl")

        elif model_path == 'tabpfn':
            try:
                from tabpfn import TabPFNRegressor
                m = TabPFNRegressor(device=str(self.device),
                                    ignore_pretraining_limits=True)
                self._icl_filter_model = m
                self._icl_filter_type = 'tabpfn'
                if self.is_main:
                    print(f"GPU ICL filter: TabPFN -> {self.device}")
                return
            except ImportError:
                raise ImportError("tabpfn not installed. Run: pip install tabpfn")

        else:
            # LimiX checkpoint
            from synthefy_tabular.utils.loading import load_model
            m = load_model(model_path, mask_prediction=True)
            m.eval()
            m.to(self.device)
            for p in m.parameters():
                p.requires_grad_(False)
            self._icl_filter_model = m
            self._icl_filter_type = 'limix'
            if self.is_main:
                print(f"GPU ICL filter: LimiX {model_path} -> {self.device}")

    @torch.no_grad()
    def _gpu_icl_filter(self, X_batch, y_batch, task_type, n_classes):
        """Run batched ICL learnability check on GPU.

        Supports LimiX (native forward), TabICLv2 (fit/predict), and
        TabPFN (fit/predict). The model type is determined at init time
        by _init_gpu_icl_filter.

        Args:
            X_batch: np.ndarray [B, n_samples, n_features]
            y_batch: np.ndarray [B, n_samples]
            task_type: 'cls' or 'reg'
            n_classes: int or None

        Returns:
            passed: np.ndarray[bool] of shape [B] — True if episode is learnable
        """
        cfg = self.config
        B, N, F = X_batch.shape

        if self._icl_filter_type in ('tabicl', 'tabpfn'):
            return self._gpu_icl_filter_sklearn(
                X_batch, y_batch, task_type, n_classes)
        else:
            return self._gpu_icl_filter_limix(
                X_batch, y_batch, task_type, n_classes)

    def _gpu_icl_filter_sklearn(self, X_batch, y_batch, task_type, n_classes):
        """Filter using sklearn-compatible models (TabICLv2, TabPFN).

        Per-episode fit/predict — slower than batched LimiX but uses
        the actual model's learning capability as the filter criterion.
        """
        cfg = self.config
        B, N, F = X_batch.shape
        max_ctx = min(500, int(N * 0.7))
        max_feat = min(100, F)
        passed = np.ones(B, dtype=bool)

        model = self._icl_filter_model

        for b in range(B):
            try:
                X = X_batch[b].copy()
                y = y_batch[b].copy()

                # Subsample features for speed
                if F > max_feat:
                    feat_idx = np.random.default_rng(b).choice(
                        F, max_feat, replace=False)
                    X = X[:, feat_idx]

                X_clean = np.nan_to_num(X, nan=0.0).astype(np.float32)

                # Context/query split
                n_ctx = min(max_ctx, N - 20)
                X_train, y_train = X_clean[:n_ctx], y[:n_ctx]
                X_test, y_test = X_clean[n_ctx:], y[n_ctx:]

                if len(X_test) < 10:
                    continue

                if task_type == 'reg':
                    # Normalize y using context stats
                    y_mean = np.nanmean(y_train)
                    y_std = np.nanstd(y_train)
                    if y_std < 1e-8:
                        passed[b] = False
                        continue
                    y_train_norm = ((y_train - y_mean) / y_std).astype(np.float64)
                    y_test_norm = ((y_test - y_mean) / y_std).astype(np.float64)

                    model.fit(X_train.astype(np.float32),
                              y_train_norm.astype(np.float64))
                    preds = model.predict(X_test.astype(np.float32))
                    preds = np.asarray(preds, dtype=np.float64).squeeze()

                    ss_res = ((y_test_norm - preds) ** 2).sum()
                    ss_tot = ((y_test_norm - y_test_norm.mean()) ** 2).sum()
                    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
                    passed[b] = r2 > cfg.icl_filter_reg_min_r2
                else:
                    model.fit(X_train.astype(np.float32),
                              y_train.astype(np.int64))
                    preds = model.predict(X_test.astype(np.float32))
                    acc = (preds == y_test.astype(np.int64)).mean()
                    chance = 1.0 / max(n_classes or 2, 2)
                    passed[b] = acc > chance + 0.05

            except Exception:
                passed[b] = True  # don't filter on error

        return passed

    def _gpu_icl_filter_limix(self, X_batch, y_batch, task_type, n_classes):
        """Filter using frozen LimiX model (batched GPU forward pass).

        Uses the full dataset size (not a tiny subsample) so the filter
        decision matches what the training model will actually see.
        Processes one episode at a time to handle variable OOM risk.
        """
        cfg = self.config
        B, N, F = X_batch.shape

        # Use 70% context / 30% query — matches training context_ratio range
        n_ctx = max(20, int(N * 0.7))
        n_qry = N - n_ctx

        X_sub = X_batch.copy()
        y_sub = y_batch.copy()

        X_sub = np.nan_to_num(X_sub, nan=0.0).astype(np.float32)

        if task_type == 'cls':
            nc = min(n_classes or 10, 10)
            y_sub = np.clip(y_sub, 0, nc - 1).astype(np.float32)
        else:
            ctx_y = y_sub[:, :n_ctx]
            ctx_mean = np.nanmean(ctx_y, axis=-1, keepdims=True)
            ctx_std = np.nanstd(ctx_y, axis=-1, keepdims=True)
            ctx_std = np.where(ctx_std < 1e-8, 1.0, ctx_std)
            y_sub = ((y_sub - ctx_mean) / ctx_std).astype(np.float32)
            nc = None

        passed = np.ones(B, dtype=bool)

        def _eval_batch(X_np, y_np):
            """Run filter on a batch slice, return R2 array or None on OOM."""
            try:
                x_t = torch.from_numpy(X_np).to(self.device)
                y_t = torch.from_numpy(y_np).to(self.device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = self._icl_filter_model(x_t, y_t, eval_pos=n_ctx,
                                                 task_type=task_type)
                if task_type == 'cls':
                    logits = out['cls_output'][:, :n_qry, :nc].float().cpu().numpy()
                    preds = logits.argmax(axis=-1)
                    true = y_np[:, n_ctx:n_ctx + n_qry].astype(np.int64)
                    acc = (preds == true).mean(axis=-1)
                    chance = 1.0 / nc
                    margin = cfg.icl_filter_cls_min_auc - 0.5
                    return acc > (chance + margin)
                else:
                    preds = out['reg_output'][:, :n_qry, 0].float().cpu().numpy()
                    true = y_np[:, n_ctx:n_ctx + n_qry]
                    ss_res = ((true - preds) ** 2).sum(axis=-1)
                    ss_tot = ((true - true.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
                    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-8)
                    return r2 > cfg.icl_filter_reg_min_r2
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                torch.cuda.empty_cache()
                return None

        # Try full batch first (fastest)
        result = _eval_batch(X_sub, y_sub)
        if result is not None:
            return result

        # OOM: fall back to per-episode
        for b in range(B):
            result = _eval_batch(X_sub[b:b+1], y_sub[b:b+1])
            if result is not None:
                passed[b] = result[0]
            # else: OOM even on single episode, keep it

        return passed

    def _filter_and_replace(self, X_batch, y_batch, task_type, n_classes,
                            n_samples, n_features):
        """Filter a batch with GPU ICL and replace rejected episodes.

        Rejected episodes are regenerated synchronously (without ICL filter,
        since the replacement itself gets a GPU check on the next pass).
        At most 2 rounds of replacement to avoid infinite loops.
        """
        passed = self._gpu_icl_filter(X_batch, y_batch, task_type, n_classes)
        n_rejected = int((~passed).sum())
        if n_rejected == 0:
            return X_batch, y_batch

        if self.is_main and self.global_step % self.config.log_interval == 0:
            print(f"  [ICL-GPU] Rejected {n_rejected}/{len(passed)} episodes")

        gen_kwargs = self._build_gen_kwargs(n_samples, n_features, task_type, n_classes)
        gen_kwargs.pop('icl_filter_model', None)
        gen_kwargs.pop('batch_size', None)

        for _round in range(2):
            bad_idx = np.where(~passed)[0]
            if len(bad_idx) == 0:
                break
            for i in bad_idx:
                rng_replace = np.random.default_rng(
                    int(self.rng.integers(0, 2**63)))
                X_new, y_new, _ = generate_batch(
                    batch_size=1, rng=rng_replace, **gen_kwargs)
                X_batch[i] = X_new[0]
                y_batch[i] = y_new[0]

            passed = self._gpu_icl_filter(X_batch, y_batch, task_type, n_classes)

        return X_batch, y_batch

    def _prepare_batch(self, X_batch, y_batch, task_type, n_classes, context_ratio):
        """Prepare a training batch with masking.

        Args:
            X_batch: [batch, n_samples, n_features]
            y_batch: [batch, n_samples]
            task_type: 'cls' or 'reg'
            n_classes: int or None
            context_ratio: float, pre-sampled from shared_rng for DDP sync

        Returns:
            x_input: [batch, seq_len, n_features] with masked values as NaN
            y_input: [batch, seq_len]
            eval_pos: int
            x_original: [batch, seq_len, n_features] unmasked
            feature_mask: [batch, seq_len, n_features]
            y_query: [batch, n_query] ground truth targets
        """
        cfg = self.config
        batch_size, n_samples, n_features = X_batch.shape

        # --- Column permutation (anti-memorization) ---
        # The SCM always places root nodes in early columns and downstream
        # nodes later. Without shuffling, the model can learn "attend to
        # later columns" as a shortcut instead of discovering feature
        # relationships in-context. Shuffle independently per episode.
        for b in range(batch_size):
            perm = self.rng.permutation(n_features)
            X_batch[b] = X_batch[b][:, perm]

        # --- Label permutation (classification anti-memorization) ---
        # Class labels from quantile bucketing always have class 0 = lowest
        # quantile, class N = highest. Without shuffling, the model learns
        # "class ordering = magnitude ordering" instead of treating labels
        # as arbitrary identifiers. Apply random bijection per episode.
        if task_type == 'cls' and n_classes is not None and n_classes > 1:
            for b in range(batch_size):
                perm = self.rng.permutation(n_classes)
                y_int = y_batch[b].astype(np.int64).clip(0, n_classes - 1)
                y_batch[b] = perm[y_int].astype(np.float32)

        # Context/query split from pre-sampled ratio (synced across ranks)
        eval_pos = max(1, int(n_samples * context_ratio))
        eval_pos = min(eval_pos, n_samples - 1)  # At least 1 query row

        # For regression, re-normalize y using context-only stats to match
        # inference. The generator normalizes with global (all-row) stats, but
        # at inference the evaluator normalizes with train-only stats. This
        # mismatch creates a systematic scale/shift error that hurts R².
        if task_type == 'reg':
            context_y = y_batch[:, :eval_pos]
            y_ctx_mean = context_y.mean(axis=-1, keepdims=True)
            y_ctx_std = context_y.std(axis=-1, keepdims=True)
            y_ctx_std = np.where(y_ctx_std < 1e-8, 1.0, y_ctx_std)
            y_batch = (y_batch - y_ctx_mean) / y_ctx_std

        n_query = n_samples - eval_pos

        # Create feature masks for query rows (per-rank rng for diversity).
        # Context rows are NOT masked here — context missingness is applied
        # during data generation instead (see _apply_context_missingness in
        # data_generator.py), which keeps the compiled graph stable because
        # the input data already contains NaNs when the compiler first traces.
        mask_type = random_mask_type(self.rng)
        mask_ratio = random_mask_ratio(cfg.mask_ratio_min, cfg.mask_ratio_max, self.rng)

        # Generate masks: [batch, n_query, n_features]
        feature_masks = []
        for _ in range(batch_size):
            m = create_masks(n_query, n_features, mask_ratio, mask_type, self.rng)
            feature_masks.append(m)
        query_mask = np.stack(feature_masks, axis=0)  # [B, n_query, n_features]

        # Full mask: context rows clean, query rows masked
        full_mask = np.zeros((batch_size, n_samples, n_features), dtype=bool)
        full_mask[:, eval_pos:, :] = query_mask

        # Apply masks to X
        x_original = X_batch.copy()
        x_masked = X_batch.copy()
        x_masked[full_mask] = np.nan

        # Convert to tensors
        x_input = torch.from_numpy(x_masked).float().to(self.device)
        y_input = torch.from_numpy(y_batch).float().to(self.device)
        x_original_t = torch.from_numpy(x_original).float().to(self.device)
        feature_mask_t = torch.from_numpy(full_mask).bool().to(self.device)
        y_query = torch.from_numpy(y_batch[:, eval_pos:]).float().to(self.device)

        return x_input, y_input, eval_pos, x_original_t, feature_mask_t, y_query

    def _log_diagnostics(self, output):
        """Print diagnostic information for debugging convergence."""
        import torch

        # --- Gradient norms by component (after unscale, before clip) ---
        component_norms = {}
        n_zero = 0
        n_total = 0
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            n_total += 1
            norm = p.grad.data.norm(2).item()
            if norm == 0:
                n_zero += 1
            if not math.isfinite(norm):
                continue
            if 'transformer_encoder' in name:
                key = 'tfm'
            elif 'cls_y_decoder' in name:
                key = 'cls_dec'
            elif 'reg_y_decoder' in name:
                key = 'reg_dec'
            elif 'feature_decoder' in name:
                key = 'feat_dec'
            elif 'encoder_x' in name:
                key = 'x_enc'
            elif 'cls_y_encoder' in name:
                key = 'cls_enc'
            elif 'reg_y_encoder' in name:
                key = 'reg_enc'
            elif 'x_preprocess' in name:
                key = 'preproc'
            elif 'feature_positional' in name:
                key = 'feat_pos'
            else:
                key = 'other'
            component_norms.setdefault(key, 0.0)
            component_norms[key] += norm ** 2

        component_norms = {k: v ** 0.5 for k, v in component_norms.items()}
        total_norm = sum(v ** 2 for v in component_norms.values()) ** 0.5

        scale = self.scaler.get_scale()
        parts = " ".join(f"{k}={v:.4e}" for k, v in sorted(component_norms.items()))
        print(f"  [DIAG] grad_norm={total_norm:.4e} scale={scale:.0f} "
              f"zero_grad={n_zero}/{n_total}")
        print(f"  [DIAG] {parts}")

        # --- Output statistics ---
        if 'cls_output' in output and output['cls_output'].numel() > 0:
            co = output['cls_output'].detach().float()
            probs = torch.softmax(co, dim=-1)
            max_prob = probs.max(dim=-1).values.mean()
            print(f"  [DIAG] cls: logit_mean={co.mean():.4f} logit_std={co.std():.4f} "
                  f"max_prob={max_prob:.4f}")
        if 'reg_output' in output and output['reg_output'].numel() > 0:
            ro = output['reg_output'].detach().float()
            print(f"  [DIAG] reg: mean={ro.mean():.4f} std={ro.std():.4f}")

        fp = output['feature_pred'].detach().float()
        print(f"  [DIAG] feat_pred: mean={fp.mean():.4f} std={fp.std():.4f}")

        # --- Process config sanity ---
        pc = output['process_config']
        if pc.get('std_for_normalization') is not None:
            std_n = pc['std_for_normalization'].detach()
            n_zero_std = (std_n.abs() < 1e-6).sum().item()
            if n_zero_std > 0:
                print(f"  [DIAG] WARNING: {n_zero_std}/{std_n.numel()} near-zero std features")

    def _ddp_should_skip(self, should_skip):
        """In DDP, sync skip decision: if ANY rank wants to skip, ALL skip.

        This prevents deadlocks where one rank calls backward() (triggering
        all-reduce) while another rank skips it.
        """
        if not self.config.distributed:
            return should_skip
        skip_tensor = torch.tensor(
            [1.0 if should_skip else 0.0], device=self.device)
        torch.distributed.all_reduce(skip_tensor, op=torch.distributed.ReduceOp.MAX)
        return skip_tensor.item() > 0.5

    def train_step(self):
        """Execute one training step. Handles all errors internally.

        In DDP mode, all ranks are guaranteed to either all call backward()
        or all skip it, preventing deadlocks.

        Returns:
            loss_dict: dict with loss components, always includes 'skipped' key
        """
        self.model.train()
        cfg = self.config

        # --- Sample data parameters (shared_rng, consumed FIRST) ---
        # All shared_rng consumption happens here, before any error-prone code,
        # so even if generate_batch/forward fails on one rank, shared_rng stays
        # in sync across all ranks.
        if (self.prefetcher is not None
                and hasattr(self, '_prefetch_params_queue')
                and self._prefetch_params_queue):
            # Async path: params were pre-sampled during prefill/previous step.
            n_samples, n_features, task_type, n_classes, context_ratio = \
                self._prefetch_params_queue.pop(0)
        else:
            n_samples, n_features, task_type, n_classes, context_ratio = \
                self._sample_data_params()

        current_feature_loss_weight = self._get_current_feature_loss_weight()
        current_tae_scale = self._get_current_target_aware_scale()
        self._set_target_aware_scale(current_tae_scale)

        # Build skip-result template
        def _skip_result():
            return {
                'total_loss': 0, 'y_loss': 0, 'feat_loss': 0,
                'skipped': True,
                'lr': self.scheduler.get_last_lr()[0],
                'task_type': task_type,
                'n_samples': n_samples,
                'n_features': n_features,
                'eval_pos': 0,
                'feature_loss_weight': current_feature_loss_weight,
                'target_aware_scale': current_tae_scale,
                'optimizer_stepped': False,
                'optimizer_step': self.optimizer_step,
            }

        should_skip = False
        output = None
        loss = None
        loss_val = 0.0
        loss_dict = None
        eval_pos = 0

        # --- Generate batch + forward + loss (can error) ---
        try:
            if self.prefetcher is not None:
                # Async path: get pre-generated batch from prefetcher.
                # The batch for THIS step was submitted earlier (in
                # _prefill_pipeline or previous train_step). We also
                # consume self.rng to keep it advancing (the seed was
                # already drawn when this batch was submitted).
                result = self.prefetcher.get()
                from synthefy_tabular.training.prefetch import _ErrorSentinel
                if isinstance(result, _ErrorSentinel):
                    # Treat worker errors as skippable (don't crash DDP).
                    # Print full traceback for diagnosis then skip via ValueError.
                    if self.is_main:
                        print(f"  [SKIP] Worker error at step {self.global_step}: "
                              f"{result.err_type}: {result.err_msg}\n"
                              f"{result.tb}")
                    raise ValueError(f"Worker error: {result.err_msg}")
                X_batch, y_batch, n_classes = result
            else:
                # Synchronous path (prefetch disabled)
                gen_kwargs = self._build_gen_kwargs(
                    n_samples, n_features, task_type, n_classes)
                X_batch, y_batch, n_classes = generate_batch(
                    rng=self.rng,
                    **gen_kwargs,
                )

            # GPU-batched ICL learnability filter: single forward pass for
            # all episodes (~31ms for batch=8), replacing unlearnable ones.
            if self._icl_filter_model is not None:
                X_batch, y_batch = self._filter_and_replace(
                    X_batch, y_batch, task_type, n_classes,
                    n_samples, n_features)

            x_input, y_input, eval_pos, x_original, feature_mask, y_query = \
                self._prepare_batch(X_batch, y_batch, task_type, n_classes,
                                    context_ratio)

            # Forward pass
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=cfg.mixed_precision):
                output = self.model(
                    x=x_input,
                    y=y_input,
                    eval_pos=eval_pos,
                    task_type=task_type,
                )

            # Compute loss outside autocast for float32 precision
            loss, loss_dict = compute_ccmm_loss(
                model_output=output,
                y_true=y_query,
                x_original=x_original,
                feature_mask=feature_mask,
                task_type=task_type,
                n_classes=n_classes,
                feature_loss_weight=current_feature_loss_weight,
                regression_loss=cfg.regression_loss,
                regression_loss_beta=cfg.regression_loss_beta,
                regression_quantiles=cfg.regression_quantiles,
                pinball_tail_weight=cfg.pinball_tail_weight,
                pinball_monotonicity_weight=cfg.pinball_monotonicity_weight,
                pinball_mse_weight=cfg.pinball_mse_weight,
                num_bars=cfg.num_bars,
                bar_borders_low=cfg.bar_borders_low,
                bar_borders_high=cfg.bar_borders_high,
                bar_target_sigma=cfg.bar_target_sigma,
            )
            loss_dict['feature_loss_weight'] = current_feature_loss_weight
            loss_dict['target_aware_scale'] = current_tae_scale

            loss_val = loss.item()

            # NaN / Inf check
            if not math.isfinite(loss_val):
                if self.is_main:
                    print(f"  [SKIP] NaN/Inf loss at step {self.global_step} "
                          f"(task={task_type}, n={n_samples}, f={n_features})")
                should_skip = True

            # Loss spike check
            elif (self._loss_ema is not None
                  and loss_val > self._loss_spike_threshold * self._loss_ema
                  and self._loss_ema > 0):
                if self.is_main:
                    y_l = loss_dict.get('y_loss', 0)
                    f_l = loss_dict.get('feat_loss', 0)
                    print(f"  [SKIP] Loss spike at step {self.global_step}: "
                          f"{loss_val:.4f} > {self._loss_spike_threshold}x EMA "
                          f"({self._loss_ema:.4f}) "
                          f"[y={y_l:.2f} feat={f_l:.2f} task={task_type}]")
                should_skip = True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if self.is_main:
                    print(f"  [SKIP] OOM at step {self.global_step} "
                          f"(n={n_samples}, f={n_features})")
                torch.cuda.empty_cache()
                should_skip = True
            else:
                raise
        except ValueError as e:
            if self.is_main:
                print(f"  [SKIP] Data error at step {self.global_step}: {e}")
            should_skip = True
        except Exception as e:
            # Catch-all for unexpected errors (OverflowError, TypeError, etc.)
            # Without this, loss stays None and the step crashes at backward()
            if self.is_main:
                print(f"  [SKIP] Unexpected error at step {self.global_step}: "
                      f"{type(e).__name__}: {e}")
            should_skip = True

        # Safety: if loss is None despite no exception, force skip
        if loss is None and not should_skip:
            if self.is_main:
                print(f"  [SKIP] loss=None at step {self.global_step}")
            should_skip = True

        # --- DDP sync: if ANY rank wants to skip, ALL skip ---
        should_skip = self._ddp_should_skip(should_skip)

        if should_skip:
            # During gradient accumulation, don't zero_grad on skip —
            # that would wipe accumulated gradients from prior micro-steps.
            # Just skip this micro-step's backward. zero_grad only happens
            # at optimizer step boundaries (line after scaler.step).
            if cfg.gradient_accumulation <= 1:
                self.optimizer.zero_grad()
            return _skip_result()

        # --- Update EMA (only for non-skipped steps) ---
        if self._loss_ema is None:
            self._loss_ema = loss_val
        else:
            self._loss_ema = (self._loss_ema_alpha * loss_val +
                              (1 - self._loss_ema_alpha) * self._loss_ema)

        # --- Backward pass ---
        # In DDP mode, backward triggers gradient all-reduce across ranks.
        # If one rank OOMs mid-backward, the all-reduce is partially done,
        # permanently desyncing NCCL. So we don't catch backward OOM in DDP —
        # the budget should be set low enough to prevent it.
        if loss is None:
            # Defensive: should never reach here (skip logic above handles it)
            # but prevents crash if DDP sync race condition lets it through
            self.optimizer.zero_grad()
            return _skip_result()

        # Scale loss by gradient accumulation steps so effective gradient
        # is the mean across micro-batches (not the sum).
        if cfg.gradient_accumulation > 1:
            loss = loss / cfg.gradient_accumulation

        # Ensure all parameters participate in the backward graph so DDP
        # can use find_unused_parameters=False.  The model conditionally
        # skips cls or reg encoder/decoder depending on task_type, which
        # would otherwise leave those params without gradients and trigger
        # a DDP hang (or a PyTorch >=2.10 is_pinned assertion failure with
        # find_unused_parameters=True).  Adding 0 * sum(p) creates autograd
        # edges without changing the gradient values.
        if cfg.distributed:
            raw_model = self.model.module if hasattr(self.model, 'module') else self.model
            loss = loss + 0.0 * sum(p.sum() for p in raw_model.parameters() if p.requires_grad)

        # Use no_sync() for intermediate gradient accumulation steps.
        # Without this, DDP triggers all_reduce on every backward() call,
        # and timing differences between ranks cause NCCL desync/timeout.
        next_accum_count = self.accumulated_micro_steps + 1
        is_accum_step = next_accum_count < cfg.gradient_accumulation
        use_no_sync = cfg.distributed and is_accum_step and cfg.gradient_accumulation > 1

        if cfg.distributed:
            ctx = self.model.no_sync() if use_no_sync else nullcontext()
            with ctx:
                self.scaler.scale(loss).backward()
        else:
            try:
                self.scaler.scale(loss).backward()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [SKIP] OOM in backward at step {self.global_step} "
                          f"(n={n_samples}, f={n_features})")
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    self.accumulated_micro_steps = 0
                    return _skip_result()
                raise

        self.accumulated_micro_steps = next_accum_count
        optimizer_stepped = False
        if self.accumulated_micro_steps >= cfg.gradient_accumulation:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)

            # Diagnostics (after unscaling, before clipping) — rank 0 only
            if cfg.log_interval > 0 and ((self.optimizer_step + 1) % cfg.log_interval == 0) and self.is_main:
                self._log_diagnostics(output)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self._update_ema()
            self.accumulated_micro_steps = 0
            self.optimizer_step += 1
            optimizer_stepped = True

        loss_dict['skipped'] = False
        loss_dict['lr'] = self.scheduler.get_last_lr()[0]
        loss_dict['task_type'] = task_type
        loss_dict['n_samples'] = n_samples
        loss_dict['n_features'] = n_features
        loss_dict['eval_pos'] = eval_pos
        loss_dict['optimizer_stepped'] = optimizer_stepped
        loss_dict['optimizer_step'] = self.optimizer_step

        return loss_dict

    def _get_bare_model(self):
        """Return the unwrapped model (handles DDP and torch.compile wrapping)."""
        model = self.model
        # Unwrap DDP
        if hasattr(model, 'module'):
            model = model.module
        # Unwrap torch.compile
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        return model

    def _clone_current_model_state(self):
        return {
            name: tensor.detach().clone()
            for name, tensor in self._get_bare_model().state_dict().items()
        }

    def _early_stop_info_path(self):
        return os.path.join(self.config.checkpoint_dir, "early_stop_info.json")

    def _resolve_early_stop_metric(self, mean_auc, mean_r2):
        metric_name = self.config.early_stop_metric
        if metric_name == "mean_auc":
            return float(mean_auc), "mean_auc"
        if metric_name == "mean_r2":
            return float(mean_r2), "mean_r2"

        values = []
        if np.isfinite(mean_auc):
            values.append(float(mean_auc))
        if np.isfinite(mean_r2):
            values.append(float(mean_r2))
        if not values:
            return float("nan"), "combined"
        return float(np.mean(values)), "combined"

    def _write_early_stop_info(self, metric_name, current_value):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        payload = {
            "triggered": True,
            "metric": metric_name,
            "best_value": self.best_early_stop_metric,
            "best_step": int(self.best_early_stop_step),
            "stopped_value": float(current_value),
            "stopped_step": int(self.optimizer_step),
            "bad_evals": int(self.early_stop_bad_evals),
            "patience_evals": int(self.config.early_stop_patience_evals),
            "min_delta": float(self.config.early_stop_min_delta),
            "min_evals": int(self.config.early_stop_min_evals),
            "best_checkpoint": "best_early_stop.pt",
        }
        with open(self._early_stop_info_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _update_early_stopping(self, mean_auc, mean_r2):
        cfg = self.config
        if cfg.early_stop_patience_evals <= 0:
            return

        self.early_stop_eval_count += 1
        metric_value, metric_name = self._resolve_early_stop_metric(mean_auc, mean_r2)

        if not np.isfinite(metric_value):
            print(
                f"  Early-stop monitor: {metric_name} is not finite; "
                "skipping patience update."
            )
            return

        improved = metric_value > (self.best_early_stop_metric + cfg.early_stop_min_delta)
        if improved:
            self.best_early_stop_metric = metric_value
            self.best_early_stop_step = self.optimizer_step
            self.early_stop_bad_evals = 0
            best_path = os.path.join(cfg.checkpoint_dir, "best_early_stop.pt")
            self.save_checkpoint(path=best_path)
            print(f"  Early-stop monitor: new best {metric_name}={metric_value:.4f}")
            return

        if self.early_stop_eval_count <= cfg.early_stop_min_evals:
            remaining = cfg.early_stop_min_evals - self.early_stop_eval_count
            print(
                f"  Early-stop warmup: {metric_name}={metric_value:.4f} | "
                f"patience starts after {max(remaining, 0)} more eval(s)"
            )
            return

        self.early_stop_bad_evals += 1
        print(
            f"  Early-stop monitor: {metric_name}={metric_value:.4f} | "
            f"no improvement ({self.early_stop_bad_evals}/{cfg.early_stop_patience_evals})"
        )
        if self.early_stop_bad_evals >= cfg.early_stop_patience_evals:
            self.should_stop_early = True
            self._write_early_stop_info(metric_name, metric_value)
            print(
                f"  Early stopping triggered on {metric_name}: "
                f"best={self.best_early_stop_metric:.4f} at step {self.best_early_stop_step}"
            )

    def _update_ema(self):
        if self.ema_state_dict is None:
            return

        with torch.no_grad():
            current_state = self._get_bare_model().state_dict()
            for name, tensor in current_state.items():
                ema_tensor = self.ema_state_dict[name]
                if torch.is_floating_point(tensor):
                    ema_tensor.mul_(self.ema_decay).add_(tensor.detach(), alpha=1.0 - self.ema_decay)
                else:
                    ema_tensor.copy_(tensor)

    @contextmanager
    def _swap_in_ema_weights(self):
        if self.ema_state_dict is None:
            yield
            return

        model = self._get_bare_model()
        current_state = self._clone_current_model_state()
        model.load_state_dict(self.ema_state_dict, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(current_state, strict=True)

    def save_checkpoint(self, path=None):
        """Save a training checkpoint (rank 0 only in DDP)."""
        if not self.is_main:
            return

        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_step_{self.global_step}.pt"
            )

        save_dict = {
            'model_state_dict': self._get_bare_model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'global_step': self.global_step,
            'optimizer_step': self.optimizer_step,
            'accumulated_micro_steps': self.accumulated_micro_steps,
            'best_loss': self.best_loss,
            'best_auc': self.best_auc,
            'best_r2': self.best_r2,
            'best_early_stop_metric': self.best_early_stop_metric,
            'best_early_stop_step': self.best_early_stop_step,
            'early_stop_bad_evals': self.early_stop_bad_evals,
            'early_stop_eval_count': self.early_stop_eval_count,
            'config': self.config,
        }
        if self.ema_state_dict is not None:
            save_dict['ema_state_dict'] = self.ema_state_dict
            save_dict['ema_decay'] = self.ema_decay
        if self.model_config is not None:
            save_dict['model_config'] = self.model_config
        torch.save(save_dict, path)
        print(f"Checkpoint saved to {path}")
        if self.on_checkpoint_saved is not None:
            self.on_checkpoint_saved()

    def load_checkpoint(self, path, model_only: bool = False):
        """Load a training checkpoint.

        After loading, re-applies config.lr to optimizer and scheduler to handle
        cases where LR changed (e.g., resuming single-GPU checkpoint with DDP
        which scales LR by world_size).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # strict=False allows new optional parameters (e.g. column_y_aware_alpha
        # added 2026-05-03 for V10) to be loaded from their fresh init when
        # FT'ing from older checkpoints. Missing/unexpected keys are logged so
        # the user notices unintended schema mismatches.
        missing, unexpected = self._get_bare_model().load_state_dict(
            ckpt['model_state_dict'], strict=False
        )
        if (missing or unexpected) and self.is_main:
            if missing:
                print(f"  [load_checkpoint] missing keys (using fresh init): {missing}")
            if unexpected:
                print(f"  [load_checkpoint] unexpected keys (ignored): {unexpected}")

        if model_only:
            # Fresh optimizer/scheduler state is required when a curriculum stage
            # intentionally restarts its LR schedule from the loaded weights.
            self.optimizer.zero_grad()
            self.global_step = 0
            self.optimizer_step = 0
            self.accumulated_micro_steps = 0
            self.best_loss = float('inf')
            self.best_auc = -float('inf')
            self.best_r2 = -float('inf')
            self.best_early_stop_metric = -float('inf')
            self.best_early_stop_step = 0
            self.early_stop_bad_evals = 0
            self.early_stop_eval_count = 0
            self.should_stop_early = False
            if self.ema_state_dict is not None:
                loaded_ema = ckpt.get('ema_state_dict')
                if loaded_ema is None:
                    self.ema_state_dict = self._clone_current_model_state()
                else:
                    # Merge: prefer loaded EMA values, but fall back to current
                    # model state for any key the older checkpoint didn't have
                    # (e.g., V10's column_y_aware_alpha when FT'ing from V8old).
                    current_state = self._clone_current_model_state()
                    for k in current_state:
                        if k in loaded_ema:
                            current_state[k] = loaded_ema[k]
                    self.ema_state_dict = current_state
            self._set_target_aware_scale(self._get_current_target_aware_scale())
            if self.is_main:
                print(
                    f'Checkpoint weights loaded from {path}; '
                    'optimizer, scheduler, scaler, and step counters reset.'
                )
            return

        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.global_step = ckpt['global_step']
        self.optimizer_step = ckpt.get('optimizer_step', max(self.scheduler.last_epoch, 0))
        loaded_accumulated_micro_steps = ckpt.get('accumulated_micro_steps', 0)
        # Checkpoints do not serialize in-flight parameter gradients, so resuming
        # mid accumulation would otherwise step with missing micro-batch grads.
        self.accumulated_micro_steps = 0
        self.best_loss = ckpt.get('best_loss', float('inf'))
        self.best_auc = ckpt.get('best_auc', -float('inf'))
        self.best_r2 = ckpt.get('best_r2', -float('inf'))
        self.best_early_stop_metric = ckpt.get('best_early_stop_metric', -float('inf'))
        self.best_early_stop_step = ckpt.get('best_early_stop_step', 0)
        self.early_stop_bad_evals = ckpt.get('early_stop_bad_evals', 0)
        self.early_stop_eval_count = ckpt.get('early_stop_eval_count', 0)
        self.should_stop_early = False
        if self.ema_state_dict is not None:
            loaded_ema = ckpt.get('ema_state_dict')
            if loaded_ema is None:
                self.ema_state_dict = self._clone_current_model_state()
            else:
                current_state = self._clone_current_model_state()
                for k in current_state:
                    if k in loaded_ema:
                        current_state[k] = loaded_ema[k]
                self.ema_state_dict = current_state
        self._set_target_aware_scale(self._get_current_target_aware_scale())

        # Re-apply current config LR (handles DDP LR scaling on resume).
        # load_state_dict overwrites param_groups with checkpoint LR;
        # we need to set it to the current effective LR.
        new_lr = self.config.lr
        self.optimizer.defaults['lr'] = new_lr
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr
            pg['initial_lr'] = new_lr
        self.scheduler.base_lrs = [new_lr] * len(self.scheduler.base_lrs)
        self.optimizer.zero_grad()

        if self.is_main:
            if loaded_accumulated_micro_steps:
                print(
                    'Checkpoint had '
                    f'accumulated_micro_steps={loaded_accumulated_micro_steps}; '
                    'reset to 0 because accumulated gradients are not serialized.'
                )
            print(f'Checkpoint loaded from {path}, step={self.global_step}, '
                  f'lr={new_lr:.2e}')

    def _run_validation(self):
        """Run TabArena validation (rank 0 only). Other ranks wait at barrier."""
        cfg = self.config
        if not cfg.eval_enabled:
            return

        # Only rank 0 runs eval
        if self.is_main:
            opt_step = self.optimizer_step
            current_feature_loss_weight = self._get_current_feature_loss_weight()
            current_tae_scale = self._get_current_target_aware_scale()
            self._set_target_aware_scale(current_tae_scale)
            print(f"\n--- Validation at step {opt_step} ---")
            try:
                from synthefy_tabular.training.evaluator import run_full_evaluation
                self.model.eval()
                torch.cuda.empty_cache()
                if self.ema_state_dict is not None:
                    print(f"  Using EMA weights for validation (decay={self.ema_decay})")
                with self._swap_in_ema_weights():
                    with torch.no_grad():
                        eval_results = run_full_evaluation(
                            model=self.model,
                            device=self.device,
                            cls_data_dir=cfg.eval_cls_data_dir,
                            reg_data_dir=cfg.eval_reg_data_dir,
                            cls_config_path=cfg.eval_cls_config,
                            reg_config_path=cfg.eval_reg_config,
                        )
                self.model.train()

                mean_auc = eval_results['mean_auc']
                mean_r2 = eval_results['mean_r2']
                elapsed = eval_results['elapsed_seconds']
                n_cls = len(eval_results['cls_datasets'])
                n_reg = len(eval_results['reg_datasets'])

                print(f"  mean_auc={mean_auc:.4f} ({n_cls} datasets) | "
                      f"mean_r2={mean_r2:.4f} ({n_reg} datasets) | "
                      f"{elapsed:.1f}s | "
                      f"feat_w={current_feature_loss_weight:.3f} "
                      f"tae={current_tae_scale:.3f}")
                # Per-dataset R2 for tracking gap closure
                if eval_results['reg_datasets']:
                    # Group datasets by subset: tabarena-13 vs hard (our tracked
                    # failing datasets from TabPFN-2.6 and LimiX-2M).
                    TABARENA_13 = {
                        'Another-Dataset-on-used-Fiat-500', 'Food_Delivery_Time',
                        'QSAR-TID-11', 'QSAR_fish_toxicity', 'airfoil_self_noise',
                        'concrete_compressive_strength', 'diamonds',
                        'healthcare_insurance_expenses', 'houses', 'miami_housing',
                        'physiochemical_protein', 'superconductivity', 'wine_quality',
                    }
                    tabarena_scores = []
                    hard_scores = []
                    for name, m in eval_results['reg_datasets'].items():
                        r2 = m.get('r2', float('nan'))
                        if not np.isfinite(r2):
                            continue
                        if name in TABARENA_13:
                            tabarena_scores.append(r2)
                        else:
                            hard_scores.append(r2)

                    # Subset means line
                    mean_parts = []
                    if tabarena_scores:
                        mean_parts.append(
                            f"tabarena_r2={np.mean(tabarena_scores):.4f} "
                            f"({len(tabarena_scores)} datasets)"
                        )
                    if hard_scores:
                        mean_parts.append(
                            f"hard_r2={np.mean(hard_scores):.4f} "
                            f"({len(hard_scores)} datasets)"
                        )
                    if mean_parts:
                        print(f"  reg_means: {' | '.join(mean_parts)}")

                    # Per-dataset vertical list, alphabetically sorted
                    print("  reg_r2:")
                    for name in sorted(eval_results['reg_datasets'].keys()):
                        m = eval_results['reg_datasets'][name]
                        print(f"    {name}: {m.get('r2', float('nan')):.4f}")
                # Log to wandb
                if cfg.use_wandb and self.wandb_run:
                    import wandb
                    log_dict = {
                        'val/mean_auc': mean_auc,
                        'val/mean_r2': mean_r2,
                        'val/eval_seconds': elapsed,
                        'val/feature_loss_weight': current_feature_loss_weight,
                        'val/target_aware_scale': current_tae_scale,
                    }
                    for name, m in eval_results['cls_datasets'].items():
                        log_dict[f'val_cls/{name}_auc'] = m['auc']
                    for name, m in eval_results['reg_datasets'].items():
                        log_dict[f'val_reg/{name}_r2'] = m['r2']
                    wandb.log(log_dict, step=opt_step)

                # Save best checkpoints
                if np.isfinite(mean_auc) and mean_auc > self.best_auc:
                    self.best_auc = mean_auc
                    best_path = os.path.join(cfg.checkpoint_dir, "best_cls_auc.pt")
                    self.save_checkpoint(path=best_path)
                    print(f"  New best AUC: {mean_auc:.4f}")

                if np.isfinite(mean_r2) and mean_r2 > self.best_r2:
                    self.best_r2 = mean_r2
                    best_path = os.path.join(cfg.checkpoint_dir, "best_reg_r2.pt")
                    self.save_checkpoint(path=best_path)
                    print(f"  New best R2: {mean_r2:.4f}")

                self._update_early_stopping(mean_auc, mean_r2)

            except Exception as e:
                print(f"  [EVAL] Validation failed: {e}")
                self.model.train()

            print(f"--- End validation ---\n")

        if cfg.distributed:
            stop_tensor = torch.tensor(
                [1 if self.should_stop_early else 0], device=self.device)
            torch.distributed.broadcast(stop_tensor, src=0)
            self.should_stop_early = stop_tensor.item() > 0.5

    def train(self):
        """Main training loop."""
        cfg = self.config

        if cfg.use_wandb and self.is_main:
            try:
                import wandb
                wandb_kwargs = {
                    "project": cfg.wandb_project,
                    "config": vars(cfg),
                }
                if cfg.wandb_entity:
                    wandb_kwargs["entity"] = cfg.wandb_entity
                if cfg.wandb_name:
                    wandb_kwargs["name"] = cfg.wandb_name
                if cfg.wandb_group:
                    wandb_kwargs["group"] = cfg.wandb_group
                if cfg.wandb_job_type:
                    wandb_kwargs["job_type"] = cfg.wandb_job_type
                if cfg.wandb_tags:
                    wandb_kwargs["tags"] = list(cfg.wandb_tags)
                self.wandb_run = wandb.init(
                    **wandb_kwargs,
                )
            except ImportError:
                print("wandb not installed, disabling logging")
                cfg.use_wandb = False
            except Exception as e:
                print(f"wandb init failed ({e}), disabling logging")
                cfg.use_wandb = False

        # Gradient accumulation: all user-facing params (total_steps, warmup,
        # decay, save/eval/log intervals) are in OPTIMIZER STEP units.
        # Accumulation windows are counted by successful micro-batches only, so
        # skipped micro-steps don't silently change the effective batch size.
        accum = cfg.gradient_accumulation
        start_optimizer_step = self.optimizer_step
        if cfg.run_steps is not None:
            target_optimizer_step = min(
                cfg.total_steps,
                start_optimizer_step + cfg.run_steps,
            )
        else:
            target_optimizer_step = cfg.total_steps

        if self.is_main:
            print(f"Starting training for {cfg.total_steps} optimizer steps")
            if cfg.run_steps is not None:
                planned_steps = max(target_optimizer_step - start_optimizer_step, 0)
                print(
                    f"  This invocation: {planned_steps} optimizer steps "
                    f"(from {start_optimizer_step} to {target_optimizer_step})"
                )
            if accum > 1:
                print(f"  Gradient accumulation: {accum} successful micro-steps per optimizer step")
            print(f"  Device: {cfg.device}")
            print(f"  Batch size: {cfg.batch_size}")
            if cfg.distributed:
                print(f"  World size: {cfg.world_size}")
                eff_batch = cfg.batch_size * cfg.world_size * accum
                print(f"  Effective batch size: {eff_batch}")
            print(f"  Learning rate: {cfg.lr}")
            if (cfg.feature_loss_weight_end is not None
                    and cfg.feature_loss_decay_end_step > cfg.feature_loss_decay_start_step):
                print(
                    f"  Feature loss schedule: {cfg.feature_loss_weight:.3f} -> "
                    f"{cfg.feature_loss_weight_end:.3f} "
                    f"(steps {cfg.feature_loss_decay_start_step}.."
                    f"{cfg.feature_loss_decay_end_step})"
                )
            else:
                print(f"  Feature loss weight: {cfg.feature_loss_weight:.3f}")
            if cfg.target_aware_warmup_steps > 0:
                print(
                    f"  Target-aware warmup: scale {cfg.target_aware_init_scale:.3f} -> 1.000 "
                    f"over {cfg.target_aware_warmup_steps} steps"
                )
            print(f"  Mixed precision: {cfg.mixed_precision}")
            if cfg.compile:
                print(f"  torch.compile: enabled")
            if self.prefetcher is not None:
                print(f"  Async prefetch: {cfg.prefetch_workers} workers, "
                      f"{cfg.prefetch_count} prefetch")

        if target_optimizer_step <= start_optimizer_step:
            if self.is_main:
                print(
                    f"No training steps to run: optimizer_step={start_optimizer_step}, "
                    f"target={target_optimizer_step}."
                )
            if cfg.use_wandb and self.wandb_run:
                import wandb
                wandb.finish()
            return

        # --- Start prefetcher and pre-fill pipeline ---
        if self.prefetcher is not None and self.optimizer_step < target_optimizer_step:
            self.prefetcher.start()
            self._prefetch_params_queue = []
            for _ in range(cfg.prefetch_count):
                params = self._sample_data_params()
                n_samples, n_features, task_type, n_classes, context_ratio = params
                self._prefetch_params_queue.append(params)
                self._submit_prefetch(n_samples, n_features, task_type, n_classes)
        else:
            self._prefetch_params_queue = []

        running_loss = 0.0
        running_y_loss = 0.0
        running_feat_loss = 0.0
        running_cls_y_loss = 0.0
        running_reg_y_loss = 0.0
        cls_count = 0
        reg_count = 0
        log_count = 0
        start_time = time.time()

        self.optimizer.zero_grad()

        if self.global_step == 0 and self.optimizer_step == 0:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
                self.scheduler.step()  # Initialize scheduler at step 0

        first_log_pending = self.global_step == 0 and self.optimizer_step == 0

        # Optional: eval at step 0 before any training, to establish baseline
        # on the current checkpoint (useful for fine-tune runs to see what
        # the seed scores on the eval dataset without any perturbation).
        if (getattr(cfg, 'eval_at_step_0', False)
                and cfg.eval_enabled
                and self.optimizer_step == 0):
            if self.is_main:
                print("Running step-0 baseline eval (before training)...")
            self._run_validation()

        while self.optimizer_step < target_optimizer_step:

            # train_step handles all errors internally (OOM, ValueError, NaN,
            # spike) and syncs skip decisions across DDP ranks. Never raises
            # except for truly fatal errors.
            loss_dict = self.train_step()
            self.global_step += 1

            # Submit next prefetch batch (keep pipeline full).
            if self.prefetcher is not None and self.optimizer_step < target_optimizer_step:
                params = self._sample_data_params()
                ns, nf, tt, nc, cr = params
                self._prefetch_params_queue.append(params)
                self._submit_prefetch(ns, nf, tt, nc)

            if not loss_dict.get('skipped', False):
                running_loss += loss_dict['total_loss']
                running_y_loss += loss_dict['y_loss']
                running_feat_loss += loss_dict['feat_loss']
                log_count += 1
                # Per-task-type loss tracking
                if loss_dict.get('task_type') == 'cls':
                    running_cls_y_loss += loss_dict['y_loss']
                    cls_count += 1
                elif loss_dict.get('task_type') == 'reg':
                    running_reg_y_loss += loss_dict['y_loss']
                    reg_count += 1

            stepped = loss_dict.get('optimizer_stepped', False)
            should_log = False
            if log_count > 0 and self.is_main:
                if first_log_pending:
                    should_log = True
                    first_log_pending = False
                elif stepped and cfg.log_interval > 0 and self.optimizer_step % cfg.log_interval == 0:
                    should_log = True
            if should_log and log_count > 0 and self.is_main:
                elapsed = time.time() - start_time
                avg_loss = running_loss / log_count
                avg_y = running_y_loss / log_count
                avg_feat = running_feat_loss / log_count
                n_steps_elapsed = log_count
                steps_per_sec = n_steps_elapsed / elapsed
                opt_step = self.optimizer_step

                print(
                    f"Step {opt_step}/{cfg.total_steps} | "
                    f"loss={avg_loss:.4f} y={avg_y:.4f} feat={avg_feat:.4f} | "
                    f"lr={loss_dict['lr']:.2e} | "
                    f"feat_w={loss_dict.get('feature_loss_weight', cfg.feature_loss_weight):.3f} "
                    f"tae={loss_dict.get('target_aware_scale', 1.0):.3f} | "
                    f"{steps_per_sec:.1f} steps/s | "
                    f"task={loss_dict['task_type']} "
                    f"n={loss_dict['n_samples']} f={loss_dict['n_features']}"
                )

                if cfg.use_wandb and self.wandb_run:
                    import wandb
                    log_dict = {
                        'train/loss': avg_loss,
                        'train/y_loss': avg_y,
                        'train/feat_loss': avg_feat,
                        'train/lr': loss_dict['lr'],
                        'train/feature_loss_weight': loss_dict.get(
                            'feature_loss_weight', cfg.feature_loss_weight),
                        'train/target_aware_scale': loss_dict.get(
                            'target_aware_scale', 1.0),
                        'train/steps_per_sec': steps_per_sec,
                        'data/n_samples': loss_dict['n_samples'],
                        'data/n_features': loss_dict['n_features'],
                    }
                    if cls_count > 0:
                        log_dict['train/cls_y_loss'] = running_cls_y_loss / cls_count
                    if reg_count > 0:
                        log_dict['train/reg_y_loss'] = running_reg_y_loss / reg_count
                    wandb.log(log_dict, step=opt_step)

                running_loss = 0.0
                running_y_loss = 0.0
                running_feat_loss = 0.0
                running_cls_y_loss = 0.0
                running_reg_y_loss = 0.0
                cls_count = 0
                reg_count = 0
                log_count = 0
                start_time = time.time()

            # Checkpoint saving (on optimizer step boundaries)
            if stepped and cfg.save_interval > 0 and self.optimizer_step % cfg.save_interval == 0:
                self.save_checkpoint()

            # Validation (on optimizer step boundaries)
            if (cfg.eval_enabled
                    and cfg.eval_interval > 0
                    and stepped
                    and self.optimizer_step % cfg.eval_interval == 0):
                self._run_validation()
                if self.should_stop_early:
                    if self.is_main:
                        print(
                            f"Stopping early at optimizer step {self.optimizer_step} "
                            f"after validation plateau."
                        )
                    break

        # Final checkpoint
        self.save_checkpoint()

        # Shutdown prefetcher
        if self.prefetcher is not None:
            self.prefetcher.shutdown()

        if cfg.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()

        if self.is_main:
            if self.should_stop_early:
                print(
                    f"Training stopped early at optimizer step "
                    f"{self.optimizer_step}/{cfg.total_steps}."
                )
            elif self.optimizer_step >= cfg.total_steps:
                print("Training complete!")
            else:
                print(
                    f"Training segment complete at optimizer step "
                    f"{self.optimizer_step}/{cfg.total_steps}."
                )
