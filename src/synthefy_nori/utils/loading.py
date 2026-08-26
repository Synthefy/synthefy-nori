from __future__ import annotations

import os
import pickle

import torch
import random
import numpy as np
from synthefy_nori.model.transformer import FeaturesTransformer


def _safe_torch_load(path):
    """Load a checkpoint without executing code embedded in it.

    ``weights_only=True`` uses torch's restricted unpickler — it reconstructs
    only tensors and plain data (dicts/lists/primitives), never arbitrary
    classes or callables — so a malicious checkpoint cannot run code on load.
    The public checkpoint is slim (``model_config`` dict + state-dict tensors)
    and loads under this restriction with no allowlist.

    Raw *training* checkpoints additionally pickle our own ``TrainingConfig``
    dataclass. That single class is safe to reconstruct (it has no custom
    ``__reduce__``/``__setstate__``, so unpickling is plain attribute
    assignment), so we allowlist exactly it and retry. We never fall back to the
    unsafe full unpickler, so anything else (e.g. ``os.system``) stays blocked.
    See SECURITY.md.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        from synthefy_nori.training.config import TrainingConfig

        with torch.serialization.safe_globals([TrainingConfig]):
            return torch.load(path, map_location="cpu", weights_only=True)


# The attention-scale flags and their defaults. ``finalize_arch_config`` stamps
# this table into every new checkpoint's ``model_config``; ``build_model`` falls
# back to it for checkpoints saved before it did. One table, so the value a run
# trains with and the value a later load assumes cannot drift apart.
_ATTENTION_SCALE_DEFAULTS = {
    "use_logn_attention": False,
    "use_learnable_attn_temperature": False,
    "attn_n_ref": 1024.0,
}


def resolve_qass_mode(config: dict) -> str:
    """Return the QASS attention mode an *architecture* config describes.

    ``config`` must be the architecture dict -- the same one ``build_model``
    consumes. Never a ``TrainingConfig``: that object carries no architecture,
    and reading it here would make the resolved mode depend on the checkpoint's
    container format rather than on the model.

    An explicit ``qass_mode`` is honored verbatim. Anything else resolves to
    "full", because "full" is the only mode this tree has ever trained:
    ``QASSMaxScaling`` had no mode switch until now, so every checkpoint written
    without a ``qass_mode`` ran the full ``base * gate`` path and carries trained
    base/gate weights. Resolving such a checkpoint to anything else would discard
    those weights and silently apply the wrong attention temperature.

    So this is a widening, not a change: nothing that loads today resolves
    differently, and a checkpoint that *does* record a mode -- the released
    ``Nori-30M`` already carries ``qass_mode: "full"`` -- is now honored instead
    of ignored. Going forward ``finalize_arch_config`` pins the mode before
    training, so new checkpoints never reach the fallback at all.
    """
    explicit_mode = config.get("qass_mode")
    if explicit_mode is not None:
        return str(explicit_mode)
    return "full"


def finalize_arch_config(model_config: dict) -> dict:
    """Complete ``model_config`` in place so it fully describes the architecture.

    ``model_config`` is the single source of truth for architecture: it is what
    ``build_model`` consumes at training time and what the trainer embeds in
    every checkpoint. So anything ``build_model`` reads has to be written here
    explicitly -- otherwise a later load re-derives it, and a re-derivation is
    only ever as good as the guess behind it.

    Call this once, after every architecture override has been applied and
    before ``build_model``.

    ``qass_mode`` is pinned before the attention-scale keys are defaulted. The
    order does not change the outcome here, but it keeps this function honest:
    the mode is derived from the config the caller actually assembled, never
    from a value this function just invented.

    ``SYNTHEFY_QASS_MODE`` is honoured here and *only* here: this is the single
    point where the environment can influence the architecture, and it does so
    by being written into ``model_config``. Everything downstream --
    ``build_model`` and the ``QASSMaxScaling`` modules it constructs -- reads the
    config, never the environment. So an override still works end to end, the
    checkpoint records the mode the run actually trained with, and there is no
    second channel that can disagree with the first.
    """
    if bool(model_config.get("use_qassmax", False)):
        env_mode = os.environ.get("SYNTHEFY_QASS_MODE")
        model_config["qass_mode"] = env_mode.strip().lower() if env_mode else resolve_qass_mode(model_config)
    for key, default in _ATTENTION_SCALE_DEFAULTS.items():
        model_config.setdefault(key, default)
    return model_config


def build_model(config: dict):
    # Pre-``finalize_arch_config`` checkpoints omit these; fall back to the same
    # table finalize stamps in, so old and new checkpoints agree.
    attn_scale = {k: config.get(k, v) for k, v in _ATTENTION_SCALE_DEFAULTS.items()}
    use_qassmax = bool(config.get("use_qassmax", False))
    model = FeaturesTransformer(
        preprocess_config_x=config["preprocess_config_x"],
        encoder_config_x=config["encoder_config_x"],
        encoder_config_y=config["encoder_config_y"],
        decoder_config=config["decoder_config"],
        feature_positional_embedding_type=config.get("feature_positional_embedding_type", "subortho"),
        feature_positional_embedding_num_slots=config.get("feature_positional_embedding_num_slots", 1000),
        nlayers=config["nlayers"],
        nhead=config["nhead"],
        embed_dim=config["embed_dim"],
        hid_dim=config["hid_dim"],
        mask_prediction=config.get("mask_prediction", False),
        features_per_group=config["features_per_group"],
        dropout=config["dropout"],
        pre_norm=config.get("pre_norm", True),
        activation=config.get("activation", "gelu"),
        layer_norm_eps=config.get("layer_norm_eps", 1e-5),
        device=config.get("device", None),
        dtype=config.get("dtype", None),
        recompute_attn=config["recompute_attn"],
        # Kept as a fail-fast compatibility input: True was historically a
        # silent no-op and must not continue pretending to select an architecture.
        mlp_use_residual=config.get("mlp_use_residual", False),
        layer_arch=config.get("layer_arch", "fmfmsm"),
        norm_type=config.get("norm_type", "layernorm"),
        deepnorm_alpha=config.get("deepnorm_alpha", None),
        self_share_all_kv_heads=config.get("self_share_all_kv_heads", False),
        cross_share_all_kv_heads=config.get("cross_share_all_kv_heads", True),
        seq_attn_isolated=config.get("seq_attn_isolated", False),
        seq_attn_serial=config.get("seq_attn_serial", False),
        use_qassmax=use_qassmax,
        # Resolved here, once, from this config — not read back out of the
        # environment inside QASSMaxScaling.
        qass_mode=resolve_qass_mode(config) if use_qassmax else None,
        use_target_aware_embedding=config.get("use_target_aware_embedding", False),
        use_column_specific_y_aware=config.get("use_column_specific_y_aware", False),
        use_logn_attention=bool(attn_scale["use_logn_attention"]),
        use_learnable_attn_temperature=bool(attn_scale["use_learnable_attn_temperature"]),
        attn_n_ref=float(attn_scale["attn_n_ref"]),
        # Legacy configs omit this flag and therefore retain the decoder. That
        # preserves their state-dict schema for strict checkpoint loading.
        omit_feature_decoder=bool(config.get("omit_feature_decoder", False)),
    )
    return model


def load_model(model_path, mask_prediction: bool = False, base_config_path: str = None, native_rms_norm: bool = True):
    """Load a Nori checkpoint for inference.

    ``native_rms_norm`` selects PyTorch's fused ``F.rms_norm`` over the
    decomposed pow/mean/rsqrt/mul chain. It defaults to True because every
    inference and evaluation path reaches the model through this function, so
    this is the one place that turns the kernel on everywhere. Measured on
    1k/4k/8k-row tables: R2 shift <= 2e-5, largest per-row difference one bf16
    ulp under autocast. Pass False to reproduce the historical path bit-for-bit.

    Training does NOT come through here (the trainer builds via build_model),
    so this default cannot silently change a training run.
    """
    state_dict = _safe_torch_load(model_path)

    # Support both pretrained (.ckpt) and training checkpoint (.pt) formats
    if "model_config" in state_dict:
        # Training checkpoint format (new): has architecture config embedded.
        # `model_config` is the architecture record -- read it and nothing else.
        # `state_dict['config']` is the TrainingConfig (hyperparameters, no
        # architecture); merging it in here would make the resolved QASS mode a
        # function of the container format instead of the model.
        config = state_dict["model_config"]
        weights = state_dict.get("ema_state_dict") or state_dict["model_state_dict"]
    elif "model_state_dict" in state_dict:
        # Training checkpoint format (legacy): no model_config saved
        # Fall back to extracting config from base pretrained checkpoint
        if base_config_path is None:
            raise KeyError(
                f"Training checkpoint at {model_path} does not contain 'model_config'. "
                "Provide base_config_path pointing to the pretrained .ckpt file."
            )
        print(f"Loading model architecture config from {base_config_path}")
        base_state = _safe_torch_load(base_config_path)
        config = base_state["config"]
        weights = state_dict.get("ema_state_dict") or state_dict["model_state_dict"]
    else:
        # Pretrained .ckpt format: {'state_dict': ..., 'config': ...}
        config = state_dict["config"]
        weights = state_dict["state_dict"]

    # Copy before mutating: `config` is a dict embedded in the loaded checkpoint,
    # and callers (e.g. the eval harness) reuse that object.
    config = dict(config)
    if mask_prediction and bool(config.get("omit_feature_decoder", False)):
        raise ValueError(
            "mask_prediction=True requires a checkpoint with feature_decoder; "
            "this checkpoint explicitly omits that head"
        )
    config["mask_prediction"] = mask_prediction

    # Strip torch.compile "_orig_mod." prefix if present
    if any(k.startswith("_orig_mod.") for k in weights):
        weights = {k.removeprefix("_orig_mod."): v for k, v in weights.items()}

    model = build_model(config)
    model.load_state_dict(weights)

    if native_rms_norm:
        from synthefy_nori.model.layer import RMSNorm

        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.use_native = True

    model.eval()
    return model
