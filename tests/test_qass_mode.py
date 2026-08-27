"""Regression tests for QASS attention-mode handling.

Guards the migration bug where the new repo dropped the ``qass_mode`` switch and
silently ran older "log_only" checkpoints (which still carry trained base/gate
weights) with the gate ON — applying the wrong attention temperature.
"""

import math

import torch

from synthefy_nori.model.layer import QASSMaxScaling
from synthefy_nori.utils.loading import resolve_qass_mode


def _poison(module: QASSMaxScaling) -> None:
    """Make every learned QASS component non-trivial, so a mode that is supposed
    to ignore them will visibly diverge if it doesn't."""
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.5)


def test_log_only_ignores_base_and_gate(monkeypatch):
    monkeypatch.setenv("SYNTHEFY_QASS_MODE", "log_only")
    m = QASSMaxScaling(num_heads=2, head_dim=4)
    _poison(m)  # trained weights present, but log_only must ignore them
    q = torch.randn(1, 5, 2, 4)
    out = m(q, key_len=100)
    expected = q * math.log(100.0)
    assert torch.allclose(out, expected, atol=1e-5)


def test_full_uses_gate(monkeypatch):
    monkeypatch.setenv("SYNTHEFY_QASS_MODE", "full")
    m = QASSMaxScaling(num_heads=2, head_dim=4)
    _poison(m)
    q = torch.randn(1, 5, 2, 4)
    out = m(q, key_len=100)
    # full mode applies base+gate, so it must differ from pure log(n) scaling
    assert not torch.allclose(out, q * math.log(100.0), atol=1e-4)


def test_base_only_ignores_gate(monkeypatch):
    monkeypatch.setenv("SYNTHEFY_QASS_MODE", "base_only")
    m = QASSMaxScaling(num_heads=2, head_dim=4)
    _poison(m)
    q = torch.randn(1, 5, 2, 4)
    out = m(q, key_len=64)
    log_n = math.log(64.0)
    base_delta = m.base_mlp(torch.tensor([[log_n]])).view(1, 1, 2, 4)
    base_scale = log_n * (1.0 + torch.tanh(base_delta))
    assert torch.allclose(out, q * base_scale, atol=1e-5)


def test_invalid_mode_raises(monkeypatch):
    monkeypatch.setenv("SYNTHEFY_QASS_MODE", "bogus")
    try:
        QASSMaxScaling(num_heads=2, head_dim=4)
    except ValueError as e:
        assert "SYNTHEFY_QASS_MODE" in str(e)
    else:
        raise AssertionError("invalid qass_mode should raise ValueError")


def test_explicit_qass_mode_beats_env(monkeypatch):
    """A config-resolved mode must not be overridable by process-global state.

    `build_model` resolves the mode from the architecture config and passes it
    down explicitly. If the env could still win, a stray SYNTHEFY_QASS_MODE in
    someone's shell would change what a checkpoint computes.
    """
    monkeypatch.setenv("SYNTHEFY_QASS_MODE", "full")
    m = QASSMaxScaling(num_heads=2, head_dim=4, qass_mode="log_only")
    assert m.qass_mode == "log_only"
    _poison(m)
    q = torch.randn(1, 5, 2, 4)
    assert torch.allclose(m(q, key_len=100), q * math.log(100.0), atol=1e-5)


def test_qass_mode_falls_back_to_env_then_full(monkeypatch):
    """Back-compat: no explicit mode -> env -> "full"."""
    monkeypatch.setenv("SYNTHEFY_QASS_MODE", "base_only")
    assert QASSMaxScaling(num_heads=2, head_dim=4).qass_mode == "base_only"

    monkeypatch.delenv("SYNTHEFY_QASS_MODE", raising=False)
    assert QASSMaxScaling(num_heads=2, head_dim=4).qass_mode == "full"


def _tiny_arch_config(**overrides) -> dict:
    """A buildable 2-layer architecture config, from the bundled base."""
    import json

    from synthefy_nori.training.config import package_config_path

    with open(package_config_path("model_base.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    config.update(nlayers=2, **overrides)
    return config


def _qass_modes(model) -> set:
    return {m.qass_mode for m in model.modules() if type(m).__name__ == "QASSMaxScaling"}


def _build(config):
    from synthefy_nori.utils.loading import build_model

    return build_model(config)


def test_built_model_mode_follows_config_not_env(monkeypatch):
    """End-to-end: the built model's attention temperature comes from its own
    config, whatever the environment says."""
    monkeypatch.setenv("SYNTHEFY_QASS_MODE", "full")
    model = _build(_tiny_arch_config(use_qassmax=True, qass_mode="log_only"))
    assert _qass_modes(model) == {"log_only"}


def test_resolve_qass_mode_three_eras():
    # Era 1: explicit qass_mode honored verbatim.
    assert resolve_qass_mode({"use_qassmax": True, "qass_mode": "full"}) == "full"

    # Era 2: qass_mode unset, use_logn_attention explicitly False -> "full".
    assert resolve_qass_mode({"use_qassmax": True, "use_logn_attention": False}) == "full"

    # Era 3: both unset (early-V13 schema, e.g. the 3M) -> "log_only".
    assert resolve_qass_mode({"use_qassmax": True}) == "log_only"


def test_no_qassmax_means_no_mode_is_passed(monkeypatch):
    """use_qassmax=False builds no QASSMaxScaling, so no mode is resolved."""
    monkeypatch.setenv("SYNTHEFY_QASS_MODE", "log_only")
    config = _tiny_arch_config(use_qassmax=False)
    model = _build(config)
    assert not [m for m in model.modules() if type(m).__name__ == "QASSMaxScaling"]


# ---------------------------------------------------------------------------
# One config. Architecture lives in `model_config` and nowhere else.
#
# A training `.pt` also stores `config` (the TrainingConfig). That object holds
# hyperparameters, not architecture -- the trainer builds from `model_config`
# alone -- so `load_model` must never read it for arch flags. It once looked
# like a second, fuller record of the architecture; reading it made the resolved
# QASS mode a function of which container the checkpoint sat in.
#
# Going forward `finalize_arch_config` pins every arch flag before training, so
# the era-2/era-3 inference in `resolve_qass_mode` only ever sees old files.
# ---------------------------------------------------------------------------


def _resolved_mode_for(state_dict, monkeypatch, mask_prediction=False):
    """Run load_model's config selection and return the QASS mode it resolves."""
    from synthefy_nori.utils import loading

    monkeypatch.delenv("SYNTHEFY_QASS_MODE", raising=False)
    monkeypatch.setattr(loading, "_safe_torch_load", lambda _p: state_dict)
    captured = {}

    def fake_build_model(config):
        captured["config"] = config
        captured["mode"] = loading.resolve_qass_mode(config)
        raise _StopBuild

    monkeypatch.setattr(loading, "build_model", fake_build_model)
    try:
        loading.load_model("ignored.pt", mask_prediction=mask_prediction)
    except _StopBuild:
        pass
    return captured.get("mode"), captured["config"]


class _StopBuild(Exception):
    """Abort load_model once build_model has seen the config."""


# An early-V13 model_config: no qass_mode, no use_logn_attention -> era 3.
_MODEL_CONFIG = {"use_qassmax": True, "nlayers": 2, "embed_dim": 8}


def test_same_arch_config_resolves_the_same_in_both_containers(monkeypatch):
    """The mode follows the architecture dict, not the container it sits in."""
    arch = {**_MODEL_CONFIG, "qass_mode": "base_only"}
    pt = {"model_config": dict(arch), "config": {}, "model_state_dict": {}}
    ckpt = {"config": dict(arch), "state_dict": {}}

    pt_mode, _ = _resolved_mode_for(pt, monkeypatch)
    ckpt_mode, _ = _resolved_mode_for(ckpt, monkeypatch)

    assert pt_mode == ckpt_mode == "base_only", (
        f"container format changed the attention temperature: .pt -> {pt_mode!r}, .ckpt -> {ckpt_mode!r}"
    )


def test_training_config_never_supplies_architecture(monkeypatch):
    """A `.pt`'s TrainingConfig must not reach the architecture.

    `TrainingConfig` used to declare `use_logn_attention: bool = False` -- an
    unread default, not a record of the run. Letting it through pushed an era-3
    checkpoint into era 2 and silently ran "full" where training ran "log_only".
    """
    pt = {
        "model_config": dict(_MODEL_CONFIG),
        "config": {"use_logn_attention": False, "attn_n_ref": 4096.0, "lr": 1e-4},
        "model_state_dict": {},
    }
    mode, config = _resolved_mode_for(pt, monkeypatch)
    assert mode == "log_only"
    assert config.get("attn_n_ref") != 4096.0


def test_load_model_does_not_mutate_the_checkpoint(monkeypatch):
    """`mask_prediction` must not be written into the loaded checkpoint's dict."""
    model_config = dict(_MODEL_CONFIG)
    pt = {"model_config": model_config, "config": {}, "model_state_dict": {}}
    _resolved_mode_for(pt, monkeypatch, mask_prediction=True)
    assert "mask_prediction" not in model_config


# ---------------------------------------------------------------------------
# finalize_arch_config: the write side. Every new checkpoint lands in era 1.
# ---------------------------------------------------------------------------


def test_finalize_pins_the_era_3_mode_not_the_defaulted_one(monkeypatch):
    """Pinning must happen before the attention-scale keys are defaulted.

    Writing `use_logn_attention=False` first would move an era-3 config into
    era 2 and flip the pinned mode from "log_only" to "full".
    """
    from synthefy_nori.utils.loading import finalize_arch_config

    monkeypatch.delenv("SYNTHEFY_QASS_MODE", raising=False)
    config = finalize_arch_config(dict(_MODEL_CONFIG))
    assert config["qass_mode"] == "log_only"
    assert config["use_logn_attention"] is False
    assert config["attn_n_ref"] == 1024.0


def test_finalized_config_round_trips_to_the_same_mode(monkeypatch):
    """Reloading a finalized config reproduces the mode training used."""
    from synthefy_nori.utils.loading import finalize_arch_config

    monkeypatch.delenv("SYNTHEFY_QASS_MODE", raising=False)
    config = finalize_arch_config(dict(_MODEL_CONFIG))
    pt = {"model_config": config, "config": {}, "model_state_dict": {}}
    mode, _ = _resolved_mode_for(pt, monkeypatch)
    assert mode == "log_only"


def test_finalize_records_the_env_override(monkeypatch):
    """finalize is the ONE place the env can influence the architecture.

    It does so by writing the override into `model_config`, which `build_model`
    then reads — so the override still works end to end, and the checkpoint
    records the mode the run actually trained with. Nothing downstream reads
    the environment, so there is no second channel to disagree with this one.
    """
    from synthefy_nori.utils.loading import finalize_arch_config

    monkeypatch.setenv("SYNTHEFY_QASS_MODE", "Full")
    config = finalize_arch_config(dict(_MODEL_CONFIG))
    assert config["qass_mode"] == "full"


def test_finalize_keeps_an_explicit_mode(monkeypatch):
    from synthefy_nori.utils.loading import finalize_arch_config

    monkeypatch.delenv("SYNTHEFY_QASS_MODE", raising=False)
    config = finalize_arch_config({**_MODEL_CONFIG, "qass_mode": "base_only"})
    assert config["qass_mode"] == "base_only"


def test_finalize_skips_qass_mode_without_qassmax(monkeypatch):
    """No QASSMax means no mode to record -- don't invent one."""
    from synthefy_nori.utils.loading import finalize_arch_config

    monkeypatch.delenv("SYNTHEFY_QASS_MODE", raising=False)
    config = finalize_arch_config({"use_qassmax": False, "nlayers": 2})
    assert "qass_mode" not in config


def test_training_config_has_no_architecture_fields():
    """Guard the invariant: architecture belongs to model_config only."""
    from synthefy_nori.training.config import TrainingConfig

    fields = set(TrainingConfig.__dataclass_fields__)
    leaked = fields & {
        "use_logn_attention",
        "attn_n_ref",
        "use_learnable_attn_temperature",
        "qass_mode",
        "use_qassmax",
    }
    assert not leaked, (
        f"architecture fields leaked back into TrainingConfig: {sorted(leaked)}. "
        "They belong in model_config (see loading.finalize_arch_config)."
    )
