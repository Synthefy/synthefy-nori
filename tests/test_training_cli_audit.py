"""Regression coverage for issue #635 sections A and E (CLI/config correctness)."""

from __future__ import annotations

from dataclasses import asdict
import sys

import pytest
import torch

from synthefy_nori.training import cli


class TrainerCaptured(Exception):
    pass


@pytest.fixture
def capture_cli(monkeypatch):
    captured = {}

    def capture(model, config, *, model_config, **kwargs):
        captured.update(model=model, config=config, model_config=model_config, **kwargs)
        raise TrainerCaptured

    monkeypatch.setattr(cli, "NoriTrainer", capture)

    def run(*options):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "synthefy-nori-train",
                "--device",
                "cpu",
                "--no-wandb",
                "--embed-dim",
                "48",
                "--hid-dim",
                "64",
                "--nlayers",
                "1",
                "--nhead",
                "2",
                *options,
            ],
        )
        with pytest.raises(TrainerCaptured):
            cli.main()
        return captured

    return run


@pytest.mark.parametrize(
    "options, message",
    [
        (["--num-b", "64"], "unrecognized arguments"),
        (["--feature-positional-embedding-num-slots", "64"], "requires --feature-positional-embedding-type learned"),
    ],
)
def test_cli_rejects_ignored_suboptions(monkeypatch, capsys, options, message):
    monkeypatch.setattr(sys, "argv", ["synthefy-nori-train", "--device", "cpu", *options])
    monkeypatch.setattr(
        cli, "build_model", lambda *_: pytest.fail("invalid options must fail before model construction")
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_enabled_suboptions_and_resume_inheritance(capture_cli, tmp_path):
    result = capture_cli(
        "--feature-positional-embedding-type",
        "learned",
        "--feature-positional-embedding-num-slots",
        "64",
    )
    config = result["model_config"]
    assert result["model"].feature_positional_embedding.num_embeddings == 64
    checkpoint = tmp_path / "resume.pt"
    torch.save({"model_config": config, "config": result["config"]}, checkpoint)
    # Enabled checkpoint features do not require repeating their master flags.
    resumed = capture_cli(
        "--resume",
        str(checkpoint),
        "--feature-positional-embedding-num-slots",
        "64",
    )
    assert resumed["model_config"] == config
    assert resumed["config"].model_config_source == str(checkpoint)


def test_architecture_only_checkpoint_is_recorded_without_loading_weights(capture_cli, tmp_path):
    config = cli.load_model_config(None)
    checkpoint = tmp_path / "architecture.pt"
    # Deliberately incompatible weights: --checkpoint must only read architecture.
    torch.save({"model_config": config, "model_state_dict": {"invalid": torch.ones(1)}}, checkpoint)
    result = capture_cli("--checkpoint", str(checkpoint), "--model-v2")
    record = asdict(result["config"])
    assert record["model_v2"] is True
    assert record["model_config_source"] == str(checkpoint)
    assert "checkpoint_path" not in record


def test_full_resume_config_and_dimension_guard(capture_cli, tmp_path, capsys):
    original = capture_cli("--lr", "0.001", "--synth-v4", "--warmup-steps", "5000")
    checkpoint = tmp_path / "resume.pt"
    config = original["config"]
    expected = asdict(config)
    # Architecture must come from model_config, even with stale training metadata.
    config.features_per_group += 1
    torch.save({"model_config": original["model_config"], "config": config}, checkpoint)
    resumed = capture_cli("--resume", str(checkpoint))
    expected["model_config_source"] = str(checkpoint)
    assert asdict(resumed["config"]) == expected
    overridden = capture_cli("--resume", str(checkpoint), "--lr=0.002", "--no-prefetch")
    assert overridden["config"].lr == 0.002
    assert overridden["config"].prefetch_workers == 0
    assert overridden["config"].synth_v4 is True
    with pytest.raises(SystemExit):
        capture_cli("--resume", str(checkpoint), "--resume-model-only", "--nlayers", "2")
    assert "--nlayers cannot change on a resume" in capsys.readouterr().err


def test_cli_uses_global_rank_but_local_cuda_device(monkeypatch, capture_cli):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    devices = []
    ddp_options = {}
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: None)
    monkeypatch.setattr(torch.cuda, "set_device", devices.append)
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *args, **kwargs: self)

    def ddp(model, **kwargs):
        ddp_options.update(kwargs)
        return model

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", ddp)
    result = capture_cli()
    assert result["config"].rank == 4
    assert result["config"].local_rank == 0
    assert devices == ["cuda:0"]
    assert ddp_options["device_ids"] == [0]
