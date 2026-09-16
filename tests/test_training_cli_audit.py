"""Regression coverage for issue #635 section E (CLI/config correctness)."""

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
