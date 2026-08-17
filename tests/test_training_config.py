from pathlib import Path

from synthefy_nori.training.config import TrainingConfig, package_config_path


def test_training_config_constructs():
    cfg = TrainingConfig()
    assert isinstance(cfg, TrainingConfig)


def test_bundled_configs_resolve():
    assert Path(package_config_path("default_inference.json")).exists()
