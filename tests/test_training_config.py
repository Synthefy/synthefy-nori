from pathlib import Path

from synthefy_nori.training.config import TrainingConfig, package_config_path


def test_training_config_constructs():
    cfg = TrainingConfig()
    assert isinstance(cfg, TrainingConfig)


def test_bundled_configs_resolve():
    assert Path(package_config_path("reg_default_noretrieval.json")).exists()
    assert Path(package_config_path("reg_allordinal_poly10_adaptive_svd256.json")).exists()
