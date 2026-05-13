from pathlib import Path

from synthefy_tabular.training.config import TrainingConfig


def test_training_config_uses_bundled_eval_configs():
    cfg = TrainingConfig()
    assert Path(cfg.eval_cls_config).exists()
    assert Path(cfg.eval_reg_config).exists()
