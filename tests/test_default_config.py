"""One default config, resolved from one constant, used by every entry point.

The public ``NoriRegressor``, the eval CLI and the eval model wrapper all fall back to
``DEFAULT_CONFIG_NAME``, so evaluation measures what inference actually does. Keeping
those in sync is the point of these tests.
"""
import json
from pathlib import Path

from synthefy_nori.configs import (
    DEFAULT_CONFIG_NAME,
    LEGACY_SVD256_CONFIG_NAME,
    config_path,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "synthefy_nori"


def _selectors(name):
    return [m["HighDimFeatureSelector"] for m in json.load(open(config_path(name)))]


def test_default_config_is_bundled_and_pins_rank_48():
    assert DEFAULT_CONFIG_NAME == "reg_allordinal_poly10_adaptive_svd48.json"
    assert Path(config_path(DEFAULT_CONFIG_NAME)).is_file()
    sels = _selectors(DEFAULT_CONFIG_NAME)
    assert sels
    for s in sels:
        assert s["svd_components"] == 48
        # Gate stays at 256: lowering it to 64 so the selector reaches 64 < p <= 256
        # cost -0.069 across the 13 standard-suite tables in that band (12/13 lost).
        assert s["n_features_threshold"] == 256


def test_legacy_config_still_shipped_for_reproducing_older_numbers():
    assert Path(config_path(LEGACY_SVD256_CONFIG_NAME)).is_file()
    for s in _selectors(LEGACY_SVD256_CONFIG_NAME):
        assert s["svd_components"] == 256


def test_default_differs_from_legacy_only_in_rank():
    a, b = _selectors(LEGACY_SVD256_CONFIG_NAME), _selectors(DEFAULT_CONFIG_NAME)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert {k: v for k, v in x.items() if k != "svd_components"} == \
               {k: v for k, v in y.items() if k != "svd_components"}


def test_no_module_hardcodes_a_config_filename():
    """Every consumer must go through the constant."""
    allowed = {SRC / "configs" / "__init__.py"}
    offenders = [
        str(py.relative_to(ROOT)) for py in SRC.rglob("*.py")
        if py not in allowed
        and (DEFAULT_CONFIG_NAME in py.read_text() or LEGACY_SVD256_CONFIG_NAME in py.read_text())
    ]
    assert not offenders, f"hardcoded config filename in: {offenders}"


def test_every_entry_point_resolves_to_the_same_default():
    for rel in ("api.py", "evaluation/cli.py", "evaluation/models.py"):
        assert "DEFAULT_CONFIG_NAME" in (SRC / rel).read_text(), f"{rel} bypasses the constant"
