"""Make changes to the migrated client source an explicit review decision.

The approved Phase-2 baseline includes the 7.0.0 package version and the minimal
Python-3.9 annotation compatibility adjustment made before PR #412.  It is not a
claim that the tree is byte-identical to the older 6.3.0 repository revision.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[1] / "libs" / "synthefy" / "src" / "synthefy"
_APPROVED_PHASE_2_SHA256 = {
    "__init__.py": "c3ff95190d22952e56d9c6c6963203587f7489eecbc6e50fcf2b5572d5aba51f",
    "api_client.py": "9e8d848ecf0d754919efd824da27f21f54021ba0d08fe6076f5cd0af0686a92c",
    "data_models.py": "f61b240cd7d8d9d1b7cac8283a45686db62460096775300834637fe1a8996953",
    "nori_client.py": "542b6cc50d15265d3fb90ede2c3e87ff6ed0cc0e4f4766f04a791a425360806e",
    "nori_data_models.py": "1ac45e913637046f8732e38d614aa33c3f70b9776b8678ec0872faed1e3ef7f0",
}


def test_migrated_synthefy_source_matches_the_approved_phase_2_baseline():
    actual = {
        path.relative_to(_SOURCE).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _SOURCE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }

    assert actual == _APPROVED_PHASE_2_SHA256, (
        "libs/synthefy/src/synthefy changed after the migration freeze. Keep migration-only "
        "PRs source-identical; update this baseline only with an explicitly approved client "
        "source change."
    )
