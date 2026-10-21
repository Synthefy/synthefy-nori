#!/usr/bin/env python3
"""Print the version a package declares, read from the file that declares it.

Workflows need the version *before* a build or an install exists, so this reads the
declaring source rather than importing the package or asking ``importlib.metadata``.

The two packages declare it in different places, and this maps each to its own, rather
than guessing:

* ``synthefy`` declares it in the package. ``libs/synthefy/pyproject.toml`` takes it from
  there via ``[tool.hatch.version]``, so there is no field in pyproject to parse.
* ``synthefy-nori`` still declares it in ``pyproject.toml`` and mirrors it in the module.

Deliberately stdlib-only, regex rather than ``tomllib``: this runs on a bare runner python
before any environment is synced, and on the Python 3.9 client lanes where ``tomllib`` does
not exist.

Usage:
    declared_version.py [synthefy|synthefy-nori]        # default: synthefy
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

_DECLARED_IN = {
    "synthefy": (
        _ROOT / "libs" / "synthefy" / "src" / "synthefy" / "__init__.py",
        r'^__version__ = "([^"]+)"',
    ),
    "synthefy-nori": (
        _ROOT / "pyproject.toml",
        r'^version = "([^"]+)"',
    ),
}


def declared_version(package: str) -> str:
    """The version ``package`` declares, or exit non-zero saying which file is wrong."""
    try:
        path, pattern = _DECLARED_IN[package]
    except KeyError:
        raise SystemExit(f"unknown package {package!r}; expected one of {sorted(_DECLARED_IN)}") from None
    match = re.search(pattern, path.read_text(), re.MULTILINE)
    if match is None:
        raise SystemExit(f"{path} must declare a version matching {pattern!r}")
    return match.group(1)


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise SystemExit(f"usage: {Path(__file__).name} [{'|'.join(_DECLARED_IN)}]")
    print(declared_version(argv[0] if argv else "synthefy"))


if __name__ == "__main__":
    main(sys.argv[1:])
