#!/usr/bin/env python3
"""Exercise candidate-client/local-runtime cutover combinations without weights."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
import sysconfig
from pathlib import Path
from typing import Any


def _installed_module(name: str, distribution: str, expected: str):
    module = importlib.import_module(name)
    actual = importlib.metadata.version(distribution)
    if actual != expected:
        raise AssertionError(
            f"{distribution} version {actual!r} does not match {expected!r}"
        )
    module_version = getattr(module, "__version__", actual)
    if module_version != actual:
        raise AssertionError(
            f"{name}.__version__={module_version!r} does not match {actual!r}"
        )
    module_path = Path(module.__file__).resolve()
    site_packages = Path(sysconfig.get_path("purelib")).resolve()
    if not module_path.is_relative_to(site_packages):
        raise AssertionError(
            f"{name} imported from {module_path}, outside {site_packages}"
        )
    return module


def probe(
    *,
    expected_client: str,
    expected_nori: str,
    order: str,
) -> None:
    if order == "client-first":
        synthefy = _installed_module("synthefy", "synthefy", expected_client)
        if "synthefy_nori" in sys.modules:
            raise AssertionError("import synthefy eagerly imported synthefy_nori")
        synthefy_nori = _installed_module(
            "synthefy_nori", "synthefy-nori", expected_nori
        )
    else:
        synthefy_nori = _installed_module(
            "synthefy_nori", "synthefy-nori", expected_nori
        )
        synthefy = _installed_module("synthefy", "synthefy", expected_client)

    capture: dict[str, Any] = {}

    def fake_predict(
        X_train,
        y_train,
        X_test,
        *,
        task=None,
        model=None,
        **kwargs,
    ):
        capture.update(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            task=task,
            model=model,
            kwargs=kwargs,
        )
        return [6.25]

    synthefy_nori.predict = fake_predict
    client = synthefy.SynthefyNoriClient(mode="local", model="nori-6m")
    try:
        result = client.predict(
            [[0.0], [1.0]],
            [0.0, 1.0],
            [[0.5]],
        )
    finally:
        client.close()

    if result != [6.25]:
        raise AssertionError(f"unexpected local result: {result!r}")
    if capture != {
        "X_train": [[0.0], [1.0]],
        "y_train": [0.0, 1.0],
        "X_test": [[0.5]],
        "task": "regression",
        "model": "nori-6m",
        "kwargs": {},
    }:
        raise AssertionError(f"unexpected local call: {capture!r}")
    print(
        f"synthefy {expected_client} -> synthefy-nori {expected_nori} "
        f"works in {order} order"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-client", required=True)
    parser.add_argument("--expected-nori", required=True)
    parser.add_argument(
        "--order",
        choices=("client-first", "nori-first"),
        required=True,
    )
    args = parser.parse_args()
    probe(
        expected_client=args.expected_client,
        expected_nori=args.expected_nori,
        order=args.order,
    )


if __name__ == "__main__":
    main()
