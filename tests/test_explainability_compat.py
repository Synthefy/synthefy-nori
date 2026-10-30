"""Published explainability imports and install commands survive the package move."""

import importlib
import pickle
import subprocess
import sys
from importlib.metadata import metadata

import pytest
from packaging.requirements import Requirement

from synthefy_nori.explainability import NoriInterpreter


def test_legacy_interpreter_pickle_resolves_the_published_class_path():
    # Pickle records the defining module even when callers import from the package.
    legacy_class = pickle.loads(b"csynthefy_nori.explainability.interpreter\nNoriInterpreter\n.")

    assert legacy_class is NoriInterpreter
    interpreter = legacy_class(model="nori-6m", render_figure=False)
    assert interpreter.get_params()["model"] == "nori-6m"


def test_published_explainability_extra_installs_its_runtime_dependencies():
    package = metadata("synthefy-nori")
    assert "explainability" in package.get_all("Provides-Extra", [])
    requirements = [Requirement(value) for value in package.get_all("Requires-Dist", [])]
    installed = {
        requirement.name
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate({"extra": "explainability"})
    }
    assert {"interpret-core", "joblib", "matplotlib", "scipy"} <= installed


@pytest.mark.parametrize(
    "module,symbol",
    [("data", "load_demo"), ("ebm", "fit_ebm"), ("pipeline", "run"), ("viz", "plot_ebm_model")],
)
def test_published_submodules_reexport_canonical_implementations(module, symbol):
    legacy = importlib.import_module(f"synthefy_nori.explainability.{module}")
    canonical = importlib.import_module(f"synthefy_nori.interpretability.glassbox.{module}")
    assert getattr(legacy, symbol) is getattr(canonical, symbol)


def test_published_pipeline_command_line_still_runs():
    result = subprocess.run(
        [sys.executable, "-m", "synthefy_nori.explainability.pipeline", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--demo" in result.stdout
    assert "--npz" in result.stdout
