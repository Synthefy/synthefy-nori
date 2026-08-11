"""Permanent guards for ordinary pull-request CI routing.

These checks protect two easy-to-miss properties: the main test command must
continue collecting both top-level and source-co-located tests, and client
artifact tests must continue running from the built distribution in a
socket-disabled environment. Billable or credentialed live validation belongs
in protected workflows, never ordinary pull-request CI.
"""

import json
import re
from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10 compatibility
    import tomli as tomllib


_ROOT = Path(__file__).resolve().parents[1]
_CI_PATH = _ROOT / ".github" / "workflows" / "ci.yml"
_PROJECT_PATH = _ROOT / "pyproject.toml"


def _workflow() -> dict:
    return yaml.load(_CI_PATH.read_text(), Loader=yaml.BaseLoader)


def _run_steps(job: dict) -> list[str]:
    return [
        step["run"]
        for step in job["steps"]
        if isinstance(step, dict) and "run" in step
    ]


def test_ordinary_pr_ci_always_reports_and_collects_every_root_test_tree():
    workflow = _workflow()
    pull_request = workflow["on"]["pull_request"]

    assert not isinstance(pull_request, dict) or not {
        "paths",
        "paths-ignore",
    }.intersection(pull_request)
    assert "uv run pytest" in _run_steps(workflow["jobs"]["unit"])

    project = tomllib.loads(_PROJECT_PATH.read_text())
    assert project["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests", "src"]


def test_client_artifact_tests_keep_their_explicit_offline_route():
    workflow = _workflow()
    scripts = "\n".join(_run_steps(workflow["jobs"]["synthefy-artifact"]))

    assert '"$GITHUB_WORKSPACE/libs/synthefy/tests"' in scripts
    assert '"$GITHUB_WORKSPACE/tests/test_synthefy_client_compat.py"' in scripts
    assert "--disable-socket" in scripts
    assert '"$RUNNER_TEMP"' in scripts
    assert '"site-packages" in synthefy.__file__' in scripts
    assert "--import-mode=importlib" in scripts
    assert "uv build --package synthefy" in scripts
    assert "--no-sources" in scripts
    forbidden_sources = (
        "candidate_ref",
        "candidate_sha",
        "git+https://github.com/synthefy/synthefy",
        "repos/synthefy/synthefy",
    )
    assert not [value for value in forbidden_sources if value in scripts.lower()]


def test_required_test_context_fails_closed_over_every_offline_package_gate():
    workflow = _workflow()
    aggregate = workflow["jobs"]["test"]

    assert aggregate["name"] == "test"
    assert set(aggregate["needs"]) == {
        "distribution-boundaries",
        "synthefy-artifact",
        "unit",
    }
    assert aggregate["if"] == "always()"
    assert "continue-on-error" not in aggregate

    step = aggregate["steps"][0]
    assert step["env"] == {
        "UNIT_RESULT": "${{ needs.unit.result }}",
        "SYNTHEFY_ARTIFACT_RESULT": (
            "${{ needs['synthefy-artifact'].result }}"
        ),
        "DISTRIBUTION_BOUNDARIES_RESULT": (
            "${{ needs['distribution-boundaries'].result }}"
        ),
    }
    assert "exit 1" in step["run"]


def test_ordinary_pr_ci_never_receives_live_validation_credentials():
    workflow = json.dumps(_workflow())

    forbidden = (
        r"\bsecrets\b",
        r"\bid-token\s*:\s*write\b",
        r"aws-actions/configure-aws-credentials",
        r"\brole-to-assume\b",
        r"\bAWS_(?:ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN)\b",
        r"\b(?:BASETEN|NORI|SYNTHEFY)[A-Z0-9_]*(?:API|MGMT|GATEWAY)"
        r"[A-Z0-9_]*KEY\b",
    )
    matches = {
        pattern: match.group(0)
        for pattern in forbidden
        if (match := re.search(pattern, workflow, flags=re.IGNORECASE))
    }
    assert not matches
