"""Fail-closed checks for the two independent PyPI release streams."""

from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_ROOT = _ROOT / ".github" / "workflows"
_PUBLIC_REPOSITORY_GATE = (
    "github.repository == 'Synthefy/synthefy-nori'"
)
_SPECS = {
    "synthefy": {
        "workflow": "publish-synthefy.yml",
        "tag_prefix": "synthefy-v",
        "project_root": "libs/synthefy",
        "build": "uv build --package synthefy --no-sources",
        "targets": ["testpypi", "pypi"],
        "publish_jobs": {"publish-testpypi", "publish-pypi"},
        "environments": {"synthefy-testpypi", "synthefy-pypi"},
    },
    "synthefy-nori": {
        "workflow": "publish-synthefy-nori.yml",
        "tag_prefix": "synthefy-nori-v",
        "project_root": ".",
        "build": "uv build --package synthefy-nori --no-sources",
        "targets": ["pypi"],
        "publish_jobs": {"publish-pypi"},
        "environments": {"synthefy-nori-pypi"},
    },
}


def _load(name: str) -> dict:
    return yaml.load(
        (_WORKFLOW_ROOT / name).read_text(),
        Loader=yaml.BaseLoader,
    )


def _run_scripts(job: dict) -> str:
    return "\n".join(
        step["run"]
        for step in job["steps"]
        if isinstance(step, dict) and "run" in step
    )


def test_generic_publisher_is_removed_and_namespaced_publishers_are_complete():
    assert not (_WORKFLOW_ROOT / "publish.yml").exists()
    assert {
        path.name for path in _WORKFLOW_ROOT.glob("publish*.yml")
    } == {spec["workflow"] for spec in _SPECS.values()}

    for distribution, spec in _SPECS.items():
        workflow = _load(spec["workflow"])
        assert workflow["name"] == f"publish-{distribution}"
        assert workflow["on"]["release"]["types"] == ["published"]

        inputs = workflow["on"]["workflow_dispatch"]["inputs"]
        assert set(inputs) == {"distribution", "version", "target", "tag"}
        assert inputs["distribution"]["required"] == "true"
        assert inputs["distribution"]["options"] == [distribution]
        assert inputs["version"]["required"] == "true"
        assert inputs["tag"]["required"] == "true"
        assert inputs["target"]["options"] == spec["targets"]

        build = workflow["jobs"]["build"]
        assert spec["tag_prefix"] in build["if"]
        scripts = _run_scripts(build)
        assert f"expected_tag=\"{spec['tag_prefix']}$version\"" in scripts
        assert '"$GITHUB_REF" != "refs/tags/$tag"' in scripts
        assert 'git rev-parse "refs/tags/$tag^{}"' in scripts
        assert "git merge-base --is-ancestor" in scripts
        assert "origin/main" in scripts
        assert spec["build"] in scripts
        assert "twine check --strict" in scripts
        assert "uv pip install" in scripts
        assert "uv pip check" in scripts
        assert spec["project_root"] in scripts or distribution == "synthefy-nori"

        body = (_WORKFLOW_ROOT / spec["workflow"]).read_text().lower()
        assert "releases/latest" not in body
        assert "latest release" not in body
        assert "secrets." not in body
        if "testpypi" not in spec["targets"]:
            assert "testpypi" not in body


def test_only_public_namespaced_jobs_can_publish_with_oidc():
    environments = set()
    artifact_names = set()

    for distribution, spec in _SPECS.items():
        workflow = _load(spec["workflow"])
        build = workflow["jobs"]["build"]
        artifact_names.add(
            build["outputs"]["artifact"]
            + distribution
        )

        publish_jobs = {
            name: job
            for name, job in workflow["jobs"].items()
            if name.startswith("publish-")
        }
        assert set(publish_jobs) == spec["publish_jobs"]
        assert build.get("permissions") is None

        for job in publish_jobs.values():
            assert _PUBLIC_REPOSITORY_GATE in job["if"]
            assert job["permissions"] == {
                "contents": "read",
                "id-token": "write",
            }
            environment = job["environment"]["name"]
            environments.add(environment)
            assert environment in spec["environments"]
            uses = [
                step.get("uses", "")
                for step in job["steps"]
                if isinstance(step, dict)
            ]
            assert "pypa/gh-action-pypi-publish@release/v1" in uses

        for name, job in workflow["jobs"].items():
            if name not in publish_jobs:
                assert job.get("permissions", {}).get("id-token") != "write"

    assert environments == set().union(
        *(spec["environments"] for spec in _SPECS.values())
    )
    assert len(artifact_names) == 2


def test_release_runbook_names_every_authority_and_package_order():
    runbook = (_ROOT / "RELEASING.md").read_text()

    assert "upstream  ->  staging  ->  public" in runbook
    assert "Only `Synthefy/synthefy-nori` is a release authority." in runbook
    assert "Publish the lightweight SDK first." in runbook
    assert "synthefy-vX.Y.Z" in runbook
    assert "synthefy-nori-vX.Y.Z" in runbook
    for spec in _SPECS.values():
        assert spec["workflow"] in runbook
        for environment in spec["environments"]:
            assert environment in runbook
    assert "revoke" in runbook.lower()
    assert "standalone `Synthefy/synthefy`" in runbook
    assert "production smoke test requires explicit approval" in runbook
