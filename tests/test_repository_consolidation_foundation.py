"""Pin inputs and boundaries for Synthefy/nori-monorepo#216."""

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ADR = _REPO_ROOT / "docs" / "architecture" / "0001-consolidate-synthefy-source-tree.md"
_MANIFEST = _ADR.with_suffix(".json")
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


def _load_manifest():
    return json.loads(_MANIFEST.read_text())


def test_integration_base_is_pinned_to_the_internal_main_tree():
    integration_base = _load_manifest()["integration_base"]

    assert integration_base["repository"] == "Synthefy/synthefy-nori-internal"
    assert integration_base["commit"] == "7cc1e6780d8394fe37a5ba98febf1a26c56d24e5"
    assert integration_base["tree"] == "5d114ba761de8082441c17938fe2d567cd453153"
    assert _FULL_SHA.fullmatch(integration_base["commit"])
    assert _FULL_SHA.fullmatch(integration_base["tree"])


def test_source_snapshot_is_pinned_to_the_reviewed_client_tree():
    manifest = _load_manifest()
    source = manifest["source_snapshot"]

    assert source["repository"] == "Synthefy/synthefy"
    assert source["commit"] == "9ecc3d2fad8e37e95869379cc05f328597e258f9"
    assert source["tree"] == "c99362d3c2449991fea1985ea1dc510e9af6d3cc"
    assert source["pyproject_blob"] == "b5af6433f10ffe069a9796b4fa0be8f074b05c12"
    assert source["version"] == "6.3.0"
    assert source["import_method"] == "single_snapshot_commit"
    assert source["target_project_root"] == "libs/synthefy"
    assert _FULL_SHA.fullmatch(source["commit"])
    assert _FULL_SHA.fullmatch(source["tree"])


def test_missing_client_license_file_remains_a_publication_blocker():
    source = _load_manifest()["source_snapshot"]

    assert source["declared_license"] == "MIT"
    assert source["required_license_file"] == "LICENSE"
    assert source["license_file_present"] is False
    assert source["publication_blocked_until_license_resolved"] is True


def test_package_and_api_boundaries_match_the_accepted_decision():
    manifest = _load_manifest()
    package = manifest["package_contract"]
    api = manifest["api_contract"]

    assert package["published_dependency_direction"] == "synthefy-nori -> synthefy"
    assert package["heavy_dependency_constraint"] == "synthefy>=7,<8"
    assert package["python_floor_target"] == ">=3.9"
    assert package["base_synthefy_must_be_torch_free"] is True
    assert api["public_facade"] == "SynthefyNoriClient"
    assert api["request_type"] == "NoriPredictRequest"
    assert api["response_type"] == "NoriPredictResponse"
    assert api["supported_modes"] == ["remote", "sagemaker", "local"]
    assert api["auto_mode_supported"] is False
    assert api["bound_sessions_in_scope"] is False
    assert api["hosted_request_fields"] == ["X_train", "y_train", "X_test"]


def test_release_scope_is_internal_first_and_does_not_expand_hosting():
    release = _load_manifest()["release_scope"]

    assert release["promotion_order"] == ["internal", "staging", "public"]
    assert release["public_is_only_publisher"] is True
    assert release["artifact_publish_order"] == ["synthefy", "synthefy-nori"]
    assert release["stop_if_lightweight_verification_fails"] is True
    assert release["fix_forward_only"] is True
    unchanged = [
        "hosted_wire_change",
        "checkpoint_or_weight_change",
        "model_slug_or_gateway_binding_change",
        "pricing_or_usage_dimension_change",
        "billing_allowlist_change",
        "key_limit_or_entitlement_change",
        "console_or_iac_behavior_change",
    ]
    assert all(release[field] is False for field in unchanged)


def test_external_gates_were_owned_bounded_and_open_at_acceptance():
    manifest = _load_manifest()
    gates = manifest["external_gates"]

    assert manifest["record_kind"] == "accepted_decision_snapshot"
    assert {gate["id"] for gate in gates} == {
        "disable-delete-subtree-writer",
        "quarantine-vendored-client-source",
        "rotate-subtree-token",
        "freeze-overlapping-client-prs",
        "approve-import-license-treatment",
        "inventory-unbounded-consumers",
        "verify-artifact-license-files",
        "configure-namespaced-trusted-publishers",
        "revoke-standalone-trusted-publisher",
        "revoke-generic-nori-trusted-publishers",
    }
    for gate in gates:
        assert gate["owner_repository"]
        assert gate["surface"]
        assert gate["required_before"]
        assert gate["status_at_acceptance"] == "open"


def test_adr_and_manifest_cross_reference_each_other():
    manifest = _load_manifest()
    adr = _ADR.read_text()

    assert manifest["decision"] == str(_ADR.relative_to(_REPO_ROOT))
    assert _MANIFEST.name in adr
    assert manifest["issue"] in adr
