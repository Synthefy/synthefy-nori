"""Pin inputs and boundaries for Synthefy/nori-monorepo#216."""

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ADR = _REPO_ROOT / "docs" / "architecture" / "0001-consolidate-synthefy-source-tree.md"
_MANIFEST = _ADR.with_suffix(".json")
_PHASE_0 = _ADR.with_name("0001-consolidate-synthefy-source-tree-phase-0.json")
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


def _load_manifest():
    return json.loads(_MANIFEST.read_text())


def _load_phase_0_status():
    return json.loads(_PHASE_0.read_text())


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


def test_approved_license_treatment_matches_the_public_nori_distribution():
    treatment = _load_phase_0_status()["license_treatment"]

    assert treatment["scope"] == "source_snapshot_import"
    assert treatment["status"] == "approved"
    assert treatment["approved_by"] == "Raimi"
    assert treatment["approved_at"] == "2026-08-11"
    assert treatment["approval_record"] == {
        "kind": "explicit_requester_direction",
        "planning_issue": "https://github.com/Synthefy/nori-monorepo/issues/216",
        "durable_confirmation": "review_and_merge_of_this_child_pull_request",
    }
    assert treatment["reference_repository"] == "Synthefy/synthefy-nori"
    assert treatment["reference_commit"] == ("3499f2ea066f96c38351d25206703c8ccc0c46fa")
    assert treatment["reference_blobs"] == {
        "LICENSE": "6fd5811a83419f2a56a1cdb9162043bb29cb98e2",
        "NOTICE": "57d16c34adc9478bb46a528bcac524af07b7ceaa",
        "pyproject.toml": "b7c1d1aa1d4b79a361e7e0728b05584b6aceb32e",
    }
    assert treatment["target_project_root"] == "libs/synthefy"
    assert treatment["target_spdx_expression"] == "Apache-2.0"
    assert treatment["license_file"] == "libs/synthefy/LICENSE"
    assert treatment["notice_file"] == "libs/synthefy/NOTICE"
    assert treatment["application_notice"] == "Copyright 2026 Synthefy"
    assert treatment["project_metadata"] == {
        "license": "Apache-2.0",
        "license_files": ["LICENSE", "NOTICE", "licenses/*"],
        "legacy_mit_classifier_allowed": False,
    }
    assert treatment["historical_source_fact"] == {
        "declared_license": "MIT",
        "license_file_present": False,
        "metadata_is_an_import_license_grant": False,
    }
    assert treatment["project_scope"] == {
        "root_license_and_notice_project": "synthefy-nori",
        "nested_license_and_notice_project": "synthefy",
        "preserve_more_specific_file_headers": True,
    }
    assert treatment["copy_root_license_verbatim"] is False
    assert treatment["copy_root_notice_verbatim"] is False
    assert treatment["third_party_notices_are_path_scoped"] is True
    assert treatment["initial_notice_components"] == ["Synthefy"]
    assert treatment["deferred_notice_components"] == [
        "tabpfn-time-series",
        "AutoGluon",
        "Chronos",
    ]
    assert treatment["excluded_notice_components_unless_source_is_packaged"] == [
        "StableAI",
        "LimiX",
        "TabICL",
        "model weights",
    ]
    assert treatment["license_treatment_approved_for_source_import"] is True
    assert treatment["artifact_publication_blocked_until_files_verified"] is True
    assert treatment["publication_gate_id"] == "verify-artifact-license-files"


def test_phase_0_containment_keeps_source_import_blocked_on_token_revocation():
    status = _load_phase_0_status()
    containment = status["containment"]
    secret = containment["actions_secret"]

    assert status["record_kind"] == "phase_0_status_observation"
    assert status["observed_at"] == "2026-08-11"
    assert status["observation_policy"] == "immutable_new_record_required_for_status_changes"
    assert containment["pull_request"] == ("https://github.com/Synthefy/synthefy-package/pull/3013")
    assert containment["merge_commit"] == "4b6f3f36ddacf687cce2bfacd724c0cb0ec7711a"
    assert containment["subtree_writer_workflow"] == "deleted_from_main"
    assert containment["vendored_client_source"] == "quarantined"
    assert containment["last_observed_writer_run_date"] == "2026-02-05"
    assert containment["standalone_dev_commit_after_containment"] == ("9efe7009c90ae447f0aa2e0879450a607c88af4f")
    assert secret["status"] == "deleted"
    assert secret["underlying_credential_owner"] == "unknown"
    assert secret["underlying_credential_revocation"] == "unverified"
    assert status["observed_gate_status"] == {
        "disable-delete-subtree-writer": "complete",
        "quarantine-vendored-client-source": "complete",
        "rotate-subtree-token": "incomplete_revocation_unverified",
        "freeze-overlapping-client-prs": "complete",
        "approve-import-license-treatment": "complete",
        "verify-artifact-license-files": "open",
    }
    assert status["source_import"] == {
        "allowed": False,
        "blocking_gate_ids": ["rotate-subtree-token"],
        "next_action": "revoke_or_rotate_and_verify_the_underlying_credential",
    }


def test_phase_0_records_the_exact_frozen_overlapping_pull_requests():
    frozen = _load_phase_0_status()["frozen_pull_requests"]
    expected = {
        "Synthefy/synthefy#40",
        "Synthefy/synthefy#46",
        "Synthefy/synthefy#53",
        "Synthefy/synthefy#54",
        "Synthefy/synthefy#56",
        "Synthefy/synthefy#57",
        "Synthefy/synthefy-nori-internal#125",
        "Synthefy/synthefy-nori-internal#224",
        "Synthefy/synthefy-nori-internal#269",
        "Synthefy/synthefy-nori-internal#348",
        "Synthefy/synthefy-nori-internal#369",
        "Synthefy/synthefy-nori-internal#371",
        "Synthefy/synthefy-nori-internal#384",
        "Synthefy/synthefy-nori-internal#402",
        "Synthefy/synthefy-nori-internal#403",
        "Synthefy/synthefy-nori-internal#404",
    }

    assert frozen["status"] == "open_as_draft_with_216_disposition"
    assert len(frozen["pull_requests"]) == len(expected)
    assert set(frozen["pull_requests"]) == expected


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
    phase_0 = _load_phase_0_status()
    adr = _ADR.read_text()

    assert manifest["decision"] == str(_ADR.relative_to(_REPO_ROOT))
    assert _MANIFEST.name in adr
    assert phase_0["decision"] == str(_ADR.relative_to(_REPO_ROOT))
    assert phase_0["accepted_decision_snapshot"] == str(_MANIFEST.relative_to(_REPO_ROOT))
    assert _PHASE_0.name in adr
    assert manifest["issue"] in adr
