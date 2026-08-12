"""Guard the reviewed Synthefy SDK snapshot import for issue #216."""

import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _REPO_ROOT / "libs" / "synthefy"
_MANIFEST_PATH = _PROJECT_ROOT / "SOURCE_SNAPSHOT.json"

_IMPORTED_BLOBS = {
    "CHANGELOG.md": "7b69d85855e255bc5af57239fa4256862ff9994f",
    "README.md": "66e6d0f2cef027f4579084d775531374cd65a884",
    "pyproject.toml": "b5af6433f10ffe069a9796b4fa0be8f074b05c12",
    "pytest.ini": "46dcf9a9271b23f5faf9417a5ef5c48c041bd316",
    "src/synthefy/__init__.py": "056405957f4d1db3659ec1ff7154b274412f37f3",
    "src/synthefy/api_client.py": "93e3706c47020139b81b0850deee0ec4895f8907",
    "src/synthefy/data_models.py": "725e79d92c73313da6ab1c4154b73701e61458f6",
    "src/synthefy/nori_client.py": "a6fb1d1d91677ff9aa8c99186088bf63fd24eea7",
    "src/synthefy/nori_data_models.py": "d915657b53526f461246b74e5de354000b508b74",
    "tests/__init__.py": "df085fb789a46419a7a00f080c6e4f917ee156d3",
    "tests/conftest.py": "878c02f191956c44e9f8518cc43ab5a09e302e0a",
    "tests/test_data_models.py": "c15648965d81080db88e230c08cba86400829ca3",
    "tests/test_nori_client.py": "c46acbe2b083e88eb569fe00f83c8056b7d378bf",
    "tests/test_nori_data_models.py": "cafc0ab4d1dc55e2345ed65a3d2f6555dc18807d",
}

_EXCLUDED_BLOBS = {
    ".github/workflows/publish.yaml": "37ec020d5cf1d529988e9cbb8ce4ffe56f25e046",
    ".github/workflows/tests.yaml": "5e2b53558884cfc4db053adbf46003beabea9e5a",
    ".gitignore": "220a3ca1f278e9e308298e95f74b6df598d86af9",
    "INTERNAL_README.md": "bf551d97e2932eb789f22d28d8fc569c9c33a73d",
    "dogfood/DOGFOOD.md": "dec0706f4d6ef3964df8bcaa76a287f5ece900f4",
    "dogfood/dogfood_local.py": "a9f78cb9e9862b7b6d829f400bfcd13039208adc",
    "dogfood/dogfood_remote.py": "e49e372206ab4eb1a27e2a90f5d63eb79493e6c9",
    "dogfood/dogfood_remote_realdata.py": "af939f573bba594928c38aa7e515f11cdf8fafee",
    "dogfood/modal_t4.py": "2009921ec1e5d00d1c58ff8f2bc8bd27db7cb8e1",
    "tests/online_tests/__init__.py": "62dcbc5f4f18dfa9da32395806c89c4e15f617ca",
    "tests/online_tests/test_core_forecast_backtest_api.py": ("c3706d9b5cc51df03d0d09e23223165a11d35bd2"),
    "tests/online_tests/test_hotel_demand.py": ("c4502e24af24d1e42fc219c7177b3009aa8a503d"),
    "tests/online_tests/test_inventory_forecasting.py": ("59927286a5a9bf768e8a046d29683577bce70e74"),
    "tests/online_tests/test_nori_memory_rungs.py": ("6da3cb8dd1fcac9163ce18f19800316f124d62db"),
    "tests/online_tests/test_pricing_simulation.py": ("520ba09a0127585e6acfa979886ec17e4d3af78d"),
    "uv.lock": "8eb03be0f17a16a394b2d7226bb2ed5526b2d8d6",
}

_RELOCATED_SOURCES = {
    "tests/test_featurize.py": {
        "source_project_path": "../../tests/test_featurize.py",
        "source_commit": "85c083c8f3fca791c9959ca5fd0d4a83eeb2492a",
        "source_blob": "5d915a0c18aa32fb888ce95437b0b3432ec2c253",
        "result_blob": "5812c855f6ee3e741ad56f15b3a6bcdf5f6e13d4",
        "sha256": "a84e883dafcda139695e6b41a8ef78604d5788506f8e30b8885ce48fa7263329",
        "phase": "tabular_preparation_ownership",
        "decision_record": "../../docs/architecture/0001-consolidate-synthefy-source-tree-phase-2.json",
        "changes": [
            "retain a small client-only contract suite for the canonical tabular featurizer",
            "cover ordinal, one-hot, validation, and non-DataFrame behavior without synthefy-nori",
        ],
    },
    "src/synthefy/featurize.py": {
        "source_project_path": "../../src/synthefy_nori/featurize.py",
        "source_commit": "85c083c8f3fca791c9959ca5fd0d4a83eeb2492a",
        "source_blob": "7a0b7f6586b27fd4a5225a3317ebb2e41205b007",
        "result_blob": "e7c6e5ed3bb7d22196fe5509f578ea08eab2a1e8",
        "sha256": "13f7c9b3007719cd2747ddeb753ee45d9c248cd6860cde300f8a56c0bfbd4668",
        "phase": "tabular_preparation_ownership",
        "decision_record": "../../docs/architecture/0001-consolidate-synthefy-source-tree-phase-2.json",
        "changes": [
            "move the existing tabular alignment and featurization behavior into the lightweight package",
            "update ownership-focused module guidance without changing public defaults or results",
            "parameterize only the internal warning stacklevel so both legacy helpers and the client retain their prior callsites",
        ],
    },
    "src/synthefy/text_features.py": {
        "source_project_path": "../../src/synthefy_nori/text_features.py",
        "source_commit": "36d1e317e33ace2d439589b2ddd8365bd1e3ff91",
        "source_blob": "104e7fb37b3c968d0156663f74d46b91aa32e341",
        "result_blob": "24065c08c891e652a3c68bd835439388866bfdae",
        "sha256": "f9b0b3de56223dabc4b933e0c4069242177bfa81b3c6946bdc1f0613f1c33122",
        "phase": "text_feature_ownership",
        "decision_record": "../../docs/architecture/0001-consolidate-synthefy-source-tree-phase-2.json",
        "changes": [
            "move the existing text-feature implementation into the lightweight package",
            "adjust package-specific install guidance without changing preprocessing behavior",
            "guard the optional scikit-learn import with actionable synthefy[text] guidance",
        ],
    },
    "src/synthefy/nori_ts/__init__.py": {
        "source_project_path": "../../src/synthefy_nori/nori_ts/__init__.py",
        "source_commit": "de7704303b5ea5725323ae20d8fe738409a198e7",
        "source_blob": "ca46d88167837c3a4ba3bb02b398a6f343fec2c3",
        "result_blob": "68e98f9d2cc004c53a4fb5aaa9f2a6d0ad4b9496",
        "sha256": "6666ba4a1138be97faf9b08079ea47bc8a3dd1caa9765ef44e61394f47ea0691",
        "phase": "time_series_feature_ownership",
        "decision_record": "../../docs/architecture/0001-consolidate-synthefy-source-tree-ci-gates.json",
        "changes": [
            "export NoriTSForecaster and DEFAULT_QUANTILES from the canonical lightweight namespace",
            "keep optional forecasting imports confined to synthefy.nori_ts rather than the base synthefy import",
        ],
    },
    "src/synthefy/nori_ts/core.py": {
        "source_project_path": "../../src/synthefy_nori/nori_ts/core.py",
        "source_commit": "304dc445bddf00271e9227aae88727b735da1cee",
        "source_blob": "89ddcd29913f809d3c82d36f4cb70a02a7ef1994",
        "result_blob": "90a0da5f81617bf02a41bed24edc9eaec81f58a2",
        "sha256": "09151778367c99c0206fcb9ec8c8dc58840f35fd02ec2ef79c0e6c48621a5244",
        "phase": "time_series_forecaster_ownership",
        "decision_record": "../../docs/architecture/0001-consolidate-synthefy-source-tree-ci-gates.json",
        "changes": [
            "move NoriTSForecaster into the lightweight package without renaming its public class or request types",
            "construct or accept SynthefyNoriClient for every backend instead of owning a second local estimator path",
            "require explicit mode and model configuration and reject auto mode without selecting a default model",
        ],
    },
    "src/synthefy/nori_ts/tsfeatures/__init__.py": {
        "source_project_path": "../../src/synthefy_nori/nori_ts/tsfeatures/__init__.py",
        "source_commit": "de7704303b5ea5725323ae20d8fe738409a198e7",
        "source_blob": "e9b2688266d6c699f5b041806dbf7030c29541f4",
        "result_blob": "1d86a7366ae2f45ad574a12a9a3c3a734cc79e66",
        "sha256": "592c2ba9fe99ba7791834eb89c380f1ea888940444592ff43ba3fad209be1a50",
        "phase": "time_series_feature_ownership",
        "decision_record": "../../docs/architecture/0001-consolidate-synthefy-source-tree-ci-gates.json",
        "changes": [
            "move the public model-free time-series preparation exports into the lightweight distribution",
            "rewrite intra-package imports and distribution-local ownership/provenance wording",
            "record postponed annotation evaluation in feature_transformer.py separately from behavioral changes",
        ],
    },
    "src/synthefy/nori_ts/tsfeatures/auto_features.py": {
        "source_project_path": "../../src/synthefy_nori/nori_ts/tsfeatures/auto_features.py",
        "source_commit": "de7704303b5ea5725323ae20d8fe738409a198e7",
        "source_blob": "fc9d15dd05347ab41af1824b60ae5935d27f78a5",
        "result_blob": "689cc202096551173a9494e6f133e22174f53f96",
        "sha256": "de92034f719123da8e9eb4331242a6ce0553ecbc04976d98c2159b95b1b81fb8",
        "phase": "time_series_feature_ownership",
        "decision_record": "../../docs/architecture/0001-consolidate-synthefy-source-tree-ci-gates.json",
        "changes": [
            "move the existing automatic seasonal feature implementation into the lightweight distribution",
            "rewrite only intra-package imports and distribution-local ownership/provenance wording",
        ],
    },
    "src/synthefy/nori_ts/tsfeatures/basic_features.py": {
        "source_project_path": "../../src/synthefy_nori/nori_ts/tsfeatures/basic_features.py",
        "source_commit": "de7704303b5ea5725323ae20d8fe738409a198e7",
        "source_blob": "79dc858ec55c78666e9c33470aaf7a284fb84cec",
        "result_blob": "f93d7cd7ad9652cfc7e6e33488c42538dccb06ef",
        "sha256": "57df80f6ab2746e04619ae27dc6e6580818f50ebcf8ad6e5c7e34c00bd10e204",
        "phase": "time_series_feature_ownership",
        "decision_record": "../../docs/architecture/0001-consolidate-synthefy-source-tree-ci-gates.json",
        "changes": [
            "move the existing model-free calendar and running-index generators into the lightweight distribution",
            "rewrite only intra-package imports and distribution-local ownership/provenance wording",
        ],
    },
    "src/synthefy/nori_ts/tsfeatures/data_preparation.py": {
        "source_project_path": "../../src/synthefy_nori/nori_ts/tsfeatures/data_preparation.py",
        "source_commit": "de7704303b5ea5725323ae20d8fe738409a198e7",
        "source_blob": "c94d5ae672bdbf397dbfebd80073f2a4ccd9fc9b",
        "result_blob": "c52dc4dc53a569599f51a3c57198c87f604a6301",
        "sha256": "7016e60000a7ae6277f9b137ba8e42849bd3040741e874ba270d8df5f2574bae",
        "phase": "time_series_feature_ownership",
        "decision_record": "../../docs/architecture/0001-consolidate-synthefy-source-tree-ci-gates.json",
        "changes": [
            "move the existing model-free horizon and GluonTS conversion helpers into the lightweight distribution",
            "rewrite only intra-package imports and distribution-local ownership/provenance wording",
            "preserve the existing explicit-frequency horizon behavior without refactoring",
        ],
    },
    "src/synthefy/nori_ts/tsfeatures/feature_generator_base.py": {
        "source_project_path": "../../src/synthefy_nori/nori_ts/tsfeatures/feature_generator_base.py",
        "source_commit": "de7704303b5ea5725323ae20d8fe738409a198e7",
        "source_blob": "b4b4d39d6d04888b7869967fe7fbbe39d60a530f",
        "result_blob": "7f529ebea97b3ae6717614251d69fc277f495834",
        "sha256": "93090df227bede92ab575ca8c0097b76d0d8f2bccaea9e7910f17050ae3f9bf7",
        "phase": "time_series_feature_ownership",
        "decision_record": "../../docs/architecture/0001-consolidate-synthefy-source-tree-ci-gates.json",
        "changes": [
            "move the existing model-free feature-generator contract into the lightweight distribution",
            "rewrite only intra-package imports and distribution-local ownership/provenance wording",
        ],
    },
    "src/synthefy/nori_ts/tsfeatures/feature_transformer.py": {
        "source_project_path": "../../src/synthefy_nori/nori_ts/tsfeatures/feature_transformer.py",
        "source_commit": "de7704303b5ea5725323ae20d8fe738409a198e7",
        "source_blob": "6ad0c7d680be505ccc93bfa49eb0d84943d7673f",
        "result_blob": "d66d4c8c8a9920c9bd3a30cd84ba2ee464861e75",
        "sha256": "ad789f99d77de4c15e755f202422c55702dc2ac6297e830441ac2cf0227c0ce2",
        "phase": "time_series_feature_ownership",
        "decision_record": "../../docs/architecture/0001-consolidate-synthefy-source-tree-ci-gates.json",
        "changes": [
            "move the existing model-free train/test feature transformer into the lightweight distribution",
            "rewrite intra-package imports and distribution-local ownership/provenance wording",
            "enable postponed annotation evaluation for Python 3.9 without changing forecasting behavior",
        ],
    },
    "src/synthefy/nori_ts/tsfeatures/ts_dataframe.py": {
        "source_project_path": "../../src/synthefy_nori/nori_ts/tsfeatures/ts_dataframe.py",
        "source_commit": "de7704303b5ea5725323ae20d8fe738409a198e7",
        "source_blob": "6e201d55467a932ab180c83559b1b06a2e3b1372",
        "result_blob": "c2362a4f77389ecc28274b0c5f2ae45d2752e421",
        "sha256": "612b9b7f020fee695d0dad1d4b5b16eb836a96d718142ca1cfa550a96ffc2e5e",
        "phase": "time_series_feature_ownership",
        "decision_record": "../../docs/architecture/0001-consolidate-synthefy-source-tree-ci-gates.json",
        "changes": [
            "move the existing TimeSeriesDataFrame implementation body into the lightweight distribution",
            "change only distribution-local provenance and ownership wording; preserve the implementation body",
        ],
    },
    "tests/test_tsfeatures.py": {
        "source_project_path": "../../tests/test_nori_ts.py",
        "source_commit": "de7704303b5ea5725323ae20d8fe738409a198e7",
        "source_blob": "5d3be3e06c143dcefecdd1eb2d2fd16d875dc4ce",
        "result_blob": "a4f353b247fa67248fa2db06b2959f18bd110935",
        "sha256": "ea57d1b91e71b74e4535aa759b319eb1a7877df47ff0d80eff409abc25a787ff",
        "phase": "time_series_feature_ownership",
        "decision_record": "../../docs/architecture/0001-consolidate-synthefy-source-tree-ci-gates.json",
        "changes": [
            "retain only model-free horizon, schema, train/test-boundary, and multi-series generator contracts",
            "build the feature set directly without importing heavy core._default_features",
            "gate execution on the synthefy[forecasting] optional runtime",
            "cover explicit-frequency gappy inputs, generated float32 columns without target downcasting, and static-feature preservation",
        ],
    },
}

_TARGET_FILES = set(_IMPORTED_BLOBS) | set(_RELOCATED_SOURCES) | {
    ".gitignore",
    "LICENSE",
    "NOTICE",
    "SOURCE_SNAPSHOT.json",
    "licenses/Apache-2.0.txt",
}

_PHASE_TWO_CHAIN_HEADS = {
    "pyproject.toml": {
        "input_blob": "0f9e10c5b6175b496176c717083cf3cf72862cda",
        "result_blob": "cd572dc5c11fbc905ae64b4c96dccc2d20371431",
    },
    "pytest.ini": {
        "input_blob": "dbbb75cc34a9c63ad2ee66e1ecc6bfe529ab51b0",
        "result_blob": "a15b1a9438668b6f3558ea2f8a29d2aacbc86e86",
    },
    "src/synthefy/__init__.py": {
        "input_blob": "056405957f4d1db3659ec1ff7154b274412f37f3",
        "result_blob": "35e85d9327d37c798c40ddda25087075ea60af0c",
    },
    "src/synthefy/api_client.py": {
        "input_blob": "93e3706c47020139b81b0850deee0ec4895f8907",
        "result_blob": "e108b75e10942c51a8af17cf8e4e34b930d838e3",
    },
    "tests/conftest.py": {
        "input_blob": "878c02f191956c44e9f8518cc43ab5a09e302e0a",
        "result_blob": "e8f50cb2d1303bb2405596412f89fa4bc8eb60fb",
    },
    "tests/test_nori_client.py": {
        "input_blob": "c46acbe2b083e88eb569fe00f83c8056b7d378bf",
        "result_blob": "16e63e493698af0fefa4136e32eb313b946f6893",
    },
}
_PHASE_TWO_DECISION = (
    "../../docs/architecture/0001-consolidate-synthefy-source-tree-phase-2.json"
)


def _load_manifest():
    return json.loads(_MANIFEST_PATH.read_text())


def _post_import_transformations():
    return _load_manifest()["post_import_history"]["transformations"]


def _git_blob(path):
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()  # noqa: S324


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_project_entries():
    project_prefix = f"{_PROJECT_ROOT.relative_to(_REPO_ROOT)}/"
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z", "--", project_prefix],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    entries = {}
    for record in result.stdout.rstrip("\0").split("\0"):
        metadata, path = record.split("\t", 1)
        entries[path.removeprefix(project_prefix)] = metadata.split(" ", 1)[0]
    return entries


def test_snapshot_manifest_pins_the_source_and_selected_boundary():
    manifest = _load_manifest()

    assert manifest["schema_version"] == 2
    assert manifest["source"] == {
        "repository": "Synthefy/synthefy",
        "commit": "9ecc3d2fad8e37e95869379cc05f328597e258f9",
        "tree": "c99362d3c2449991fea1985ea1dc510e9af6d3cc",
        "subtrees": {
            "src/synthefy": "4caf181f092208d2fdb5d5a64997d6d1046e7944",
            "tests": "b5eb4713a8af8d0cc7031675da67c82d46cdd0f7",
            "tests/online_tests": "74cc3c0a994a113633392900a8b1fcb05c2c173f",
        },
        "version": "6.3.0",
    }
    assert manifest["import"]["method"] == "single_snapshot_commit"
    assert manifest["import"]["target_project_root"] == "libs/synthefy"
    assert manifest["import"]["included_path_count"] == len(_IMPORTED_BLOBS)
    assert manifest["import"]["included_blobs"] == _IMPORTED_BLOBS
    assert manifest["post_import_history"]["current_version"] == "7.0.0"
    assert manifest["post_import_history"]["source_relocations"] == _RELOCATED_SOURCES


def test_byte_identical_imports_match_the_pinned_git_blobs():
    imported = _load_manifest()["import"]
    evolved = set(_post_import_transformations())

    for relative_path in imported["byte_identical_paths"]:
        if relative_path not in evolved:
            assert _git_blob(_PROJECT_ROOT / relative_path) == _IMPORTED_BLOBS[relative_path]


def test_reviewed_transformations_and_license_files_are_pinned():
    manifest = _load_manifest()
    evolved = manifest["post_import_history"]["transformations"]

    for relative_path, transformation in manifest["import"]["transformed_paths"].items():
        assert transformation["source_blob"] == _IMPORTED_BLOBS[relative_path]
        if relative_path in evolved:
            assert transformation["result_blob"] == evolved[relative_path][0]["input_blob"]
        else:
            assert transformation["result_blob"] == _git_blob(_PROJECT_ROOT / relative_path)

    for relative_path, license_file in manifest["license_treatment"]["added_files"].items():
        target = _PROJECT_ROOT / relative_path
        assert license_file["git_blob"] == _git_blob(target)
        assert license_file["sha256"] == _sha256(target)

    for relative_path, build_file in manifest["build_treatment"]["added_files"].items():
        target = _PROJECT_ROOT / relative_path
        assert build_file["git_blob"] == _git_blob(target)
        assert build_file["sha256"] == _sha256(target)
        assert build_file["replaces_source_blob"] == _EXCLUDED_BLOBS[relative_path]

    for relative_path, relocation in manifest["post_import_history"]["source_relocations"].items():
        target = _PROJECT_ROOT / relative_path
        assert relocation["result_blob"] == _git_blob(target)
        assert relocation["sha256"] == _sha256(target)
        assert relocation["changes"]
        decision = (_PROJECT_ROOT / relocation["decision_record"]).resolve()
        assert decision.is_file()


def test_post_import_transformations_preserve_phase_two_and_form_continuous_chains():
    manifest = _load_manifest()
    imported = manifest["import"]
    chains = manifest["post_import_history"]["transformations"]

    assert set(chains) == {
        "CHANGELOG.md",
        "README.md",
        "pyproject.toml",
        "pytest.ini",
        "src/synthefy/__init__.py",
        "src/synthefy/api_client.py",
        "src/synthefy/nori_client.py",
        "tests/conftest.py",
        "tests/test_nori_client.py",
    }
    for relative_path, transformations in chains.items():
        if relative_path in _PHASE_TWO_CHAIN_HEADS:
            assert {
                key: transformations[0][key] for key in ("input_blob", "result_blob")
            } == _PHASE_TWO_CHAIN_HEADS[relative_path]
            assert transformations[0]["phase"] == "workspace_and_package_wiring"
            assert transformations[0]["decision_record"] == _PHASE_TWO_DECISION
        phase_one = imported["transformed_paths"].get(relative_path)
        expected_input = (
            phase_one["result_blob"] if phase_one else _IMPORTED_BLOBS[relative_path]
        )
        for transformation in transformations:
            assert transformation["input_blob"] == expected_input
            assert transformation["changes"]
            assert transformation["phase"]
            decision = (_PROJECT_ROOT / transformation["decision_record"]).resolve()
            assert decision.is_file()
            expected_input = transformation["result_blob"]
        assert expected_input == _git_blob(_PROJECT_ROOT / relative_path)


def test_imported_project_has_the_exact_reviewed_file_boundary():
    tracked = _tracked_project_entries()

    assert len(_TARGET_FILES) == 32
    assert set(tracked) == _TARGET_FILES
    assert set(tracked.values()) == {"100644"}


@pytest.mark.slow
def test_built_artifacts_match_reviewed_file_boundaries(tmp_path):
    manifest = _load_manifest()
    subprocess.run(
        ["uv", "build", "--package", "synthefy", "--out-dir", str(tmp_path)],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    version = manifest["post_import_history"]["current_version"]
    assert len(manifest["build_treatment"]["artifact_files"]["sdist"]) == 24
    assert len(manifest["build_treatment"]["artifact_files"]["wheel"]) == 22
    sdist_prefix = f"synthefy-{version}/"
    with tarfile.open(tmp_path / f"synthefy-{version}.tar.gz") as archive:
        files = {member.name.removeprefix(sdist_prefix) for member in archive.getmembers() if member.isfile()}
        assert files == set(manifest["build_treatment"]["artifact_files"]["sdist"])
        gitignore = archive.extractfile(f"{sdist_prefix}.gitignore")
        assert gitignore is not None
        assert gitignore.read() == (_PROJECT_ROOT / ".gitignore").read_bytes()

    wheel = tmp_path / f"synthefy-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        assert set(archive.namelist()) == set(manifest["build_treatment"]["artifact_files"]["wheel"])


def test_deliberately_excluded_snapshot_paths_are_pinned_and_absent():
    manifest = _load_manifest()
    excluded = manifest["excluded"]
    replacement_paths = set(manifest["build_treatment"]["added_files"])
    recorded = {
        **excluded["repository_control_and_stale_guidance"]["blobs"],
        **excluded["dogfood_and_live_tests"]["blobs"],
    }

    assert excluded["path_count"] == len(_EXCLUDED_BLOBS)
    assert recorded == _EXCLUDED_BLOBS
    assert replacement_paths == {".gitignore"}
    assert all(
        not (_PROJECT_ROOT / relative_path).exists()
        for relative_path in recorded
        if relative_path not in replacement_paths
    )
    assert excluded["dogfood_and_live_tests"]["status"] == ("omitted_pending_rehome_review")
