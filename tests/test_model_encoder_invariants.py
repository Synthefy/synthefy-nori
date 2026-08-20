from __future__ import annotations

import json

import pytest
import torch

from synthefy_nori.model.encoders import (
    MaskEmbEncoder,
    NanEncoder,
    NormalizationEncoder,
    RBFembedding,
    ValidFeatureEncoder,
    get_reg_y_encoder,
)
from synthefy_nori.training.config import package_config_path
from synthefy_nori.utils.loading import build_model


def _tiny_model(
    *,
    features_per_group: int = 3,
    mask_prediction: bool = False,
    nan_handling_enabled: bool = True,
    nan_handling_y_encoder: bool = True,
    use_nan_indicator: bool = False,
    normalize_on_train_only: bool = True,
    legacy_random_rbf_flag: bool | None = None,
):
    with open(package_config_path("model_base.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(
        nlayers=1,
        embed_dim=12,
        hid_dim=24,
        nhead=3,
        features_per_group=features_per_group,
        mask_prediction=mask_prediction,
        feature_positional_embedding_type="none",
        device=None,
    )
    config["preprocess_config_x"].update(
        num_features=features_per_group,
        nan_handling_enabled=nan_handling_enabled,
        normalize_on_train_only=normalize_on_train_only,
    )
    config["encoder_config_x"].update(
        num_features=features_per_group,
        embedding_size=12,
        mask_embedding_size=12,
    )
    if use_nan_indicator:
        config["encoder_config_x"]["in_keys"] = ["data", "nan_encoding"]
    config["encoder_config_x"]["RBF_config"].update(
        token_embed_dim=4,
        n_kernels=8,
    )
    if legacy_random_rbf_flag is not None:
        config["encoder_config_x"]["RBF_config"]["use_random_kernels"] = (
            legacy_random_rbf_flag
        )
    config["encoder_config_y"].update(
        embedding_size=12,
        nan_handling_y_encoder=nan_handling_y_encoder,
    )
    return build_model(config).eval()


def test_valid_feature_count_uses_context_and_can_be_frozen():
    encoder = ValidFeatureEncoder(num_features=3)
    # Only feature 0 varies in context. Query variation in features 1 and 2 must
    # not change the group scale.
    context = torch.tensor([[[[0.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]]])
    query = torch.tensor([[[[2.0, 10.0, 20.0]]]])
    whole = torch.cat((context, query), dim=1)

    direct = encoder({"data": whole.clone(), "eval_pos": 2})
    assert direct["_valid_feature_num"].item() == 1

    frozen = encoder({
        "data": query.clone(),
        "eval_pos": 1,
        "_frozen_valid_feature_num": direct["_valid_feature_num"],
    })
    torch.testing.assert_close(frozen["data"], direct["data"][:, 2:])


def test_direct_preprocessing_and_prediction_do_not_depend_on_query_batch():
    model = _tiny_model()
    context = torch.tensor([[
        [0.0, 1.0, 1.0, 0.0, 4.0, 7.0],
        [1.0, 1.0, 1.0, 1.0, 4.0, 7.0],
        [2.0, 1.0, 1.0, 2.0, 4.0, 7.0],
        [3.0, 1.0, 1.0, 3.0, 4.0, 7.0],
    ]])
    common_query = torch.tensor([[[4.0, 10.0, 1.0, 4.0, 9.0, 7.0]]])
    extra_query = torch.tensor([[[5.0, 1.0, 20.0, 5.0, 8.0, 9.0]]])
    y_train = torch.arange(4.0).unsqueeze(0)

    one = torch.cat((context, common_query), dim=1)
    many = torch.cat((context, common_query, extra_query), dim=1)
    one_dict, _ = model._build_x_preprocess_inputs(one, 4)
    many_dict, _ = model._build_x_preprocess_inputs(many, 4)
    one_preprocessed = model.process_4_x(model.x_preprocess(one_dict))["data"]
    many_preprocessed = model.process_4_x(model.x_preprocess(many_dict))["data"]
    torch.testing.assert_close(one_preprocessed[:, 4], many_preprocessed[:, 4])

    with torch.no_grad():
        one_prediction = model(one, y_train, 4, task_type="reg")
        many_predictions = model(many, y_train, 4, task_type="reg")
    torch.testing.assert_close(one_prediction, many_predictions[:, :1], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(("num_features", "expected_padding"), [(4, 2), (5, 1)])
def test_features_per_group_three_padding_matches_direct_and_cache(num_features, expected_padding):
    model = _tiny_model(features_per_group=3)
    values = torch.arange(7 * num_features, dtype=torch.float32).reshape(1, 7, num_features)
    values = values + torch.arange(num_features, dtype=torch.float32).square()
    y_train = torch.tensor([[0.0, 1.0, -1.0, 2.0]])

    grouped, padding = model._build_x_preprocess_inputs(values, 4)
    assert padding == expected_padding
    assert grouped["data"].shape[-2:] == (2, 3)

    with torch.no_grad():
        direct = model(values, y_train, 4, task_type="reg")
        context = model.build_context_cache(values[:, :4], y_train)
        cached = model.apply_context_cache(values[:, 4:], context)
    assert context.valid_feature_num is not None
    torch.testing.assert_close(cached, direct, atol=1e-6, rtol=1e-6)


def test_nan_encoder_excludes_every_nonfinite_value_and_keeps_signed_indicator():
    encoder = NanEncoder()
    values = torch.tensor([[
        [1.0, float("inf"), 1.0],
        [3.0, float("-inf"), 2.0],
        [float("inf"), float("nan"), 3.0],
        [float("nan"), float("nan"), 4.0],
    ]])

    result = encoder({"data": values, "eval_pos": 4})
    torch.testing.assert_close(result["_nan_mean"], torch.tensor([[2.0, 0.0, 2.5]]))
    assert torch.isfinite(result["data"]).all()
    assert result["nan_encoding"][0, 0, 1].item() == encoder.inf_value
    assert result["nan_encoding"][0, 1, 1].item() == encoder.neg_info_value
    assert result["nan_encoding"][0, 2, 1].item() == encoder.nan_value


@pytest.mark.parametrize(
    ("nan_handling_enabled", "use_nan_indicator"),
    [(True, False), (True, True), (False, False)],
)
def test_nonfinite_values_are_missing_and_direct_matches_cache(
    nan_handling_enabled,
    use_nan_indicator,
):
    model = _tiny_model(
        nan_handling_enabled=nan_handling_enabled,
        nan_handling_y_encoder=nan_handling_enabled,
        use_nan_indicator=use_nan_indicator,
    )
    values = torch.tensor([[
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, float("inf"), 4.0],
        [2.0, float("nan"), 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, float("-inf"), 6.0, 7.0],
        [5.0, 6.0, 7.0, float("inf")],
    ]])
    y_train = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    inputs, _ = model._build_x_preprocess_inputs(values, 4)
    # NaN, +Inf and -Inf are all represented by the original-value mask.
    assert inputs["mask"].sum().item() == 4
    restored = model.process_4_x(model.x_preprocess(inputs))["data"]
    assert torch.isnan(restored).sum().item() == 4
    if use_nan_indicator:
        assert model.encoder_x[0].in_keys == ["data", "nan_encoding"]

    with torch.no_grad():
        direct = model(values, y_train, 4, task_type="reg")
        context = model.build_context_cache(values[:, :4], y_train)
        cached = model.apply_context_cache(values[:, 4:], context)
    assert torch.isfinite(direct).all()
    assert torch.isfinite(cached).all()
    torch.testing.assert_close(cached, direct, atol=1e-6, rtol=1e-6)


def test_rbf_centers_are_deterministic_and_reconstructed_from_config():
    def make_embedding():
        return RBFembedding(
            embedding_size=4,
            exponent_digits=0,
            token_embed_dim=2,
            n_kernels=5,
            sigma=0.5,
            as_tokenizer=True,
        )

    torch.manual_seed(1)
    source = make_embedding()
    state = source.state_dict()
    assert "centers" not in state
    torch.testing.assert_close(
        source.centers,
        torch.linspace(0.0, 10.0, steps=5, dtype=source.centers.dtype),
    )

    torch.manual_seed(2)
    restored = make_embedding()
    torch.testing.assert_close(restored.centers, source.centers)
    restored.load_state_dict(state, strict=True)
    sample = torch.tensor([[[0.25], [1.5]]])
    torch.testing.assert_close(restored(sample), source(sample))


def test_rbf_strict_load_ignores_only_transient_persisted_centers():
    def make_embedding():
        return RBFembedding(
            embedding_size=4,
            exponent_digits=0,
            token_embed_dim=2,
            n_kernels=5,
            sigma=0.5,
            as_tokenizer=True,
        )

    source = make_embedding()
    transient_state = source.state_dict()
    transient_state["centers"] = source.centers.clone()

    restored = make_embedding()
    incompatible = restored.load_state_dict(transient_state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(
        restored.centers,
        torch.linspace(0.0, 10.0, steps=5, dtype=restored.centers.dtype),
    )

    transient_state["centers"] = torch.full_like(source.centers, -123.0)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        restored.load_state_dict(transient_state, strict=True)

    transient_state = source.state_dict()
    transient_state["centers"] = source.centers.clone()
    transient_state["other_legacy_buffer"] = torch.tensor(1.0)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        restored.load_state_dict(transient_state, strict=True)


def test_legacy_random_rbf_config_is_ignored():
    model = _tiny_model(legacy_random_rbf_flag=True)
    centers = model.encoder_x[0].numeric_mlp.centers
    torch.testing.assert_close(
        centers,
        torch.linspace(0.0, 10.0, steps=8, dtype=centers.dtype),
    )


def test_normalization_train_only_false_includes_the_final_row():
    encoder = NormalizationEncoder(
        train_only=False,
        normalize_x=True,
        remove_outliers=False,
    )
    values = torch.tensor([[[0.0], [2.0], [10.0]]])
    result = encoder({"data": values, "eval_pos": 1})
    assert result["_norm_mean"].item() == pytest.approx(4.0)
    assert result["data"].mean().item() == pytest.approx(0.0, abs=1e-6)


def test_context_cache_rejects_transductive_normalization():
    model = _tiny_model(normalize_on_train_only=False)
    values = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    y_train = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    with pytest.raises(NotImplementedError, match="uncached forward path"):
        model.build_context_cache(values[:, :4], y_train)


def test_y_encoder_without_nan_handling_is_finite_for_masked_queries():
    encoder = get_reg_y_encoder(
        num_inputs=1,
        embedding_size=1,
        nan_handling_y_encoder=False,
    )
    linear = encoder[0].layer
    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.zero_()
    values = torch.tensor([[[1.0], [float("nan")], [float("inf")], [float("-inf")]]])
    result = encoder({"data": values, "eval_pos": 1})["data"]
    torch.testing.assert_close(result, torch.tensor([[[1.0], [0.0], [0.0], [0.0]]]))


def test_nan_handling_disabled_has_no_nan_stage_and_mask_metadata_still_works():
    model = _tiny_model(
        mask_prediction=True,
        nan_handling_enabled=False,
        nan_handling_y_encoder=False,
    )
    assert not any(isinstance(stage, NanEncoder) for stage in model.x_preprocess)
    values = torch.tensor([[
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 1.0, 2.0, 4.0],
        [2.0, float("inf"), 2.0, 5.0],
        [3.0, 9.0, 8.0, 6.0],
        [4.0, float("-inf"), 6.0, 7.0],
    ]])
    y_train = torch.tensor([[0.0, 1.0, 2.0]])

    with torch.no_grad():
        result = model(values, y_train, 3, task_type="reg")
    assert torch.isfinite(result["reg_output"]).all()
    process_config = result["process_config"]
    assert process_config["n_x_padding"] == 2
    assert process_config["features_per_group"] == 3
    assert process_config["num_used_features"] is not None
    assert process_config["num_used_features"][0, 0, 0].item() == 1
    assert process_config["mean_for_normalization"] is not None
    assert process_config["std_for_normalization"] is not None


def test_mask_embedding_nan_to_zero_flag_uses_the_numeric_zero_path():
    encoder = MaskEmbEncoder(
        num_features=1,
        emsize=4,
        mask_embedding_size=4,
        nan_to_zero=True,
        RBF_config={
            "token_embed_dim": 2,
            "n_kernels": 4,
            "sigma": 0.5,
            "use_learn_sigma": False,
            "use_learn_embeddings": False,
            "use_original_features": False,
        },
    )
    values = torch.tensor([[[[float("nan")]], [[0.0]]]])
    result = encoder({
        "data": values,
        "nan_encoding": torch.zeros_like(values),
        "eval_pos": 1,
    })["data"]
    torch.testing.assert_close(result[:, 0], result[:, 1])
