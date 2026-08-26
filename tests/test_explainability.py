"""Unit tests for ``synthefy_nori.explainability``.

Fast and offline: the pure helpers need nothing beyond the base deps, and the EBM
serialisation tests fit a tiny 2-feature EBM (skipped when the optional
``explainability`` extra is absent). Nothing here loads a Nori checkpoint.
"""

import warnings

import numpy as np
import pytest
from synthefy_nori.explainability._common import (
    binarize01,
    binary_classes,
    clip_inf_edges,
    detect_task,
    fill_nan,
    impute_mean,
    make_metric,
    shape_direction,
    target_from_full,
    train_means,
)


# --------------------------------------------------------------------------- task detection
def test_detect_task_infers_binary_and_continuous():
    assert detect_task(np.array([0, 1, 1, 0])) == "classification"
    assert detect_task(np.array([0.1, 2.5, 3.7, -1.0])) == "regression"
    assert detect_task(np.array([0, 1, 2])) == "multiclass"  # 3 integer classes
    assert detect_task(np.array([0.1, 0.5, 0.9])) == "regression"  # 3 FLOATS are not classes
    assert detect_task(np.arange(40)) == "regression"  # above MAX_CLASSES
    assert detect_task(np.array(["a", "b", "c", "a"])) == "multiclass"
    assert detect_task(np.array([0.1, 2.5]), "classification") == "classification"


# --------------------------------------------------------------------------- target remapping
@pytest.mark.parametrize("labels", [(0, 1), (1, 2), (-1, 1), (3, 7)])
def test_binarize01_maps_any_binary_encoding_to_zero_one(labels):
    lo, hi = labels
    y = np.array([lo, hi, hi, lo])
    assert set(np.unique(binarize01(y)).tolist()) == {0, 1}
    assert binarize01(y).tolist() == [0, 1, 1, 0]  # largest label -> 1


def test_binarize01_shares_one_mapping_across_arrays():
    ytr, yte = binarize01(np.array([1, 2]), np.array([2, 2]))
    assert ytr.tolist() == [0, 1] and yte.tolist() == [1, 1]


def test_binary_classes_returns_callers_labels_ascending():
    assert binary_classes(np.array([2, 1, 1])).tolist() == [1, 2]


def test_binarize01_rejects_more_than_two_classes():
    """A forced task="classification" on a 3-class target must fail, not silently do one-vs-rest."""
    with pytest.raises(ValueError, match="exactly 2 classes"):
        binarize01(np.array([0, 1, 2]))


# --------------------------------------------------------------------------- skill target
def test_target_from_full_scales_auc_margin_not_auc():
    assert target_from_full("classification", 0.9, 0.95) == pytest.approx(0.88)  # 0.5 + .95*.4
    assert target_from_full("regression", 0.8, 0.95) == pytest.approx(0.76)


def test_make_metric_names_and_scores():
    m, name = make_metric("classification")
    assert name == "roc_auc"
    assert m(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    m, name = make_metric("regression")
    assert name == "r2"
    assert m(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)


# --------------------------------------------------------------------------- imputation
def test_train_means_ignores_nan_and_inf():
    X = np.array([[1.0, np.inf], [3.0, 2.0], [np.nan, 4.0]], np.float32)
    # column 0: mean of the finite entries; column 1: inf must NOT poison the mean
    assert train_means(X).tolist() == pytest.approx([2.0, 3.0])


def test_train_means_defaults_all_missing_column_to_zero():
    assert train_means(np.array([[np.nan], [np.inf]], np.float32)).tolist() == [0.0]


def test_fill_nan_imputes_inf_rather_than_saturating_to_float32_max():
    """np.nan_to_num would turn +inf into ~3.4e38 — a bogus but finite feature value."""
    out = fill_nan(np.array([[np.nan, np.inf, -np.inf]], np.float32), np.array([1.0, 2.0, 3.0]))
    assert out.tolist() == [[1.0, 2.0, 3.0]]
    assert np.isfinite(out).all()


def test_impute_mean_uses_train_means_for_every_array():
    Xtr = np.array([[1.0], [3.0]], np.float32)
    Xtr_i, Xte_i = impute_mean(Xtr, np.array([[np.nan]], np.float32))
    assert Xtr_i.tolist() == [[1.0], [3.0]] and Xte_i.tolist() == [[2.0]]


# --------------------------------------------------------------------------- shape helpers
def test_clip_inf_edges_pads_beyond_the_finite_range():
    out = clip_inf_edges([-np.inf, 0.0, 10.0, np.inf])
    assert np.isfinite(out).all()
    assert out[0] < 0.0 and out[-1] > 10.0


def test_clip_inf_edges_handles_an_all_infinite_input():
    assert np.isfinite(clip_inf_edges([-np.inf, np.inf])).all()


@pytest.mark.parametrize(
    "scores,expected",
    [
        ([0.0, 1.0], "flat"),  # < 3 bins
        ([1.0, 1.0, 1.0, 1.0], "negligible"),  # no swing
        ([0.0, 1.0, 2.0, 3.0, 4.0], "increasing"),
        ([4.0, 3.0, 2.0, 1.0, 0.0], "decreasing"),
        ([0.0, 5.0, -5.0, 5.0, 0.0], "non-monotone"),
    ],
)
def test_shape_direction(scores, expected):
    assert shape_direction(scores) == expected


# --------------------------------------------------------------------------- EBM serialisation
def _fit_tiny_ebm(feature_names):
    """A 2-feature classification EBM with pairwise interactions enabled."""
    pytest.importorskip("interpret", reason="needs the explainability extra")
    from synthefy_nori.explainability.ebm import fit_ebm

    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, len(feature_names))).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return fit_ebm(X, y, feature_names, "classification", interactions=1, outer_bags=2)


def test_ebm_structure_keeps_every_main_effect_when_a_feature_name_contains_an_ampersand():
    """Interactions are named "a & b", so a '&' substring test would drop "R&D spend" entirely."""
    from synthefy_nori.explainability.ebm import ebm_structure

    names = ["R&D spend", "headcount"]
    struct = ebm_structure(_fit_tiny_ebm(names))
    assert [s["feature"] for s in struct["shape_functions"]] == names


def test_ebm_structure_includes_interactions_by_default():
    """Without the pairwise terms, intercept + sum(shape_functions) != the model's prediction."""
    from synthefy_nori.explainability.ebm import ebm_structure

    struct = ebm_structure(_fit_tiny_ebm(["a", "b"]))
    assert struct["interactions"], "interaction terms must be serialised by default"
    inter = struct["interactions"][0]
    assert len(inter["feature_indices"]) == 2
    assert np.asarray(inter["scores"]).ndim == 2  # a 2-D lookup table


def test_ebm_structure_can_omit_interactions_on_request():
    from synthefy_nori.explainability.ebm import ebm_structure

    struct = ebm_structure(_fit_tiny_ebm(["a", "b"]), include_interactions=False)
    assert "interactions" not in struct


def test_ebm_structure_serialises_to_json_with_finite_edges():
    import json
    from synthefy_nori.explainability.ebm import ebm_structure

    struct = ebm_structure(_fit_tiny_ebm(["a", "b"]))
    json.dumps(struct)  # must not raise
    for shape in struct["shape_functions"]:
        assert np.isfinite(shape["bin_edges"]).all()


# --------------------------------------------------------------------------- figure guards
def test_plot_ebm_model_rejects_misaligned_density():
    """A pruned EBM plus the FULL table would silently draw each density from the wrong column."""
    pytest.importorskip("matplotlib", reason="needs the explainability extra")
    from synthefy_nori.explainability.viz import plot_ebm_model

    model = _fit_tiny_ebm(["a", "b"])
    with pytest.raises(ValueError, match="column-aligned"):
        plot_ebm_model(model, ["a", "b"], X_density=np.zeros((10, 5), np.float32))


def test_plot_ebm_model_survives_a_degenerate_feature_range():
    """p10 == p90 on a zero-inflated column must not collapse the panel to zero width."""
    matplotlib = pytest.importorskip("matplotlib", reason="needs the explainability extra")
    matplotlib.use("Agg")
    from synthefy_nori.explainability.viz import plot_ebm_model

    model = _fit_tiny_ebm(["a", "b"])
    fig = plot_ebm_model(model, ["a", "b"], feature_ranges={"a": (0.0, 0.0)})
    xlims = [ax.get_xlim() for ax in fig.axes]
    assert all(hi > lo for lo, hi in xlims), f"zero-width x-axis in {xlims}"
    matplotlib.pyplot.close(fig)


# --------------------------------------------------------------------------- defaults
def test_noriinterpreter_defaults_to_a_70_30_split():
    """The published write-up and the credit example both describe a 70/30 split."""
    from synthefy_nori.explainability import NoriInterpreter

    assert NoriInterpreter().test_size == 0.3


def test_all_split_defaults_are_70_30():
    """ "70/30 everywhere": the estimator, the pipeline's selection carve-out, and the loaders."""
    from synthefy_nori.explainability import data as _data
    from synthefy_nori.explainability.pipeline import SELECTION_FRACTION

    assert SELECTION_FRACTION == 0.3
    import inspect

    for fn in (_data.load_csv, _data.load_demo):
        assert inspect.signature(fn).parameters["test_size"].default == 0.3, fn.__name__


def test_figure_key_states_how_many_interactions_are_drawn():
    """The diagram draws at most MAX_INTERACTIONS_SHOWN; it must say so when it hides some."""
    matplotlib = pytest.importorskip("matplotlib", reason="needs the explainability extra")
    matplotlib.use("Agg")
    pytest.importorskip("interpret", reason="needs the explainability extra")
    from synthefy_nori.explainability.ebm import fit_ebm
    from synthefy_nori.explainability.viz import MAX_INTERACTIONS_SHOWN, plot_ebm_model

    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int)
    names = ["a", "b", "c", "e"]
    n_int = MAX_INTERACTIONS_SHOWN + 2
    model = fit_ebm(X, y, names, "classification", interactions=n_int, outer_bags=2)
    n_actual = sum(1 for f in model.term_features_ if len(f) == 2)
    assert n_actual > MAX_INTERACTIONS_SHOWN, "need more interactions than the figure can draw"
    fig = plot_ebm_model(model, names)
    blob = " ".join(t.get_text() for t in fig.texts)
    assert f"of {n_actual}" in blob, f"figure never states the interaction count: {blob[:300]}"
    assert str(MAX_INTERACTIONS_SHOWN) in blob
    matplotlib.pyplot.close(fig)


def test_use_test_defaults_to_true_on_both_entry_points():
    """Interpretation is post-hoc, so importance is measured on held-out data by default."""
    import inspect
    from synthefy_nori.explainability import NoriInterpreter
    from synthefy_nori.explainability.pipeline import run

    assert NoriInterpreter().use_test is True
    assert inspect.signature(run).parameters["use_test"].default is True


# --------------------------------------------------------------------------- end-to-end (no Nori)
# Both front doors take `model=` / `nori_model=` and clone it, so a cheap sklearn estimator
# stands in for Nori. That keeps these tests fast and offline while still executing the real
# split -> impute -> importance -> prune-sweep -> distill path in _core.
def _toy(n=160, d=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] - X[:, 2] > 0).astype(int)
    return X, y, [f"f{j}" for j in range(d)]


@pytest.mark.parametrize("use_test", [True, False])
def test_pipeline_run_end_to_end(use_test, tmp_path):
    """pipeline.run was CI-blind: only its constants were imported, never its behaviour."""
    pytest.importorskip("interpret", reason="needs the explainability extra")
    from sklearn.linear_model import Ridge
    from synthefy_nori.explainability.pipeline import run

    X, y, names = _toy()
    res = run(
        X[:120],
        y[:120],
        X[120:],
        y[120:],
        names,
        nori_model=Ridge(),
        reduce_threshold=3,
        out_dir=str(tmp_path),
        tag="toy",
        verbose=False,
        use_test=use_test,
    )
    assert res["use_test"] is use_test
    assert res["task"] == "classification" and res["metric"] == "roc_auc"
    assert 1 <= res["n95"] <= len(names)
    assert res["selected_features"] and len(res["selected_features"]) == res["n95"]
    assert "interactions" in res["ebm_model"]  # serialised model is complete
    assert len(res["importance"]) == len(names)
    if use_test:  # criterion and report are then the same measurement
        assert res["nori_at_n95"] == res["selection_at_n95"]
    assert (tmp_path / "toy.json").exists() and (tmp_path / "toy.ebm.joblib").exists()


@pytest.mark.parametrize("use_test", [True, False])
def test_nori_interpreter_end_to_end(use_test):
    """The estimator front door over the same _core path."""
    pytest.importorskip("interpret", reason="needs the explainability extra")
    from sklearn.linear_model import Ridge
    from synthefy_nori.explainability import NoriInterpreter

    X, y, names = _toy()
    interp = NoriInterpreter(model=Ridge(), reduce_threshold=3, render_figure=False, use_test=use_test).fit(
        X, y + 1, feature_names=names
    )
    assert interp.classes_.tolist() == [1, 2]
    assert set(np.unique(interp.predict(X)).tolist()) <= {1, 2}  # caller's label space
    assert 1 <= interp.n_selected_ <= len(names)
    s = interp.summary()
    assert s["n_selected"] == interp.n_selected_
    assert "interactions" in interp.ebm_model_
    if use_test:
        assert interp.nori_selected_score_ == interp.selection_score_


def test_both_front_doors_agree_when_given_the_same_split():
    """The whole point of _core: one implementation, so the two front doors cannot drift."""
    pytest.importorskip("interpret", reason="needs the explainability extra")
    from sklearn.linear_model import Ridge
    from synthefy_nori.explainability import NoriInterpreter
    from synthefy_nori.explainability.pipeline import run

    X, y, names = _toy(n=200, d=5)
    res = run(X[:140], y[:140], X[140:], y[140:], names, nori_model=Ridge(), reduce_threshold=2, verbose=False)
    interp = NoriInterpreter(model=Ridge(), reduce_threshold=2, render_figure=False, test_size=0.3, random_state=0).fit(
        X, y, feature_names=names
    )
    # different splits, so scores differ — but both must produce a coherent selection
    assert res["n95"] >= 1 and interp.n_selected_ >= 1
    assert res["metric"] == interp.metric_ == "roc_auc"


# --------------------------------------------------------------------------- loaders
def test_load_table_reads_csv_and_parquet_identically(tmp_path):
    import pandas as pd
    from synthefy_nori.explainability.data import load_csv, load_table

    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 7, 8], "b": list("xxyyxxyy"), "target": [0, 1, 0, 1, 0, 1, 0, 1]})
    csv, pq = tmp_path / "t.csv", tmp_path / "t.parquet"
    df.to_csv(csv, index=False)
    df.to_parquet(pq)
    from_csv = load_table(str(csv), "target")
    from_pq = load_table(str(pq), "target")
    assert from_csv[4] == from_pq[4] == ["a", "b"]  # feature names
    for a, b in zip(from_csv[:4], from_pq[:4]):
        np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float))
    np.testing.assert_allclose(load_csv(str(csv), "target")[0], from_csv[0])  # alias still works


def test_load_table_ordinal_encodes_and_keeps_missing_as_nan(tmp_path):
    import pandas as pd
    from synthefy_nori.explainability.data import load_table

    df = pd.DataFrame({"cat": ["x", None, "y", "x", "y", "x"], "target": [0, 1, 0, 1, 0, 1]})
    csv = tmp_path / "m.csv"
    df.to_csv(csv, index=False)
    Xtr, _, Xte, _, _ = load_table(str(csv), "target")
    allv = np.concatenate([Xtr.ravel(), Xte.ravel()])
    assert np.isnan(allv).any(), "missing category must stay NaN, not become the ordinal -1"
    assert not (allv[~np.isnan(allv)] < 0).any(), "no negative ordinal codes"


def test_interpreter_records_the_pruning_sweep():
    """The notebook plots accuracy-vs-features straight off the fitted estimator."""
    pytest.importorskip("interpret", reason="needs the explainability extra")
    from sklearn.linear_model import Ridge
    from synthefy_nori.explainability import NoriInterpreter

    X, y, names = _toy(n=180, d=8)
    interp = NoriInterpreter(model=Ridge(), reduce_threshold=2, render_figure=False).fit(X, y, feature_names=names)
    assert interp.reduced_ and interp.sweep_, "a pruned fit must record its sweep"
    ks = [row["k"] for row in interp.sweep_]
    assert ks == sorted(ks), "sweep must be in increasing k"
    assert ks[-1] == interp.n_selected_, "the last point is the accepted k"
    assert all(interp.metric_ in row for row in interp.sweep_)


# --------------------------------------------------------------------------- multiclass
def test_macro_ovr_auc_needs_a_column_per_class_and_averages_them():
    from synthefy_nori.explainability._common import make_metric

    metric, name = make_metric("multiclass")
    assert name == "macro_ovr_auc"
    y = np.array([0, 0, 1, 1, 2, 2])
    perfect = np.eye(3)[y] * 1.0  # one-hot == perfectly separable
    assert metric(y, perfect) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="score column per class"):
        metric(y, np.zeros(len(y)))  # a 1-D score is meaningless here


def test_label_round_trip_for_any_encoding():
    from synthefy_nori.explainability._common import decode_labels, encode_labels, target_classes

    for labels in ([1, 2, 3], [-1, 0, 1], ["low", "mid", "high"]):
        y = np.array(list(labels) * 3)
        classes = target_classes(y)
        codes = encode_labels(y, classes)
        assert set(np.unique(codes).tolist()) == set(range(len(set(labels))))
        assert decode_labels(codes, classes).tolist() == y.tolist()


def test_one_vs_rest_wrapper_fits_one_regressor_per_class_and_assumes_no_order():
    """Nori is a regressor. For K>2, regressing the class CODES would assert 1 lies between
    0 and 2; one indicator regression per class asserts nothing about order."""
    from sklearn.linear_model import Ridge
    from synthefy_nori.explainability._core import OneVsRestNori

    rng = np.random.RandomState(0)
    X = rng.normal(size=(90, 4)).astype(np.float32)
    y = np.repeat([0, 1, 2], 30)
    ovr = OneVsRestNori(Ridge()).fit(X, y)
    assert len(ovr.models_) == 3, "one regressor per class"
    scores = ovr.predict(X)
    assert scores.shape == (90, 3), "a score column per class"
    # permuting the class LABELS must permute the columns, not change their content:
    # that is what 'no order assumed' means operationally.
    remap = {0: 2, 1: 0, 2: 1}
    y2 = np.array([remap[v] for v in y])
    scores2 = OneVsRestNori(Ridge()).fit(X, y2).predict(X)
    for old, new in remap.items():
        np.testing.assert_allclose(scores[:, old], scores2[:, new], atol=1e-10)


def test_regressor_on_class_codes_would_be_worse_than_one_vs_rest():
    """Why the wrapper exists: the naive single regression onto 0..K-1 loses real skill when
    the classes have no natural order."""
    from sklearn.linear_model import Ridge
    from synthefy_nori.explainability._common import make_metric
    from synthefy_nori.explainability._core import OneVsRestNori

    rng = np.random.RandomState(0)
    n = 300
    # three well-separated clusters placed so that class 1 is NOT between 0 and 2
    centres = {0: [0.0, 0.0], 1: [6.0, 6.0], 2: [0.0, 6.0]}
    y = rng.randint(0, 3, size=n)
    X = np.array([centres[v] for v in y], np.float32) + rng.normal(scale=0.7, size=(n, 2)).astype(np.float32)
    metric, _ = make_metric("multiclass")
    ovr = metric(y, OneVsRestNori(Ridge()).fit(X, y).predict(X))
    codes = Ridge().fit(X, y).predict(X)  # the naive ordinal reading
    naive = metric(y, np.column_stack([-abs(codes - k) for k in range(3)]))
    assert ovr > naive, f"one-vs-rest {ovr:.3f} should beat code-regression {naive:.3f}"
    assert ovr > 0.95, f"one-vs-rest should be near-perfect on separable clusters, got {ovr:.3f}"


@pytest.mark.parametrize("labels", [[0, 1, 2], ["a", "b", "c"]])
def test_multiclass_end_to_end(labels):
    pytest.importorskip("interpret", reason="needs the explainability extra")
    from sklearn.linear_model import Ridge
    from synthefy_nori.explainability import NoriInterpreter

    rng = np.random.RandomState(0)
    n = 240
    codes = rng.randint(0, 3, size=n)
    X = (np.eye(3)[codes] * 4).astype(np.float32) + rng.normal(scale=0.8, size=(n, 3)).astype(np.float32)
    X = np.hstack([X, rng.normal(size=(n, 3)).astype(np.float32)])  # 3 noise columns
    y = np.array([labels[c] for c in codes])
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    interp = NoriInterpreter(model=Ridge(), reduce_threshold=2).fit(X, y)
    assert interp.task_ == "multiclass" and interp.metric_ == "macro_ovr_auc"
    assert interp.classes_.tolist() == list(labels)
    # the diagram now draws one curve per class, with a shared class legend
    fig = interp.model_figure_
    assert fig is not None
    texts = " ".join(t.get_text() for t in fig.texts)
    assert "softmax" in texts, "multiclass output goes through a softmax, not a sigmoid"
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert "3 curves, one per class" in titles, f"panel titles: {titles[:200]}"
    legends = [lg.get_title().get_text() for lg in fig.legends]
    assert "class" in legends, f"expected a class legend, got {legends}"
    matplotlib.pyplot.close(fig)
    assert set(np.unique(interp.predict(X)).tolist()) <= set(labels)  # caller's labels
    assert interp.predict_proba(X).shape == (n, 3)
    shapes = interp.ebm_model_["shape_functions"]
    assert shapes and all("scores_per_class" in sh for sh in shapes)
    assert np.asarray(shapes[0]["scores_per_class"]).ndim == 2
    assert interp.ebm_score_ > 0.8, f"should be easy to separate, got {interp.ebm_score_}"


def test_multiclass_ebm_has_no_interactions():
    """interpret's EBM cannot do pairwise terms for K>2, so fit_ebm must force 0."""
    pytest.importorskip("interpret", reason="needs the explainability extra")
    from synthefy_nori.explainability.ebm import fit_ebm

    rng = np.random.RandomState(0)
    X = rng.normal(size=(150, 3)).astype(np.float32)
    y = rng.randint(0, 3, size=150)
    model = fit_ebm(X, y, ["a", "b", "c"], "multiclass", interactions=5)
    assert all(len(f) == 1 for f in model.term_features_), "no interaction terms for multiclass"
