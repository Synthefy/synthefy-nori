"""NoriCausal: network invariances, graph decoding, estimator modes and NoriInterpreter integration."""

from __future__ import annotations


import numpy as np
import pytest
import torch

from synthefy_nori.causal import NoriCausal, get_decoded_graph
from synthefy_nori.causal._network import NoriCausalNetwork


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """A small randomly initialised NoriCausal checkpoint in the training-checkpoint format."""
    torch.manual_seed(0)
    net = NoriCausalNetwork(d=32, r=16, nhead=2, n_blocks=1, n_pooled_blocks=1, max_rows=128)
    path = tmp_path_factory.mktemp("noricausal") / "tiny.pt"
    torch.save(
        {
            "state_dict": net.state_dict(),
            "d": 32,
            "r": 16,
            "nhead": 2,
            "n_blocks": 1,
            "n_pooled_blocks": 1,
            "max_rows": 128,
            "step": 0,
        },
        path,
    )
    return path


def _table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 2 * a + rng.normal(size=n)
    c = np.tanh(b) + rng.normal(size=n) * 0.3
    return np.column_stack([a, b, c]).astype(np.float32), (c + rng.normal(size=n) * 0.2).astype(np.float32)


def test_network_shapes_and_row_order_invariance(tiny_checkpoint):
    net, meta = NoriCausalNetwork.from_checkpoint(tiny_checkpoint)
    X, y = _table(n=100)  # below max_rows (128): above it rows are evenly subsampled, which depends on order
    with torch.no_grad():
        target_logits = net(torch.as_tensor(X), torch.as_tensor(y))
        perm = torch.randperm(len(y))
        permuted_logits = net(torch.as_tensor(X)[perm], torch.as_tensor(y)[perm])
        null_logits = net(torch.as_tensor(np.column_stack([X, y])), torch.zeros(len(y)), null_target=True)
    assert target_logits.shape == (4, 4) and null_logits.shape == (5, 5) and meta["d"] == 32
    assert torch.allclose(target_logits, permuted_logits, atol=1e-4)  # rows are a set


def test_checkpoint_missing_a_parameter_is_rejected(tiny_checkpoint, tmp_path):
    ck = torch.load(tiny_checkpoint, weights_only=False)
    ck["state_dict"].pop("head.A.weight")
    bad = tmp_path / "bad.pt"
    torch.save(ck, bad)
    with pytest.raises(ValueError, match="missing"):
        NoriCausalNetwork.from_checkpoint(bad)


def test_get_decoded_graph_thresholds_and_breaks_two_cycles():
    probabilities = np.array([[0.0, 0.9, 0.2], [0.95, 0.0, 0.8], [0.1, 0.7, 0.0]])
    graph = get_decoded_graph(probabilities, 0.75)
    assert graph.tolist() == [[0, 0, 0], [1, 0, 1], [0, 0, 0]]  # 1->0 beats 0->1; 1->2 kept; 2->1 below cut
    tie = get_decoded_graph(np.array([[0.0, 0.9], [0.9, 0.0]]), 0.5)
    assert tie.tolist() == [[0, 1], [0, 0]]  # exact tie keeps i -> j for i < j


@pytest.mark.parametrize("mode", ["auto", "subgraph", "target", "null"])
def test_estimator_modes(tiny_checkpoint, mode):
    X, y = _table()
    causal = NoriCausal(
        checkpoint=tiny_checkpoint, model="custom", mode=mode, n_subsets=6, window=3, threshold=0.5, device="cpu"
    )
    causal.fit(X, y, feature_names=["a", "b", "c"], target_name="y")
    n_nodes = 4
    assert causal.node_names_ == ["a", "b", "c", "y"] and causal.target_index_ == 3
    assert causal.adjacency_.shape == (n_nodes, n_nodes) and np.allclose(np.diag(causal.adjacency_), 0)
    assert ((causal.graph_ == 1) & (causal.graph_.T == 1)).sum() == 0
    assert all(set(e) == {"source", "target", "probability"} for e in causal.edges_)
    assert set(causal.parents("y")) <= set(causal.node_names_) and set(causal.children(0)) <= set(causal.node_names_)
    if mode == "subgraph":
        assert causal.pair_coverage_.shape == (n_nodes, n_nodes)


def test_estimator_without_y_and_input_checks(tiny_checkpoint):
    X, _ = _table()
    causal = NoriCausal(checkpoint=tiny_checkpoint, model="custom", mode="null", threshold=0.5, device="cpu").fit(X)
    assert causal.target_index_ is None and causal.adjacency_.shape == (3, 3)
    with pytest.raises(ValueError, match="needs y"):
        NoriCausal(checkpoint=tiny_checkpoint, model="custom", mode="target", threshold=0.5, device="cpu").fit(X)
    with pytest.raises(ValueError, match="calibrated"):
        NoriCausal(checkpoint=tiny_checkpoint, model="custom", mode="null", threshold="calibrated", device="cpu").fit(X)
    causal = NoriCausal(checkpoint=tiny_checkpoint, model="custom", mode="null", device="cpu").fit(X)
    assert causal.graph_ is None and causal.threshold_ is None and causal.adjacency_.shape == (3, 3)  # default
    with pytest.raises(ValueError, match="threshold=None"):
        causal.parents(0)


def test_named_model_without_local_checkpoint_downloads_from_the_hub(monkeypatch, tiny_checkpoint):
    import synthefy_nori.hf as hf
    from synthefy_nori.causal import get_checkpoint_path_and_threshold

    calls = []
    monkeypatch.delenv("SYNTHEFY_NORICAUSAL_31M_CHECKPOINT", raising=False)
    monkeypatch.setattr(hf, "download_checkpoint", lambda **kw: calls.append(kw) or str(tiny_checkpoint))
    path, threshold = get_checkpoint_path_and_threshold("noricausal-31m")
    assert calls == [{"repo_id": "Synthefy/Nori-Causal", "filename": "noricausal_31m.pt"}]
    assert path == tiny_checkpoint and threshold is not None
    monkeypatch.setenv("SYNTHEFY_NORICAUSAL_31M_CHECKPOINT", str(tiny_checkpoint))
    get_checkpoint_path_and_threshold("noricausal-31m")
    assert len(calls) == 1  # a local checkpoint skips the download


def test_interpreter_causal_attributes(tiny_checkpoint):
    pytest.importorskip("interpret")
    from sklearn.linear_model import Ridge

    from synthefy_nori.interpretability.glassbox import NoriInterpreter

    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    y = X[:, 0] - 2 * X[:, 1] + rng.normal(size=300) * 0.1
    interp = NoriInterpreter(
        model=Ridge(),
        reduce_threshold=2,
        render_figure=False,
        causal=True,
        causal_checkpoint=tiny_checkpoint,
        causal_model="custom",  # no calibrated threshold: the interpreter must not need one
        causal_mode="target",
        target_name="y",
    )
    interp.fit(X, y, feature_names=["w", "x", "z", "v"])
    A = interp.causal_adjacency_
    assert interp.causal_node_names_ == ["w", "x", "z", "v", "y"] and A.shape == (5, 5)
    np.testing.assert_array_equal(interp.causal_cause_probabilities_, A[:4, 4])
    np.testing.assert_array_equal(interp.causal_effect_probabilities_, A[4, :4])
    p = [r["p_cause"] for r in interp.causal_ranking_]
    assert sorted(r["feature"] for r in interp.causal_ranking_) == ["v", "w", "x", "z"] and p == sorted(p, reverse=True)
    assert all(r["target_coverage"] == 1 for r in interp.causal_ranking_)  # one target-aware pass
    assert interp.summary()["causal_top_causes"] == [r["feature"] for r in interp.causal_ranking_]
    assert not hasattr(interp, "causal_graph_")
    labels = (y > 0).astype(int)
    with pytest.raises(ValueError, match="regression targets only"):
        NoriInterpreter(model=Ridge(), causal=True, causal_checkpoint=tiny_checkpoint, causal_model="custom").fit(
            X, labels
        )
    assert NoriInterpreter().causal is False  # off by default


class _RecordingNet:
    """Stub network: records each pass's target column and returns zero logits of the right size."""

    def __init__(self):
        self.targets = []

    def __call__(self, X, y, null_target=False):
        self.targets.append(y.numpy().copy())
        return torch.zeros(X.shape[1] + 1, X.shape[1] + 1)


def test_auto_mode_picks_one_pass_or_target_anchored_subgraph(monkeypatch, tiny_checkpoint):
    import synthefy_nori.causal.estimator as est

    rng = np.random.default_rng(0)
    y = rng.normal(size=60).astype(np.float32)
    small = rng.normal(size=(60, 5)).astype(np.float32)
    wide = rng.normal(size=(60, 12)).astype(np.float32)
    for X, expected_mode in ((small, "target"), (wide, "target_subgraph")):
        net = _RecordingNet()
        monkeypatch.setattr(est, "_get_cached_network", lambda path, device, net=net: (net, {}))
        causal = NoriCausal(
            checkpoint=tiny_checkpoint,
            model="custom",
            threshold=0.5,
            device="cpu",
            target_max_features=8,
            n_subsets=20,
            window=6,
        ).fit(X, y)
        assert causal.mode_ == expected_mode
        is_y = [np.array_equal(target, y) for target in net.targets]
        if expected_mode == "target":
            assert is_y == [True]
        else:
            target_index = causal.target_index_
            # y sits in the target slot of exactly the subsets that contain it
            assert sum(is_y) == causal.pair_coverage_[target_index, target_index] and 0 < sum(is_y) < len(is_y)


def test_auto_without_y_falls_back_to_random_target_subgraph(tiny_checkpoint):
    X, _ = _table()
    causal = NoriCausal(
        checkpoint=tiny_checkpoint, model="custom", threshold=0.5, device="cpu", n_subsets=4, window=2
    ).fit(X)
    assert causal.mode_ == "subgraph"


def test_target_voting_reads_every_feature_with_y_and_floors_uncovered_pairs(monkeypatch, tiny_checkpoint):
    import synthefy_nori.causal.estimator as est

    net = _RecordingNet()
    monkeypatch.setattr(est, "_get_cached_network", lambda path, device: (net, {}))
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 30)).astype(np.float32), rng.normal(size=40).astype(np.float32)
    causal = NoriCausal(
        checkpoint=tiny_checkpoint,
        model="custom",
        threshold=None,
        device="cpu",
        target_max_features=8,
        n_subsets=2,
        window=5,
        min_target_coverage=3,
    ).fit(X, y)
    coverage = causal.pair_coverage_
    target_index = causal.target_index_
    assert causal.mode_ == "target_subgraph" and coverage[:target_index, target_index].min() >= 3
    # coverage-driven passes always put y in the target slot
    assert sum(np.array_equal(target, y) for target in net.targets) == coverage[target_index, target_index]
    uncovered = coverage == 0
    assert uncovered.any() and (causal.adjacency_[uncovered] < 1e-6).all()
    assert np.allclose(causal.adjacency_[(coverage > 0) & ~np.eye(len(coverage), dtype=bool)], 0.5)  # zero logits
