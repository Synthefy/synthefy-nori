"""NoriCausal — zero-shot causal structure from a single observational table.

    from synthefy_nori.causal import NoriCausal
    causal = NoriCausal(checkpoint="noricausal_31m.pt").fit(X, y)    # y optional
    causal.adjacency_          # (T, T) edge probabilities, [i, j] = P(node i -> node j)

By default no cutoff is applied (``threshold=None``): the output is the probability matrix. With a
cutoff, e.g. ``NoriCausal(threshold=0.8)``, ``graph_`` (0/1, no two-cycles), ``edges_`` and
``parents()`` / ``children()`` are also set.

Nodes are the columns of ``X`` followed by ``y`` (when given). Inference modes:

* ``"auto"`` (default): with ``y`` and at most ``target_max_features`` features, one target-aware pass
  with ``y`` as the target; with ``y`` and more features, target-aware subgraph voting in which ``y``
  takes the target slot of every subset that contains it; without ``y``, ``"subgraph"``.
* ``"subgraph"``: ``n_subsets`` random subsets of at most ``window`` columns; each subset is read once
  with its last column as the model's target slot (``y`` gets no special role), and every pair's
  logits are averaged over the subsets that contain it. This is the protocol of the CausalArena
  evaluation; it costs ``n_subsets`` forward passes.
* Target-aware voting (``"auto"`` on a wide table) then adds target-anchored subsets (``y`` plus the
  ``window - 1`` features least often seen with it) until every feature has shared at least
  ``min_target_coverage`` subsets with ``y``, so each feature's score against ``y`` is a model reading.

In voting, a pair never sampled together gets ``UNCOVERED_LOGIT`` (probability ~0): it is never an
edge and ranks below every scored pair. ``pair_coverage_`` counts the subsets behind every pair.
* ``"target"``: one target-aware pass with ``y`` as the target (requires ``y``).
* ``"null"``: one pass with no privileged column.

Rows above the model's ``max_rows`` (2048) are evenly subsampled inside the network.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from synthefy_nori.causal.registry import get_checkpoint_path_and_threshold

_NETWORK_CACHE: dict = {}
# Logit of a pair that no voting subset contained: probability ~2e-9, below every scored pair.
UNCOVERED_LOGIT = -20.0


def _get_cached_network(path, device):
    """Return ``(network, checkpoint metadata)`` for ``path`` on ``device``, loading it on first use."""
    key = (str(path), str(device))
    if key not in _NETWORK_CACHE:
        from synthefy_nori.causal._network import NoriCausalNetwork

        _NETWORK_CACHE[key] = NoriCausalNetwork.from_checkpoint(path, device)
    return _NETWORK_CACHE[key]


def get_decoded_graph(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Return the 0/1 graph of edges above ``threshold``, keeping only the stronger direction of any two-cycle."""
    edge_probabilities = np.asarray(probabilities, np.float64)
    graph = edge_probabilities > float(threshold)
    np.fill_diagonal(graph, False)
    both_directions = graph & graph.T
    graph &= ~(both_directions & (edge_probabilities < edge_probabilities.T))  # drop the weaker direction
    exact_tie = both_directions & (edge_probabilities == edge_probabilities.T)
    graph &= ~np.tril(exact_tie)  # exact ties: keep i -> j for i < j (as the eval harness does)
    return graph.astype(np.uint8)


class NoriCausal(BaseEstimator):
    """Estimate the causal graph over the columns of a table (plus an optional target).

    Parameters
    ----------
    model : str, default="noricausal-31m"
        Registry name (see :data:`synthefy_nori.causal.registry.NORI_CAUSAL_MODELS`).
    checkpoint : str or path, optional
        Checkpoint file; overrides the registry entry's environment variable.
    mode : {"auto", "subgraph", "target", "null"}, default="auto"
    target_max_features : int, default=25
        ``"auto"`` with ``y``: up to this many features use one target-aware pass, more use
        target-anchored subgraph voting.
    n_subsets, window : int, default=64, 30
        Subgraph voting: number of random column subsets and their maximum size.
    min_target_coverage : int, default=3
        Target-aware voting: target-anchored subsets are added until every feature has shared at least
        this many subsets with ``y``.
    threshold : None, float or "calibrated", default=None
        Edge-probability cutoff for ``graph_``. ``None`` skips decoding: only ``logits_`` /
        ``adjacency_`` are set (``graph_`` is ``None``). "calibrated" uses the registry value, which was
        calibrated on synthetic graphs in null-target mode, so it does not match the target-aware modes.
    device : str, default="auto"
    random_state : int, default=0

    Attributes
    ----------
    node_names_ : list[str]
    target_index_ : int or None
        Index of ``y`` among the nodes (the last one) when ``y`` was given.
    mode_ : str
        The procedure actually run: "target", "target_subgraph", "subgraph" or "null".
    logits_, adjacency_ : np.ndarray, shape (T, T)
        Edge logits / probabilities, ``[i, j]`` for ``i -> j``; zero diagonal.
    pair_coverage_ : np.ndarray, shape (T, T)
        Number of forward passes behind each pair (1 everywhere for the one-pass modes; 0 = never
        compared, scored ``UNCOVERED_LOGIT``).
    graph_ : np.ndarray, shape (T, T)
        0/1 graph at ``threshold_``, at most one direction per pair; ``None`` when ``threshold=None``.
    threshold_ : float or None
    edges_ : list[dict]
        Edges of ``graph_`` with their probability, most confident first.
    """

    def __init__(
        self,
        *,
        model="noricausal-31m",
        checkpoint=None,
        mode="auto",
        target_max_features=25,
        n_subsets=64,
        window=30,
        min_target_coverage=3,
        threshold=None,
        device="auto",
        random_state=0,
    ):
        self.model = model
        self.checkpoint = checkpoint
        self.mode = mode
        self.target_max_features = target_max_features
        self.n_subsets = n_subsets
        self.window = window
        self.min_target_coverage = min_target_coverage
        self.threshold = threshold
        self.device = device
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None, feature_names=None, target_name="target"):
        import torch

        if self.mode not in ("auto", "subgraph", "target", "null"):
            raise ValueError(f"mode must be 'auto', 'subgraph', 'target' or 'null', got {self.mode!r}")
        if self.mode == "target" and y is None:
            raise ValueError("mode='target' needs y")
        if feature_names is None and hasattr(X, "columns"):
            feature_names = [str(c) for c in X.columns]
        X = np.asarray(X, np.float32)
        if X.ndim != 2 or X.shape[1] < 1:
            raise ValueError(f"X must be 2-D with at least one column, got shape {X.shape}")
        names = list(feature_names) if feature_names is not None else [f"f{j}" for j in range(X.shape[1])]
        table = X if y is None else np.column_stack([X, np.asarray(y, np.float32).reshape(-1)])
        if y is not None:
            names = names + [target_name]
        if table.shape[1] < 2:
            raise ValueError("need at least two variables (columns of X plus y)")

        device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        path, calibrated_threshold = get_checkpoint_path_and_threshold(self.model, self.checkpoint)
        network, self.checkpoint_meta_ = _get_cached_network(path, device)
        rng = np.random.default_rng(self.random_state)

        def _get_pass_logits(columns, null_target):
            """Logits of one forward pass over ``columns`` (last column = target slot unless ``null_target``)."""
            with torch.no_grad():
                if null_target:
                    placeholder = np.random.default_rng(0).standard_normal(len(columns)).astype(np.float32)
                    logits = network(
                        torch.as_tensor(columns, device=device),
                        torch.as_tensor(placeholder, device=device),
                        null_target=True,
                    )
                    return logits.float().cpu().numpy()[: columns.shape[1], : columns.shape[1]]
                logits = network(
                    torch.as_tensor(np.ascontiguousarray(columns[:, :-1]), device=device),
                    torch.as_tensor(np.ascontiguousarray(columns[:, -1]), device=device),
                )
                return logits.float().cpu().numpy()

        n_nodes = table.shape[1]
        target_index = n_nodes - 1 if y is not None else None
        mode = self.mode
        if mode == "auto":
            if target_index is None:
                mode = "subgraph"
            else:
                mode = "target" if X.shape[1] <= int(self.target_max_features) else "target_subgraph"
        self.mode_ = mode
        if mode in ("null", "target"):
            logits = _get_pass_logits(table, mode == "null")
            self.pair_coverage_ = np.ones((n_nodes, n_nodes), np.int64)
        else:
            window = min(max(int(self.window), 2), n_nodes)
            logit_sum, pair_count = np.zeros((n_nodes, n_nodes)), np.zeros((n_nodes, n_nodes), np.int64)

            def _add_subset_logits(subset):
                """Run one pass over ``subset`` (its last column in the target slot) and accumulate it."""
                logit_sum[np.ix_(subset, subset)] += _get_pass_logits(table[:, subset], False)
                pair_count[np.ix_(subset, subset)] += 1

            for _ in range(int(self.n_subsets)):
                subset = rng.permutation(n_nodes)[:window]
                if mode == "target_subgraph" and target_index in subset:  # the target takes the target slot
                    subset = np.append(subset[subset != target_index], target_index)
                _add_subset_logits(subset)
            if mode == "target_subgraph":
                features = np.arange(n_nodes - 1)
                while (target_coverage := pair_count[features, target_index]).min() < int(self.min_target_coverage):
                    least_covered = np.lexsort((rng.random(len(features)), target_coverage))[: window - 1]
                    _add_subset_logits(np.append(features[least_covered], target_index))
            logits = np.where(pair_count > 0, logit_sum / np.maximum(pair_count, 1), UNCOVERED_LOGIT)
            self.pair_coverage_ = pair_count
        logits = np.asarray(logits, np.float64)
        np.fill_diagonal(logits, 0.0)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
        np.fill_diagonal(probabilities, 0.0)

        self.node_names_, self.target_index_ = names, target_index
        self.logits_, self.adjacency_ = logits, probabilities
        self.n_rows_, self.device_ = int(len(table)), device
        if self.threshold is None:
            self.threshold_, self.graph_, self.edges_ = None, None, []
            return self
        threshold = calibrated_threshold if self.threshold == "calibrated" else float(self.threshold)
        if threshold is None:
            raise ValueError(
                "threshold='calibrated' needs a registry model; pass a float threshold for a custom checkpoint"
            )
        self.threshold_ = float(threshold)
        self.graph_ = get_decoded_graph(probabilities, threshold)
        self.edges_ = sorted(
            (
                {"source": names[i], "target": names[j], "probability": float(probabilities[i, j])}
                for i, j in zip(*np.nonzero(self.graph_))
            ),
            key=lambda edge: -edge["probability"],
        )
        return self

    # ------------------------------------------------------------------ queries
    def _get_node_index(self, node):
        """Index of ``node`` (an int index or a node name) among ``node_names_``; needs a decoded graph."""
        check_is_fitted(self, "graph_")
        if self.graph_ is None:
            raise ValueError("fitted with threshold=None: no graph; read adjacency_ instead")
        return node if isinstance(node, (int, np.integer)) else self.node_names_.index(node)

    def parents(self, node):
        """Direct causes of ``node`` in ``graph_`` (names), most confident first."""
        child = self._get_node_index(node)
        parent_indices = np.nonzero(self.graph_[:, child])[0]
        order = np.argsort(-self.adjacency_[parent_indices, child])
        return [self.node_names_[i] for i in parent_indices[order]]

    def children(self, node):
        """Direct effects of ``node`` in ``graph_`` (names), most confident first."""
        parent = self._get_node_index(node)
        child_indices = np.nonzero(self.graph_[parent])[0]
        order = np.argsort(-self.adjacency_[parent, child_indices])
        return [self.node_names_[j] for j in child_indices[order]]

    def to_frame(self):
        """``adjacency_`` as a labelled pandas DataFrame (rows = cause, columns = effect)."""
        import pandas as pd

        check_is_fitted(self, "adjacency_")
        return pd.DataFrame(self.adjacency_, index=self.node_names_, columns=self.node_names_)
