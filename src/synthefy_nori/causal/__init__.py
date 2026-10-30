"""NoriCausal: zero-shot causal-graph estimation from a single observational table.

    from synthefy_nori.causal import NoriCausal
    causal = NoriCausal(checkpoint="noricausal_31m.pt").fit(X, y)
    causal.adjacency_                       # edge probabilities, [i, j] = P(node i -> node j)

Also exposed through :class:`synthefy_nori.interpretability.glassbox.NoriInterpreter` (``causal=True``).
"""

from synthefy_nori.causal.estimator import NoriCausal, get_decoded_graph
from synthefy_nori.causal.registry import NORI_CAUSAL_MODELS, get_checkpoint_path_and_threshold

__all__ = ["NORI_CAUSAL_MODELS", "NoriCausal", "get_checkpoint_path_and_threshold", "get_decoded_graph"]
