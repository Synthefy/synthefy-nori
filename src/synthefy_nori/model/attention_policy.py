"""Explicit, checkpointed controls for two-phase Q/K norm and context-label attention."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QKNormalization(str, Enum):
    """How the two-phase attentions normalize queries and keys.

    ``AFFINE`` is the learned per-head gain every released checkpoint uses. The other two values are
    declared by internal research checkpoints; this release refuses them (see ``TwoPhaseEncoder``).
    """

    AFFINE = "affine"
    NON_AFFINE = "non_affine"
    RENORMALIZE = "renormalize"


class ContextLabelScale(str, Enum):
    """Which key count scales the first layer's context-label attention (``log(n)`` query scaling)."""

    LOG_LENGTH = "log_length"
    CAP = "cap"
    FIXED = "fixed"


@dataclass(frozen=True)
class TwoPhaseAttentionPolicy:
    """The attention settings a two-phase checkpoint records in ``two_phase_config["attention_policy"]``.

    The default (``AFFINE`` and ``LOG_LENGTH``) is what a checkpoint without the entry means: the log-n query
    scale uses the number of context keys. ``CAP`` uses ``min(context keys, reference_keys)`` and ``FIXED`` uses
    ``reference_keys`` for the context rows of the first layer's target-stream attention.
    """

    qk_normalization: QKNormalization = QKNormalization.AFFINE
    context_label_scale: ContextLabelScale = ContextLabelScale.LOG_LENGTH
    reference_keys: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "qk_normalization", QKNormalization(self.qk_normalization))
        object.__setattr__(self, "context_label_scale", ContextLabelScale(self.context_label_scale))
        if self.context_label_scale == ContextLabelScale.LOG_LENGTH:
            if self.reference_keys is not None:
                raise ValueError("reference_keys requires cap or fixed context-label scale")
        elif type(self.reference_keys) is not int or self.reference_keys < 2:
            raise ValueError("cap/fixed context-label scale requires integer reference_keys >= 2")

    def to_config(self) -> dict:
        """Return the plain-value record for ``model_config``; enums would break ``torch.load(weights_only=True)``."""
        return {
            "qk_normalization": self.qk_normalization.value,
            "context_label_scale": self.context_label_scale.value,
            "reference_keys": self.reference_keys,
        }

    def resolve_context_scale_keys(self, context_keys: int) -> int:
        """Return the key count that the log-n query scale of the context-label attention uses."""
        if self.context_label_scale == ContextLabelScale.LOG_LENGTH:
            return context_keys
        assert self.reference_keys is not None
        if self.context_label_scale == ContextLabelScale.CAP:
            return min(context_keys, self.reference_keys)
        return self.reference_keys
