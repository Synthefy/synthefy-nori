"""Named NoriCausal checkpoints.

A named model resolves to a local checkpoint file in this order: an explicit ``checkpoint=...``, the
model's environment variable, then a download from the model's Hugging Face repo (cached by
``huggingface_hub``). Each entry also carries an edge-probability threshold calibrated for that
checkpoint on synthetic graphs in null-target mode, which is what ``threshold="calibrated"`` uses (not
the default; see ``NoriCausal``).
"""

from __future__ import annotations

import os
from pathlib import Path

NORI_CAUSAL_MODELS = {
    "noricausal-31m": {
        "env": "SYNTHEFY_NORICAUSAL_31M_CHECKPOINT",
        "repo_id": "Synthefy/Nori-Causal",
        "filename": "noricausal_31m.pt",
        # global threshold of the checkpoint's synthetic calibrator (null-target fit), as edge probability
        "threshold": 0.8656008543608323,
        "description": "31M NoriCausal, 600k steps (d=352, 6 row-and-column + 15 column-only blocks)",
    },
}


def get_checkpoint_path_and_threshold(
    model: str = "noricausal-31m", checkpoint: str | os.PathLike | None = None
) -> tuple[Path, float | None]:
    """Return ``(checkpoint path, calibrated threshold or None)`` for a named model or an explicit file."""
    entry = NORI_CAUSAL_MODELS.get(model)
    if entry is None and checkpoint is None:
        raise ValueError(
            f"unknown NoriCausal model {model!r}; choose one of {sorted(NORI_CAUSAL_MODELS)} or pass checkpoint="
        )
    path = checkpoint or (os.environ.get(entry["env"]) if entry else None)
    if not path:
        from synthefy_nori.hf import download_checkpoint

        path = download_checkpoint(repo_id=entry["repo_id"], filename=entry["filename"])
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"NoriCausal checkpoint not found: {path}")
    return path, (entry["threshold"] if entry else None)
