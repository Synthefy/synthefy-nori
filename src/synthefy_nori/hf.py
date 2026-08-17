"""Hugging Face checkpoint helpers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_REPO_ID = os.environ.get(
    "SYNTHEFY_NORI_HF_REPO",
    "Synthefy/Nori",
)
DEFAULT_CHECKPOINT_FILENAME = os.environ.get(
    "SYNTHEFY_NORI_HF_FILENAME",
    "nori.pt",
)

# Model-variant registry: friendly name -> Hugging Face repo id. A size is REQUIRED -- there is no
# default "nori"; every caller must pick ``model="nori-6m"`` or ``model="nori-30m"`` on NoriRegressor
# / infer / predict (or ``download_checkpoint(model=...)``). Naming the size keeps the identifier
# stable: it never silently changes which weights it loads. ``"nori-6m"`` is the ~6M base and honors
# the SYNTHEFY_NORI_HF_REPO override. Add one line per new variant. An unknown name is treated as a
# raw repo id, so an explicit ``"org/repo"`` still works.
NORI_MODELS = {
    "nori-6m": DEFAULT_MODEL_REPO_ID,  # ~6M base (honors SYNTHEFY_NORI_HF_REPO)
    "nori-30m": "Synthefy/Nori-30M",   # ~29.2M scaling-law variant
}


def _is_thinking_model(model: str) -> bool:
    """Return ``True`` if ``model`` names a Nori Thinking (test-time-compute) variant.

    Thinking variants (e.g. ``"nori-30m-thinking-medium"`` / the ``"synthefy/..."`` gateway
    slug) run only on the hosted Synthefy API; ``synthefy-nori`` does single-pass local
    inference and ships no Thinking checkpoint. Matching on the ``"thinking"`` token covers
    every budget tier and both the friendly and slug spellings.
    """
    return "thinking" in model.lower()


def resolve_model_repo(model: str | None) -> str:
    """Map a variant name to its HF repo id. A size is required: ``None`` or a bare ``"nori"``
    raises (there is no default) -- pick ``"nori-6m"`` or ``"nori-30m"``. A known name -> its repo;
    anything else is returned unchanged (so a raw ``"org/repo"`` id also works).

    A Nori Thinking selector raises :class:`ValueError`: it has no downloadable checkpoint here,
    so without this guard it would fall through to a raw-repo lookup and fail with an opaque
    "repo not found" error instead of telling the caller it is a hosted-API-only variant."""
    if model is None or model == "nori":
        raise ValueError(
            "model is required and must name a size -- choose one of: "
            f"{', '.join(NORI_MODELS)}. There is no bare 'nori' default."
        )
    if _is_thinking_model(model):
        raise ValueError(
            f"model={model!r} selects a Nori Thinking (test-time-compute) variant, which runs "
            "only on the hosted Synthefy API. The synthefy-nori package does single-pass local "
            "inference and has no Thinking checkpoint. Use the hosted API (e.g. the `synthefy` "
            "client with mode='remote') for Thinking, or select 'nori-6m' / 'nori-30m' "
            "for local inference."
        )
    return NORI_MODELS.get(model, model)


LIMIX_REPO_ID = "stableai-org/LimiX-2M"
LIMIX_FILENAME = "LimiX-2M.ckpt"


class CheckpointAccessError(RuntimeError):
    """Raised when the Hugging Face checkpoint cannot be downloaded due to auth."""


def _access_error_message(repo_id: str) -> str:
    return (
        f"Could not download '{repo_id}' from the Hugging Face Hub: access denied.\n"
        f"This checkpoint requires authentication. To resolve:\n"
        f"  1. Request access at https://huggingface.co/{repo_id}\n"
        f"  2. Get a token at https://huggingface.co/settings/tokens (read scope)\n"
        f"  3. Provide it via `export HF_TOKEN=hf_...`, `hf auth login`,\n"
        f"     or pass token=... to NoriRegressor.\n"
        f"If you already have a local checkpoint, pass model_path=... to skip the download."
    )


def download_checkpoint(
    repo_id: str | None = None,
    filename: str = DEFAULT_CHECKPOINT_FILENAME,
    *,
    model: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    force_download: bool = False,
) -> str:
    """Download a checkpoint from the Hugging Face Hub and return its local path.

    ``model`` selects a registry variant (e.g. ``"nori-6m"``) and overrides ``repo_id``. A size is
    required: with neither ``model`` nor ``repo_id`` given, this raises -- pass
    ``model="nori-6m"``/``"nori-30m"`` or an explicit ``repo_id``.
    """
    if model is not None:
        repo_id = resolve_model_repo(model)
    elif repo_id is None:
        raise ValueError(
            "download_checkpoint requires model= ('nori-6m'/'nori-30m') or an explicit repo_id="
        )
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError as exc:
        raise ImportError("Install huggingface-hub to download checkpoints.") from exc

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
        )
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise CheckpointAccessError(_access_error_message(repo_id)) from exc
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise CheckpointAccessError(_access_error_message(repo_id)) from exc
        raise

    if repo_id in NORI_MODELS.values() and filename != "config.json":
        # The Hub counts model downloads only via its query file (config.json),
        # never via .pt requests, so fetch the small config alongside the
        # checkpoint to make downloads show up in the repo's stats (for every
        # Synthefy Nori variant, not just the base).
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
        except Exception:
            pass

    return path


def download_limix(
    *,
    cache_dir: str | None = None,
    force_download: bool = False,
) -> str:
    """Download the LimiX-2M base checkpoint from HuggingFace.

    Used as an ICL learnability filter during training. The checkpoint is
    publicly hosted at ``stableai-org/LimiX-2M``.
    """
    return download_checkpoint(
        repo_id=LIMIX_REPO_ID,
        filename=LIMIX_FILENAME,
        force_download=force_download,
        cache_dir=cache_dir,
    )


def push_checkpoint(
    checkpoint_path: str,
    repo_id: str,
    *,
    filename: str | None = None,
    private: bool = True,
    token: str | bool | None = None,
    commit_message: str | None = None,
    create_repo: bool = True,
) -> str:
    """Upload a local checkpoint to a Hugging Face model repository."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("Install huggingface-hub to upload checkpoints.") from exc

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(checkpoint_path)

    api = HfApi(token=token)
    if create_repo:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    return api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=filename or path.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message or f"Upload {filename or path.name}",
    )


def download_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download a Nori checkpoint")
    parser.add_argument("--repo-id", default=DEFAULT_MODEL_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_CHECKPOINT_FILENAME)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(argv)
    print(
        download_checkpoint(
            repo_id=args.repo_id,
            filename=args.filename,
            revision=args.revision,
            cache_dir=args.cache_dir,
            token=args.token,
            force_download=args.force_download,
        )
    )


def upload_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Upload a Nori checkpoint")
    parser.add_argument("checkpoint_path")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--filename", default=None)
    parser.add_argument("--public", action="store_true", help="Create the model repo as public")
    parser.add_argument("--token", default=None)
    parser.add_argument("--commit-message", default=None)
    parser.add_argument("--no-create-repo", action="store_true")
    args = parser.parse_args(argv)
    print(
        push_checkpoint(
            args.checkpoint_path,
            args.repo_id,
            filename=args.filename,
            private=not args.public,
            token=args.token,
            commit_message=args.commit_message,
            create_repo=not args.no_create_repo,
        )
    )
