"""Hugging Face checkpoint helpers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_REPO_ID = os.environ.get(
    "SYNTHEFY_TABULAR_HF_REPO",
    "Synthefy/synthefy-tabular",
)
DEFAULT_CHECKPOINT_FILENAME = os.environ.get(
    "SYNTHEFY_TABULAR_HF_FILENAME",
    "synthefy-tabular.pt",
)


def download_checkpoint(
    repo_id: str = DEFAULT_MODEL_REPO_ID,
    filename: str = DEFAULT_CHECKPOINT_FILENAME,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    force_download: bool = False,
) -> str:
    """Download a checkpoint from the Hugging Face Hub and return its local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install huggingface-hub to download checkpoints.") from exc

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        force_download=force_download,
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
    parser = argparse.ArgumentParser(description="Download a Synthefy Tabular checkpoint")
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
    parser = argparse.ArgumentParser(description="Upload a Synthefy Tabular checkpoint")
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
