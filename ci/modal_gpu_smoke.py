"""GPU CI smoke test driven by Modal.

Runs the two end-to-end checks on a real (Ampere+) GPU in a fresh, isolated,
ephemeral Modal container:

* inference e2e (downloads the public Synthefy/Nori checkpoint, real forward pass)
* one training step from scratch on ``cuda:0`` (mixed precision / autocast path)

It mounts the locally-checked-out source (whatever the GitHub job checked out),
``uv sync``s it, and runs the same pytest selection the CPU gate uses. A failing
test makes ``modal run`` exit non-zero, which fails the GitHub job -> merge gate.

Run locally / in CI::

    modal run ci/modal_gpu_smoke.py

One-time setup:

* a Modal account (https://modal.com), token via ``modal token new``
* in CI: repo secrets ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET``

Note: pin the ``modal`` version and verify the image/GPU API against your
installed client -- this targets modal 1.x (``App`` / ``add_local_dir`` /
``gpu=`` / ``max_containers``).
"""
from __future__ import annotations

import modal

app = modal.App("nori-gpu-ci")

# Heavy deps are pre-warmed into the uv cache at *build* time (keyed on
# pyproject.toml + uv.lock) so the runtime `uv sync` is a fast cache hit with no
# per-run torch re-download. UV_LINK_MODE=copy avoids hardlink errors. The source
# tree is mounted read-only at /src and copied to a writable dir in the function
# (uv writes .venv into the project dir).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .env({"UV_LINK_MODE": "copy", "UV_CACHE_DIR": "/uvcache"})
    .add_local_file("pyproject.toml", "/build/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/build/uv.lock", copy=True)
    .run_commands("cd /build && uv sync --no-install-project --extra dev")
    .add_local_dir(
        ".",
        "/src",
        # Keep the upload lean and never ship local build/venv/data artifacts.
        ignore=[
            ".git", ".venv", "dist", "build", "*.png",
            "data", "checkpoints", "results", "wandb", "cache",
        ],
    )
)


@app.function(
    gpu="A100",  # Ampere+ -> flash-attn-capable if flash code is reintroduced.
    image=image,
    timeout=1800,
    # Cap simultaneous GPU containers (e.g. many PRs at once); extra runs queue.
    max_containers=10,
)
def gpu_smoke() -> None:
    import os
    import shutil
    import subprocess

    # torch lives in the uv-managed venv (after `uv sync`), not the base Python,
    # so all torch usage goes through `uv run` below.

    # Writable copy of the mounted (read-only) source so `uv sync` can write .venv.
    repo = "/root/repo"
    shutil.copytree("/src", repo)

    env = {
        **os.environ,
        "SYNTHEFY_NORI_SMOKE_DEVICE": "cuda:0",
        "WANDB_MODE": "disabled",
        # Stream subprocess (uv / pytest) output to the GitHub Actions log in
        # real time instead of in buffered chunks.
        "PYTHONUNBUFFERED": "1",
    }

    subprocess.run(["uv", "sync", "--extra", "dev"], cwd=repo, env=env, check=True)
    # Fail fast with a clear message if the GPU / driver isn't usable.
    subprocess.run(
        ["uv", "run", "python", "-c",
         "import torch; assert torch.cuda.is_available(), 'no CUDA device on the Modal runner';"
         " print('GPU:', torch.cuda.get_device_name(0))"],
        cwd=repo, env=env, check=True,
    )
    # PROOF RUN ONLY (throwaway branch): turn flash-attention ON to show the GPU
    # gate catches the bug. Prebuilt wheel matching torch 2.8 / cu12 / cp311 /
    # cxx11abiTRUE (source build would need nvcc, which the image lacks).
    flash_wheel = (
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/"
        "flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
    )
    subprocess.run(["uv", "pip", "install", flash_wheel], cwd=repo, env=env, check=True)
    subprocess.run(
        ["uv", "run", "python", "-c",
         "import flash_attn, synthefy_nori.model.layer as L;"
         " print('flash_attn', flash_attn.__version__, 'HAVE_FLASH_ATTN', L.HAVE_FLASH_ATTN)"],
        cwd=repo, env=env, check=True,
    )
    subprocess.run(
        [
            "uv", "run", "pytest", "-m", "slow",
            "tests/test_inference_e2e.py", "tests/test_training_smoke.py", "-q",
        ],
        cwd=repo, env=env, check=True,
    )


@app.local_entrypoint()
def main() -> None:
    gpu_smoke.remote()
