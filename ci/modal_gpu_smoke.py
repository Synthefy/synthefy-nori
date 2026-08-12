"""GPU CI smoke test driven by Modal.

Runs the two end-to-end checks on a real (Ampere+) GPU in a fresh, isolated,
ephemeral Modal container:

* inference e2e (downloads the public Synthefy/Nori checkpoint, real forward pass)
* one training step from scratch on ``cuda:0`` (mixed precision / autocast path)

It mounts the locally-checked-out source (whatever the GitHub job checked out),
``uv sync``s it, and runs the same pytest selection the CPU gate uses. A failing
test makes ``modal run`` exit non-zero, which fails the GitHub job -> merge gate.
The ``--torch-version`` entrypoint option accepts ``locked`` for the repository's
benchmarked build or an explicit version installed from the CUDA 13.0 index.

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

import tempfile

import modal

app = modal.App("nori-gpu-ci")

# Heavy deps are pre-warmed into the uv cache at *build* time (keyed on
# pyproject.toml + uv.lock) so the runtime `uv sync` is a fast cache hit with no
# per-run torch re-download. UV_LINK_MODE=copy avoids hardlink errors. The source
# tree is mounted read-only at /src and copied to a writable dir in the function
# (uv writes .venv into the project dir).
image = (
    modal.Image.debian_slim(python_version="3.11")
    # cu130 support was added after the uv version cached by the old image.
    .pip_install("uv==0.11.27")
    .env({"UV_LINK_MODE": "copy", "UV_CACHE_DIR": "/uvcache"})
    .add_local_file("pyproject.toml", "/build/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/build/uv.lock", copy=True)
    # The root project depends on the consolidated workspace member, so the
    # pre-warm layer must include that member before uv parses the workspace.
    .add_local_dir(
        "libs/synthefy",
        "/build/libs/synthefy",
        copy=True,
        ignore=["__pycache__", "*.pyc", "dist", "build"],
    )
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
def gpu_smoke(torch_version: str = "locked") -> None:
    import os
    import shutil
    import subprocess

    # torch lives in a uv-managed venv, not the base Python, so each subprocess
    # uses that environment's interpreter/test runner explicitly.

    # Writable copy of the mounted (read-only) source so `uv sync` can write .venv.
    repo = f"{tempfile.mkdtemp(prefix='nori-gpu-')}/repo"
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

    if torch_version == "locked":
        python = f"{repo}/.venv/bin/python"
        pytest = f"{repo}/.venv/bin/pytest"
    else:
        compat_venv = f"{repo}/.venv-cu130"
        python = f"{compat_venv}/bin/python"
        pytest = f"{compat_venv}/bin/pytest"
        subprocess.run(
            ["uv", "venv", "--python", "3.11", compat_venv],
            cwd=repo,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                "uv", "pip", "install", "--no-config", "--python", python,
                f"torch=={torch_version}", "--torch-backend=cu130",
            ],
            cwd=repo,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                "uv", "pip", "install", "--no-config", "--python", python,
                "-e", ".[dev]",
            ],
            cwd=repo,
            env=env,
            check=True,
        )

    # Fail fast with a clear message if the GPU / driver isn't usable.
    subprocess.run(
        [python, "-c",
         "import torch; assert torch.cuda.is_available(), 'no CUDA device on the Modal runner';"
         " print('GPU:', torch.cuda.get_device_name(0));"
         " print('torch:', torch.__version__, 'CUDA:', torch.version.cuda,"
         " 'cuDNN:', torch.backends.cudnn.version())"],
        cwd=repo, env=env, check=True,
    )
    # OPTIONAL flash-attn coverage (A100 is sm80, so FA2 is supported). Left off
    # because the codebase currently has no flash path; enable only once flash
    # code returns (it is a long source build):
    # subprocess.run(["uv", "pip", "install", "flash-attn", "--no-build-isolation"],
    #                cwd=repo, env=env, check=True)
    subprocess.run(
        [
            pytest, "-m", "slow",
            "tests/test_inference_e2e.py", "tests/test_training_smoke.py", "-q",
        ],
        cwd=repo, env=env, check=True,
    )


@app.local_entrypoint()
def main(torch_version: str = "locked") -> None:
    gpu_smoke.remote(torch_version)
