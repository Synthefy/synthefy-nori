#!/usr/bin/env python
"""Single-H100 latency sweep for the LOCAL synthefy-nori build, run on Modal.

Measures end-to-end inference latency (fit -> predict for one request) of
``NoriRegressor`` across a grid of context/query shapes:

* ``X_train`` rows : 100 .. 10_000  step 100          (100 values)
* ``X_train`` cols : 5   .. 100     step 5            ( 20 values)
* ``X_test``  rows : ceil(n_rows * 0.25)

For each of the 2_000 (rows, cols) combinations we run ``n_warmup + n_measured``
timed calls (default 3 + 10 = 13), DISCARD the warm-ups, and report
``mean / p90 / p99 / std`` (plus min/max) over the measured calls. Every
combination is measured on ONE H100 (one warmed predictor, checkpoint loaded
once and NOT counted). The actual GPU name is recorded per row so any container
that landed on non-H100 hardware is auditable / re-runnable.

To finish fast we SHARD the 2_000 combos across many H100 containers running in
parallel (load-balanced so the heavy high-row combos are spread evenly). This
keeps the per-measurement condition (one H100 per combo) and costs the same
total H100-hours -- it only cuts wall-clock.

DURABILITY: the job is DEPLOYED and SPAWNED, so it runs fully server-side and is
independent of any client connection (a detached local entrypoint is fragile --
Modal keeps only the last-triggered function alive on disconnect, which can kill
a nested driver). Each shard commits its OWN CSV under ``shards/`` as it goes;
the driver merges those durable CSVs at the end. Nothing depends on your laptop
staying online.

    uv build                                                      # refresh the wheel first

    # quick checks (single container, blocking)
    uv run --no-sync --with modal modal run benchmarks/modal_latency_sweep.py --mode smoke
    uv run --no-sync --with modal modal run benchmarks/modal_latency_sweep.py --mode probe

    # full sweep -- deploy then spawn (durable, laptop can be offline)
    uv run --no-sync --with modal modal deploy benchmarks/modal_latency_sweep.py
    uv run --no-sync --with modal python -c "import modal; \
        print(modal.Function.from_name('nori-latency-sweep','drive').spawn('full',50).object_id)"

Fetch results any time (driver writes the merged CSV at the end; shards/ fills in live):

    uv run --no-sync --with modal modal volume get nori-latency-results \
        latency_sweep_h100.csv benchmarks/latency_sweep_h100.csv --force
"""

from __future__ import annotations

import glob
import heapq
import math
import os

import modal

APP_NAME = "nori-latency-sweep"


# ---- ship the LOCAL wheel into the image -------------------------------------
# Modal re-imports this module INSIDE the container to find the function, which
# re-runs everything at module scope. The local wheel only exists on the client
# (it's copied to /root/<wheel> in the image), so the glob + add_local_file must
# be guarded by modal.is_local(). In the container the prebuilt image is used as
# is, so the base image without those layers is fine for decorator resolution.
def _build_image():
    image = (
        modal.Image.debian_slim(python_version="3.11")
        # Persist the HF checkpoint cache across containers/runs.
        .env({"HF_HOME": "/cache/hf"})
    )
    if modal.is_local():
        here = os.path.dirname(os.path.abspath(__file__))
        wheels = sorted(glob.glob(os.path.join(here, "..", "dist", "synthefy_nori-*-py3-none-any.whl")))
        if not wheels:
            raise SystemExit("No wheel in dist/. Build it first: `uv build`")
        wheel = wheels[-1]
        name = os.path.basename(wheel)
        image = image.add_local_file(wheel, f"/root/{name}", copy=True).run_commands(
            f"pip install --no-cache-dir /root/{name}"
        )
    return image


image = _build_image()

app = modal.App(APP_NAME)

# Results CSV(s) live here; HF checkpoint cache is a separate volume.
results_vol = modal.Volume.from_name("nori-latency-results", create_if_missing=True)
cache_vol = modal.Volume.from_name("nori-hf-cache", create_if_missing=True)

RESULTS_DIR = "/results"
SHARDS_DIR = os.path.join(RESULTS_DIR, "shards")
COMMIT_EVERY = 10  # combos between intra-shard Volume commits
CSV_COLS = [
    "n_rows", "n_cols", "n_test", "n_warmup", "n_measured",
    "mean_ms", "p90_ms", "p99_ms", "std_ms", "min_ms", "max_ms", "gpu", "error",
]


def _grid(mode: str):
    """Return (combos, n_warmup, n_measured, out_name) for a run mode."""
    if mode == "smoke":      # validates image/H100/inference end-to-end
        rows, cols, nw, nm, out = [100, 5000], [5, 100], 1, 2, "latency_smoke.csv"
    elif mode == "probe":    # large-end points to estimate full-sweep runtime
        rows, cols, nw, nm, out = [2500, 5000, 7500, 10000], [5, 50, 100], 2, 3, "latency_probe.csv"
    elif mode == "full":     # the real grid
        rows, cols, nw, nm, out = list(range(100, 10_001, 100)), list(range(5, 101, 5)), 3, 10, "latency_sweep_h100.csv"
    elif mode == "large":    # large grid: 10k..100k rows x 50..1000 cols
        rows = list(range(10_000, 100_001, 5_000))    # 19 values
        cols = list(range(50, 1_001, 100)) + [1_000]  # 50,150,..,950,1000 (11 values)
        nw, nm, out = 3, 10, "latency_sweep_large.csv"
    elif mode == "large_probe":  # 4 corners (cheapest + most expensive) -- feasibility/OOM/timing
        rows, cols, nw, nm, out = [10_000, 100_000], [50, 1_000], 1, 2, "latency_large_probe.csv"
    else:
        raise SystemExit(f"unknown mode {mode!r}; use smoke|probe|full|large|large_probe")
    return [(r, c) for r in rows for c in cols], nw, nm, out


def _balance(combos, n_shards):
    """LPT bin-packing: spread combos so each shard has ~equal estimated work.

    Cost proxy follows the probe: latency ~ rows*(0.1 + 0.0075*cols) ms. Sorting
    heaviest-first and dropping each onto the least-loaded shard keeps wall-clock
    even across shards (otherwise a shard full of 10k-row combos lags badly).
    """
    def cost(rc):
        r, c = rc
        return r * (0.1 + 0.0075 * c)

    n_shards = max(1, min(n_shards, len(combos)))
    heap = [(0.0, i) for i in range(n_shards)]
    heapq.heapify(heap)
    shards = [[] for _ in range(n_shards)]
    for rc in sorted(combos, key=cost, reverse=True):
        load, i = heapq.heappop(heap)
        shards[i].append(rc)
        heapq.heappush(heap, (load + cost(rc), i))
    return [s for s in shards if s]


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,           # generous: large-grid shards can hold minutes-long combos
    retries=2,                     # self-heal transient container/preemption failures
    max_containers=128,            # allow aggressive fan-out (capped by acct quota)
    volumes={RESULTS_DIR: results_vol, "/cache": cache_vol},
)
def run_shard(shard_id: int, combos: list, n_warmup: int, n_measured: int):
    """Measure a shard's combos on one GPU; commit its own CSV; return the rows.

    The GPU name is recorded in every row -- ``gpu="H100"`` is occasionally
    satisfied with H200, and a latency benchmark must be auditable for that.
    """
    import time

    import numpy as np
    import pandas as pd
    import torch

    from synthefy_nori import NoriRegressor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -- expected a single H100")
    gpu = torch.cuda.get_device_name(0)
    print(f"[shard {shard_id}] GPU={gpu}  combos={len(combos)}", flush=True)

    # One predictor, loaded once. fit() only stores arrays; all compute is in
    # predict(). The checkpoint download/load happens here and is NOT timed.
    reg = NoriRegressor()
    wrng = np.random.default_rng(0)  # non-constant features (preprocessing rejects constants)
    reg.fit(wrng.standard_normal((8, 5)).astype(np.float32), wrng.standard_normal(8).astype(np.float64))
    reg.predict(wrng.standard_normal((4, 5)).astype(np.float32))
    torch.cuda.synchronize()

    def timed_call(x_tr, y_tr, x_te):
        """End-to-end latency of one inference request (fit + predict), in ms."""
        torch.cuda.synchronize()
        s = time.perf_counter()
        reg.fit(x_tr, y_tr)
        reg.predict(x_te)
        torch.cuda.synchronize()
        return (time.perf_counter() - s) * 1000.0

    os.makedirs(SHARDS_DIR, exist_ok=True)
    out_path = os.path.join(SHARDS_DIR, f"shard_{shard_id:03d}.csv")

    rows: list[dict] = []
    for i, (n_rows, n_cols) in enumerate(combos, 1):
        n_test = math.ceil(n_rows * 0.25)
        rng = np.random.default_rng(n_rows * 1000 + n_cols)
        X_train = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
        w = rng.standard_normal(n_cols).astype(np.float32)
        y_train = (X_train @ w + 0.1 * rng.standard_normal(n_rows).astype(np.float32)).astype(np.float64)
        X_test = rng.standard_normal((n_test, n_cols)).astype(np.float32)

        rec = {"n_rows": n_rows, "n_cols": n_cols, "n_test": n_test,
               "n_warmup": n_warmup, "n_measured": n_measured, "gpu": gpu, "error": ""}
        try:
            for _ in range(n_warmup):  # discarded
                timed_call(X_train, y_train, X_test)
            lat = np.array([timed_call(X_train, y_train, X_test) for _ in range(n_measured)])
            rec.update(
                mean_ms=float(lat.mean()),
                p90_ms=float(np.percentile(lat, 90)),
                p99_ms=float(np.percentile(lat, 99)),
                std_ms=float(lat.std(ddof=1)) if n_measured > 1 else 0.0,
                min_ms=float(lat.min()), max_ms=float(lat.max()),
            )
        except Exception as exc:  # OOM / context- or feature-limit errors -- record, keep going
            rec.update(mean_ms=float("nan"), p90_ms=float("nan"), p99_ms=float("nan"),
                       std_ms=float("nan"), min_ms=float("nan"), max_ms=float("nan"),
                       error=str(exc)[:300])
            torch.cuda.empty_cache()
        rows.append(rec)

        if i % COMMIT_EVERY == 0 or i == len(combos):
            pd.DataFrame(rows, columns=CSV_COLS).to_csv(out_path, index=False)
            results_vol.commit()
            print(f"[shard {shard_id}] {i}/{len(combos)} "
                  f"last={n_rows}x{n_cols} mean={rec.get('mean_ms', float('nan')):.0f}ms", flush=True)

    cache_vol.commit()
    return rows


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,   # generous: pinned + cooldowns is sequential
    retries=1,
    volumes={RESULTS_DIR: results_vol, "/cache": cache_vol},
)
def run_pinned(combos: list, n_warmup: int = 3, n_measured: int = 10,
               out_name: str = "latency_rerun_pinned.csv",
               target_temp_c: int = 65, max_cooldown_s: float = 30.0):
    """Re-measure an explicit combo list on ONE pinned GPU, back-to-back, with an
    ACTIVE thermal cooldown before each combo (poll temp, idle until it drops below
    ``target_temp_c`` or ``max_cooldown_s`` elapses). Records GPU temp + cooldown per
    row so the de-throttling is auditable. Writes a self-contained CSV to the volume.
    """
    import subprocess
    import time

    import numpy as np
    import pandas as pd
    import torch

    from synthefy_nori import NoriRegressor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -- expected a single H100")
    gpu = torch.cuda.get_device_name(0)
    print(f"[pinned] GPU={gpu}  combos={len(combos)}  target<= {target_temp_c}C", flush=True)

    def gpu_temp():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip().splitlines()[0])
        except Exception:  # noqa: BLE001 -- temp is best-effort
            return None

    def cooldown():
        """Idle until the chip is below target (or cap). Returns (waited_s, temp_after)."""
        waited = 0.0
        t = gpu_temp()
        if t is None:                       # no telemetry -> fixed conservative idle
            time.sleep(min(5.0, max_cooldown_s))
            return min(5.0, max_cooldown_s), None
        while t > target_temp_c and waited < max_cooldown_s:
            time.sleep(2.0)
            waited += 2.0
            t = gpu_temp()
        return waited, t

    reg = NoriRegressor()
    wrng = np.random.default_rng(0)
    reg.fit(wrng.standard_normal((8, 5)).astype(np.float32), wrng.standard_normal(8).astype(np.float64))
    reg.predict(wrng.standard_normal((4, 5)).astype(np.float32))
    torch.cuda.synchronize()

    def timed_call(x_tr, y_tr, x_te):
        torch.cuda.synchronize()
        s = time.perf_counter()
        reg.fit(x_tr, y_tr)
        reg.predict(x_te)
        torch.cuda.synchronize()
        return (time.perf_counter() - s) * 1000.0

    out_path = os.path.join(RESULTS_DIR, out_name)
    cols = CSV_COLS + ["gpu_temp_c", "cooldown_s"]
    rows: list[dict] = []
    for i, (n_rows, n_cols) in enumerate(combos, 1):
        n_rows, n_cols = int(n_rows), int(n_cols)
        waited, temp_after = cooldown()   # start each combo from a cool, comparable state
        n_test = math.ceil(n_rows * 0.25)
        rng = np.random.default_rng(n_rows * 1000 + n_cols)
        X_train = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
        w = rng.standard_normal(n_cols).astype(np.float32)
        y_train = (X_train @ w + 0.1 * rng.standard_normal(n_rows).astype(np.float32)).astype(np.float64)
        X_test = rng.standard_normal((n_test, n_cols)).astype(np.float32)

        rec = {"n_rows": n_rows, "n_cols": n_cols, "n_test": n_test,
               "n_warmup": n_warmup, "n_measured": n_measured, "gpu": gpu, "error": "",
               "gpu_temp_c": temp_after, "cooldown_s": round(waited, 1)}
        try:
            for _ in range(n_warmup):
                timed_call(X_train, y_train, X_test)
            lat = np.array([timed_call(X_train, y_train, X_test) for _ in range(n_measured)])
            rec.update(mean_ms=float(lat.mean()), p90_ms=float(np.percentile(lat, 90)),
                       p99_ms=float(np.percentile(lat, 99)),
                       std_ms=float(lat.std(ddof=1)) if n_measured > 1 else 0.0,
                       min_ms=float(lat.min()), max_ms=float(lat.max()))
        except Exception as exc:  # OOM / context- or feature-limit errors -- record, keep going
            rec.update(mean_ms=float("nan"), p90_ms=float("nan"), p99_ms=float("nan"),
                       std_ms=float("nan"), min_ms=float("nan"), max_ms=float("nan"),
                       error=str(exc)[:300])
            torch.cuda.empty_cache()
        rows.append(rec)

        if i % 5 == 0 or i == len(combos):
            pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
            results_vol.commit()
            print(f"[pinned] {i}/{len(combos)} {n_rows}x{n_cols} mean={rec.get('mean_ms', float('nan')):.0f}ms "
                  f"temp={temp_after}C cooled={waited:.0f}s", flush=True)

    cache_vol.commit()
    return {"combos": len(rows), "gpu": gpu, "out_name": out_name}


def _merge_volume_shards(out_name: str):
    """Concat every shards/shard_*.csv on the volume into the merged CSV (durable path)."""
    import pandas as pd

    results_vol.reload()
    files = sorted(glob.glob(os.path.join(SHARDS_DIR, "shard_*.csv")))
    if not files:
        return None, 0
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["n_rows", "n_cols"]).sort_values(["n_rows", "n_cols"]).reset_index(drop=True)
    out_path = os.path.join(RESULTS_DIR, out_name)
    df.to_csv(out_path, index=False)
    results_vol.commit()
    return df, len(files)


@app.function(
    image=image,
    timeout=24 * 60 * 60,  # orchestrator; CPU-only
    volumes={RESULTS_DIR: results_vol},
)
def drive(mode: str = "full", n_shards: int = 50):
    """Build the grid, fan shards out across H100s in parallel, merge their CSVs.

    Runs server-side (spawn this; do not depend on a client). ``return_exceptions``
    keeps one failed shard from cancelling the rest, and the final merge reads the
    DURABLE per-shard CSVs from the volume rather than in-memory returns.
    """
    import time

    combos, n_warmup, n_measured, out_name = _grid(mode)
    shards = _balance(combos, n_shards)
    total_calls = len(combos) * (n_warmup + n_measured)
    print(f"mode={mode}: {len(combos)} combos over {len(shards)} shards; "
          f"{n_warmup} warmup + {n_measured} measured each ({total_calls} forward passes)", flush=True)

    inputs = [(i, sh, n_warmup, n_measured) for i, sh in enumerate(shards)]
    t0 = time.perf_counter()
    done = ok = 0
    for res in run_shard.starmap(inputs, order_outputs=False, return_exceptions=True):
        done += 1
        if isinstance(res, Exception):
            print(f"shard FAILED ({done}/{len(shards)}): {type(res).__name__}: {str(res)[:200]}", flush=True)
        else:
            ok += 1
            print(f"shard ok ({done}/{len(shards)}, {ok} good)  {(time.perf_counter()-t0)/60:.1f}m elapsed", flush=True)

    df, n_files = _merge_volume_shards(out_name)
    if df is None:
        print("NO shard CSVs found on volume -- nothing merged.", flush=True)
        return {"combos": 0, "shards_ok": ok, "shards_total": len(shards), "out_name": out_name}

    gpu_counts = df["gpu"].value_counts().to_dict()
    n_err = int((df["error"].fillna("") != "").sum())
    print(f"DONE: merged {len(df)}/{len(combos)} combos from {n_files} shard files, "
          f"{n_err} measure-errors, {(time.perf_counter()-t0)/60:.1f}m wall.", flush=True)
    print(f"GPU breakdown: {gpu_counts}", flush=True)
    print(f"CSV -> volume:nori-latency-results/{out_name}", flush=True)
    return {"combos": int(len(df)), "expected": len(combos), "shards_ok": ok,
            "shards_total": len(shards), "measure_errors": n_err, "gpu_counts": gpu_counts,
            "out_name": out_name, "wall_minutes": (time.perf_counter() - t0) / 60.0}


@app.function(image=image, timeout=600, volumes={RESULTS_DIR: results_vol})
def merge(out_name: str = "latency_sweep_h100.csv"):
    """Recovery helper: rebuild the merged CSV from whatever shards are on the volume."""
    df, n_files = _merge_volume_shards(out_name)
    if df is None:
        return {"merged": 0, "files": 0}
    return {"merged": int(len(df)), "files": n_files, "gpu_counts": df["gpu"].value_counts().to_dict()}


@app.local_entrypoint()
def main(mode: str = "smoke", shards: int = 50):
    """For smoke/probe (blocking). For the full sweep, deploy + spawn drive instead."""
    summary = drive.remote(mode, shards)
    print("Summary:", summary)
    print(f"Fetch: modal volume get nori-latency-results {summary['out_name']} "
          f"benchmarks/{summary['out_name']} --force")
