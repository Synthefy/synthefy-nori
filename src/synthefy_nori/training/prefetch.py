"""Async data prefetching for Nori training.

Uses a pool of background processes to generate synthetic data batches
while the GPU runs forward/backward. This overlaps CPU data generation
with GPU compute, typically giving 1.5-2.5x training speedup.

Design constraints:
  - shared_rng must stay in main thread (DDP rank sync)
  - Each worker gets an independent RNG seeded deterministically
  - Workers are long-lived (avoid process spawn overhead per step)
  - Batches are returned in submission order (FIFO)
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from collections import deque
from typing import Any

import numpy as np


def _worker_loop(task_queue, result_queue, worker_id, worker_task_ids):
    """Long-lived worker process that generates batches from the task queue.

    Each task contains all parameters needed for generate_batch plus a
    deterministic RNG seed so results are reproducible.
    """
    # Import here to avoid pickling issues and ensure each process has
    # its own copy of the module.
    from synthefy_nori.training.data_generator import generate_batch

    while True:
        task = task_queue.get()
        if task is None:
            # Poison pill — shut down
            break

        task_id, seed, gen_kwargs = task
        worker_task_ids[worker_id] = task_id
        try:
            rng = np.random.default_rng(seed)
            X_batch, y_batch, n_classes = generate_batch(rng=rng, **gen_kwargs)
            result_queue.put((task_id, True, (X_batch, y_batch, n_classes)))
        except Exception as e:
            # Send error back instead of crashing the worker
            tb = traceback.format_exc()
            result_queue.put((task_id, False, (type(e).__name__, str(e), tb)))


class DataPrefetcher:
    """Async data prefetcher using a pool of worker processes.

    Usage:
        prefetcher = DataPrefetcher(num_workers=4, prefetch_count=4)
        prefetcher.start()

        # In training loop:
        prefetcher.submit(seed=..., gen_kwargs={...})
        result = prefetcher.get()  # blocks until next result ready
        X_batch, y_batch, n_classes = result

        prefetcher.shutdown()

    Batches are returned in FIFO (submission) order.
    """

    def __init__(self, num_workers=4, prefetch_count=4):
        self.num_workers = num_workers
        self.prefetch_count = prefetch_count
        self._started = False
        self._task_counter = 0
        self._pending_ids = deque()
        self._results_cache = {}  # task_id -> result (for out-of-order arrival)

    def start(self):
        """Spawn worker processes. Must be called before submit/get."""
        ctx = mp.get_context("spawn")
        self._task_queue = ctx.Queue(maxsize=self.prefetch_count + self.num_workers)
        self._result_queue = ctx.Queue(maxsize=self.prefetch_count + self.num_workers)
        self._workers = []
        # Diagnostic only: avoid locks that a killed worker could leave held.
        # Keep the last assigned logical ID even while its result is in transit.
        self._worker_task_ids = ctx.Array("q", [-1] * self.num_workers, lock=False)
        for i in range(self.num_workers):
            p = ctx.Process(
                target=_worker_loop,
                args=(self._task_queue, self._result_queue, i, self._worker_task_ids),
                daemon=True,
            )
            p.start()
            self._workers.append(p)
        self._started = True

    def _check_workers(self, *, awaiting_task):
        """Fail the pool if any worker exits; its consumed task may be lost."""
        for worker_id, worker in enumerate(self._workers):
            if not worker.is_alive():
                last_task = self._worker_task_ids[worker_id]
                raise RuntimeError(
                    f"DataPrefetcher worker {worker_id} (pid={worker.pid}, "
                    f"exitcode={worker.exitcode}) exited unexpectedly; "
                    f"last assigned logical task={last_task if last_task >= 0 else 'none'}, "
                    f"awaiting logical task={awaiting_task}"
                )

    def _enqueue_task(self, task):
        """Submit through bounded queues while servicing worker backpressure."""
        while True:
            self._check_workers(awaiting_task=task[0])
            try:
                self._task_queue.put_nowait(task)
                return
            except queue.Full:
                try:
                    result = self._result_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                # Queue arrival never changes FIFO order, including errors.
                self._cache_result(*result)

    def submit(self, seed: int, gen_kwargs: dict[str, Any]):
        """Submit a batch, collecting completed results if transport is full.

        Args:
            seed: deterministic RNG seed for this batch
            gen_kwargs: keyword arguments for generate_batch (everything
                        except 'rng', which is created from seed in the worker)
        """
        if not self._started:
            raise RuntimeError("DataPrefetcher not started. Call start() first.")
        task_id = self._task_counter
        self._task_counter += 1
        self._pending_ids.append(task_id)
        self._enqueue_task((task_id, seed, gen_kwargs))

    def _cache_result(self, task_id, success, payload):
        if not success:
            payload = _ErrorSentinel(*payload)
        self._results_cache[task_id] = payload

    def get(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int | None] | _ErrorSentinel:
        """Block until the next batch (in submission order) is ready.

        Returns:
            ``(X_batch, y_batch, n_classes)`` or an ``_ErrorSentinel`` that
            preserves the worker exception type for trainer classification.
        """
        if not self._started:
            raise RuntimeError("DataPrefetcher not started. Call start() first.")
        if not self._pending_ids:
            raise RuntimeError("No pending tasks. Call submit() first.")

        target_id = self._pending_ids[0]
        while target_id not in self._results_cache:
            # Check even when other workers keep returning later batches.
            self._check_workers(awaiting_task=target_id)
            try:
                result = self._result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._cache_result(*result)

        self._pending_ids.popleft()
        return self._results_cache.pop(target_id)

    def pending_count(self) -> int:
        """Number of submitted but not-yet-retrieved tasks."""
        return len(self._pending_ids)

    def shutdown(self):
        """Gracefully shut down all worker processes."""
        if not self._started:
            return
        # Send poison pills
        for _ in self._workers:
            try:
                self._task_queue.put(None, timeout=5)
            except Exception:
                pass
        # Wait for workers to finish
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self._started = False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


class _ErrorSentinel:
    """Placeholder for worker errors cached out of order."""

    def __init__(self, err_type, err_msg, tb):
        self.err_type = err_type
        self.err_msg = err_msg
        self.tb = tb
