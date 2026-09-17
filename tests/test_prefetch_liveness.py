"""Worker death must fail promptly without losing FIFO/chunk assembly."""

import queue
import threading
import time

import numpy as np
import pytest

from synthefy_nori.training import prefetch


def _controlled_worker(task_queue, result_queue, worker_id, worker_task_ids):
    while (task := task_queue.get()) is not None:
        task_id, seed, kwargs = task
        logical_id = task_id
        worker_task_ids[worker_id] = logical_id
        if kwargs.get("stall"):
            time.sleep(60)
        elif kwargs.get("error"):
            result_queue.put((task_id, False, ("SyntheticDataFilterError", "rejected", "traceback")))
        else:
            time.sleep(kwargs.get("delay", 0))
            result_queue.put((task_id, True, (np.full((1, 2, 1), seed), np.full((1, 2), seed), None)))


def _bounded_get(monkeypatch, pool):
    original = pool._result_queue.get
    deadline = time.monotonic() + 5

    def get(*args, **kwargs):
        assert 0 < kwargs.get("timeout", 0) <= 0.5, "result waits must be bounded"
        assert time.monotonic() < deadline, "dead worker was never detected"
        return original(*args, **kwargs)

    monkeypatch.setattr(pool._result_queue, "get", get)


def test_killed_worker_fails_during_result_wait(monkeypatch):
    monkeypatch.setattr(prefetch, "_worker_loop", _controlled_worker)
    pool = prefetch.DataPrefetcher(num_workers=1, prefetch_count=1)
    pool.start()
    try:
        pool.submit(42, {"batch_size": 2, "stall": True})
        deadline = time.monotonic() + 15
        while pool._worker_task_ids[0] != 0:
            assert time.monotonic() < deadline, "worker never consumed task"
            time.sleep(0.01)
        _bounded_get(monkeypatch, pool)
        worker = pool._workers[0]
        # Kill after get() has begun polling, not just before entering it.
        timer = threading.Timer(0.2, worker.kill)
        timer.start()
        try:
            with pytest.raises(
                RuntimeError, match=r"worker 0 .*exitcode=-9.*assigned logical task=0.*awaiting logical task=0"
            ):
                pool.get()
        finally:
            timer.join()
        assert pool.pending_count() == 1
    finally:
        monkeypatch.undo()
        pool.shutdown()
    assert not worker.is_alive()


def test_dead_worker_detected_with_other_results_already_queued(monkeypatch):
    monkeypatch.setattr(prefetch, "_worker_loop", _controlled_worker)
    pool = prefetch.DataPrefetcher(num_workers=2, prefetch_count=2)
    pool.start()
    try:
        pool.submit(1, {"stall": True})
        pool.submit(2, {})
        # The later result proves another worker is healthy and making progress.
        result = pool._result_queue.get(timeout=15)
        assert result[0] == 1
        pool._result_queue.put(result)
        worker_id = list(pool._worker_task_ids).index(0)
        pool._workers[worker_id].kill()
        pool._workers[worker_id].join(timeout=5)
        _bounded_get(monkeypatch, pool)
        with pytest.raises(RuntimeError, match=r"awaiting logical task=0"):
            pool.get()
    finally:
        monkeypatch.undo()
        pool.shutdown()


def test_dead_worker_detected_while_submission_is_backpressured(monkeypatch):
    monkeypatch.setattr(prefetch, "_worker_loop", _controlled_worker)
    pool = prefetch.DataPrefetcher(num_workers=1, prefetch_count=1)
    pool.start()
    original_put = pool._task_queue.put_nowait
    calls = 0

    def full_then_kill(task):
        nonlocal calls
        calls += 1
        assert calls == 1, "submission retried without detecting the dead worker"
        pool._workers[0].kill()
        pool._workers[0].join(timeout=5)
        raise queue.Full

    try:
        monkeypatch.setattr(pool._task_queue, "put_nowait", full_then_kill)
        with pytest.raises(RuntimeError, match=r"assigned logical task=none.*awaiting logical task=0"):
            pool.submit(1, {})
    finally:
        monkeypatch.setattr(pool._task_queue, "put_nowait", original_put)
        pool.shutdown()


def test_live_workers_preserve_fifo_and_backpressure(monkeypatch):
    monkeypatch.setattr(prefetch, "_worker_loop", _controlled_worker)
    pool = prefetch.DataPrefetcher(num_workers=2, prefetch_count=1)
    pool.start()
    try:
        # Fill both transport queues, including a slow first batch.
        for seed in range(10):
            pool.submit(seed, {"delay": 0.15 if seed == 0 else 0})
        _bounded_get(monkeypatch, pool)
        for seed in range(10):
            X, y, n_classes = pool.get()
            np.testing.assert_array_equal(X, np.full((1, 2, 1), seed))
            np.testing.assert_array_equal(y, np.full((1, 2), seed))
            assert n_classes is None
        assert pool.pending_count() == 0
    finally:
        monkeypatch.undo()
        pool.shutdown()


def test_worker_errors_keep_their_type_and_fifo_order(monkeypatch):
    monkeypatch.setattr(prefetch, "_worker_loop", _controlled_worker)
    pool = prefetch.DataPrefetcher(num_workers=2, prefetch_count=1)
    pool.start()
    try:
        pool.submit(0, {"delay": 0.2})
        pool.submit(1, {"error": True})
        # Enqueue enough work to also cache results during submission.
        for seed in range(2, 8):
            pool.submit(seed, {})
        assert pool.get()[2] is None
        error = pool.get()
        assert isinstance(error, prefetch._ErrorSentinel)
        assert (error.err_type, error.err_msg, error.tb) == ("SyntheticDataFilterError", "rejected", "traceback")
        for seed in range(2, 8):
            assert pool.get()[0][0, 0, 0] == seed
    finally:
        monkeypatch.undo()
        pool.shutdown()
