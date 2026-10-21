"""A training job's CloudWatch log lines and the curves read from them (``synthefy._training_logs``), against the
CloudWatch Logs fake in ``_fine_tune_fakes``. The curves read from a report are covered through the tuner, in
``test_nori_fine_tune.py``."""

from synthefy._training_logs import (
    CURVE_KEYS,
    TrainingLogFollower,
    build_curves_from_report,
    list_training_log_streams,
    parse_curves_from_log_lines,
)
from synthefy.nori_data_models import TrainingReport

from ._fine_tune_fakes import JOB, STREAM, FakeLogs, build_report


def test_follower_reads_every_stream_of_the_job_across_pages_each_line_once():
    logs = FakeLogs()
    logs.streams = {
        STREAM: ["a1", "a2", "a3"],
        f"{JOB}/algo-2-1759320000": ["b1"],
        f"{JOB}0/algo-1-1759320000": ["another job whose name starts with this one"],
    }
    follower = TrainingLogFollower(logs, JOB)

    assert follower.read_new_lines() == ["a1", "a2", "a3", "b1"]
    logs.streams[STREAM].append("a4")
    assert follower.read_new_lines() == ["a4"]
    assert follower.stream_names == {STREAM, f"{JOB}/algo-2-1759320000"}


def test_follower_reads_past_an_empty_page_while_the_forward_token_advances():
    logs = FakeLogs()
    logs.streams = {STREAM: ["child=0 step=1 train_loss=0.9 val_mse=0.5"]}
    logs.empty_pages = {STREAM: 2}

    lines = TrainingLogFollower(logs, JOB).read_new_lines()

    assert lines == ["child=0 step=1 train_loss=0.9 val_mse=0.5"]
    assert parse_curves_from_log_lines(lines)["val_mse"] == [0.5]


def test_no_log_streams_are_listed_while_the_log_group_does_not_exist():
    logs = FakeLogs()
    logs.group_exists = False
    assert list_training_log_streams(logs, JOB) == []
    assert TrainingLogFollower(logs, JOB).read_new_lines() == []


def test_metric_lines_become_curve_rows_and_progress_lines_are_skipped():
    lines = [
        "nori-train: starting",
        "child=0 step=1 train_loss=0.9 val_mse=0.5",
        "nori-train: a progress line between two metric lines",
        "child=0 step=3 train_loss=0.7 val_mse=0.4\n",
        "child=1 step=1 train_loss=1.5e-07 val_mse=1.23457e+07",
    ]
    assert parse_curves_from_log_lines(lines) == {
        "child": [0, 0, 1],
        "step": [1, 3, 1],
        "train_loss": [0.9, 0.7, 1.5e-07],
        "val_mse": [0.5, 0.4, 1.23457e07],
    }


def test_curves_from_a_report_are_the_same_long_rows_child_by_child():
    report = TrainingReport(**build_report())

    curves = build_curves_from_report(report)

    assert tuple(curves) == CURVE_KEYS
    assert curves["child"] == [0, 0, 0, 1, 1, 1]
    assert curves["step"] == [1, 2, 3, 1, 2, 3]
    assert curves["train_loss"] == [0.9, None, 0.7, 0.9, None, 0.7]
    assert curves["val_mse"] == [0.5, 0.45, 0.4, 0.5, 0.45, 0.4]
