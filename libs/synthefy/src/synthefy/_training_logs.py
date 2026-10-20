"""A fine-tuning job's CloudWatch log lines, and the training curves read from them or from its report.

SageMaker writes a training job's output to the ``/aws/sagemaker/TrainingJobs`` log group, one stream per instance.
:class:`TrainingLogFollower` returns each line once across repeated reads; the curve functions turn the job's
metric lines, or a :class:`~synthefy.TrainingReport`'s per-child curves, into the same long rows. Every call goes
through the CloudWatch Logs client it is given, so this module imports no AWS SDK.
"""

import re
from typing import Any, Dict, List, Optional, Set

from synthefy._aws import read_aws_error
from synthefy.nori_data_models import TrainingReport

#: The job's metric line, ``child=<k> step=<n> train_loss=<x> val_mse=<x>``, with the Algorithm's own number
#: patterns, so every line the curves read is one SageMaker also records as the four metrics.
METRIC_LINE_PATTERN = re.compile(
    r"^child=(?P<child>[0-9]+) step=(?P<step>[0-9]+) "
    r"train_loss=(?P<train_loss>[0-9.eE+-]+) val_mse=(?P<val_mse>[0-9.eE+-]+)$"
)
#: The columns of :meth:`~synthefy.SynthefyNoriFineTuner.get_training_curves`: the Algorithm's metric names, in the
#: metric line's order.
CURVE_KEYS = ("child", "step", "train_loss", "val_mse")
TRAINING_LOG_GROUP = "/aws/sagemaker/TrainingJobs"


def list_training_log_streams(logs: Any, job_name: str) -> List[Dict[str, Any]]:
    """The job's CloudWatch log streams; none while the log group does not exist.

    The group exists only once some job in the account has logged, so its absence means "no lines yet"; any other
    error reaches the caller.
    """
    request: Dict[str, Any] = {"logGroupName": TRAINING_LOG_GROUP, "logStreamNamePrefix": f"{job_name}/"}
    streams: List[Dict[str, Any]] = []
    while True:
        try:
            page = logs.describe_log_streams(**request)
        except Exception as exc:
            if read_aws_error(exc)[0] == "ResourceNotFoundException":
                return []
            raise
        streams.extend(page.get("logStreams") or [])
        token = page.get("nextToken")
        if not token:
            return streams
        request["nextToken"] = token


class TrainingLogFollower:
    """The new lines of a training job's CloudWatch log streams, each line returned once.

    SageMaker writes one stream per instance, ``<job>/algo-1-<epoch>``, which appears only once the container
    starts, so every read lists the streams again; each stream's forward token is kept so a read returns only what
    arrived since the previous one.
    """

    def __init__(self, logs: Any, job_name: str) -> None:
        self._logs = logs
        self._job_name = job_name
        self._tokens: Dict[str, str] = {}
        self.stream_names: Set[str] = set()

    def read_new_lines(self) -> List[str]:
        """Every line the job's streams gained since the previous call, stream by stream.

        A stream is read until GetLogEvents hands back the forward token it was given, which is CloudWatch's only
        end-of-stream signal: a page can be empty while the token still advances, and the lines after it still come.
        """
        lines: List[str] = []
        for stream in list_training_log_streams(self._logs, self._job_name):
            name = stream["logStreamName"]
            self.stream_names.add(name)
            while True:
                token: Optional[str] = self._tokens.get(name)
                request: Dict[str, Any] = {
                    "logGroupName": TRAINING_LOG_GROUP,
                    "logStreamName": name,
                    "startFromHead": True,
                }
                if token:
                    request["nextToken"] = token
                page = self._logs.get_log_events(**request)
                lines.extend(event["message"] for event in page.get("events") or [])
                next_token = page.get("nextForwardToken")
                if not next_token or next_token == token:
                    break
                self._tokens[name] = next_token
        return lines


def create_empty_curves() -> Dict[str, List[Any]]:
    """The curves of a job with no validated step yet: every column, no rows."""
    return {key: [] for key in CURVE_KEYS}


def parse_curves_from_log_lines(lines: List[str]) -> Dict[str, List[Any]]:
    """One row per metric line; progress lines (``nori-train: ...``) never match the pattern."""
    curves = create_empty_curves()
    for line in lines:
        match = METRIC_LINE_PATTERN.match(line.strip())
        if match is None:
            continue
        curves["child"].append(int(match.group("child")))
        curves["step"].append(int(match.group("step")))
        curves["train_loss"].append(float(match.group("train_loss")))
        curves["val_mse"].append(float(match.group("val_mse")))
    return curves


def build_curves_from_report(report: TrainingReport) -> Dict[str, List[Any]]:
    """The report's per-child curves as the same long rows; a non-finite score is None there."""
    curves = create_empty_curves()
    for curve in report.child_curves:
        for index, step in enumerate(curve.steps):
            curves["child"].append(curve.child)
            curves["step"].append(step)
            curves["train_loss"].append(curve.train_loss[index])
            curves["val_mse"].append(curve.val_mse[index])
    return curves
