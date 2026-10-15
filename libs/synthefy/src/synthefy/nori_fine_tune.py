"""Fine-tune Nori on Amazon SageMaker and deploy the result, all in the caller's own AWS account.

:class:`SynthefyNoriFineTuner` drives the Nori fine-tuning Algorithm (a SageMaker Algorithm resource, named by
ARN, or by name in the caller's own account) through a job's lifecycle: it starts one training job, reports its
status, log and curves while it runs, reads the report its artifact carries, deploys the artifact as an
asynchronous endpoint, and builds the :class:`~synthefy.SynthefyNoriClient` that predicts through it with
``mode="sagemaker"`` and the fine-tuned model name.

It is a class of its own rather than methods on :class:`~synthefy.SynthefyNoriClient` because that client binds one
existing endpoint, model and staging prefix when it is built, and none of them exist before a job has trained.
Like the client's SageMaker mode it signs with boto3's standard credential chain (``pip install "synthefy[aws]"``)
and accepts no raw keys; boto3 is imported only when a fine-tuner is built, so importing this module needs neither
boto3 nor torch. The request and result schemas are in :mod:`synthefy.nori_data_models`; the shared AWS helpers,
the training job request, its description and the checks before it is created, the log and curve reading, the
report reading and the deployment's steps are in the private ``_aws``, ``_training_jobs``, ``_training_logs``,
``_training_artifacts`` and ``_sagemaker_deploy`` modules.
"""

import logging
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from synthefy._aws import (
    AWS_USER_AGENT_EXTRA,
    FINE_TUNING_LOGGER_NAME,
    check_sagemaker_resource_name,
    check_timeout_and_retries,
    create_aws_clients,
    format_seconds,
    parse_s3_uri,
    read_aws_error,
)
from synthefy._sagemaker_deploy import (
    DEFAULT_ENDPOINT_INSTANCE_TYPE,
    ENDPOINT_TIMEOUT_S,
    build_endpoint_config_name,
    create_deployment,
    delete_deployment,
    find_deployable_artifact,
    refuse_existing_endpoint,
)
from synthefy._training_artifacts import read_training_report
from synthefy._training_jobs import (
    build_job_name,
    build_training_job_request,
    parse_job_description,
    refuse_csv_siblings,
    refuse_foreign_region,
    revalidate_request,
)
from synthefy._training_logs import (
    TrainingLogFollower,
    build_curves_from_report,
    create_empty_curves,
    parse_curves_from_log_lines,
)
from synthefy.errors import APITimeoutError, FineTuningError
from synthefy.nori_client import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT_S, SynthefyNoriClient
from synthefy.nori_data_models import FineTuneJob, FineTuneJobRequest, FineTunedEndpoint, TrainingReport

_LOG = logging.getLogger(FINE_TUNING_LOGGER_NAME)

#: SageMaker's terminal training job statuses: a job in one of them holds no instance and changes no more.
TERMINAL_JOB_STATUSES = frozenset({"Completed", "Failed", "Stopped"})
#: Statuses in which StopTrainingJob has nothing left to do: the job ended, or a stop is already under way.
_NOTHING_TO_STOP_JOB_STATUSES = TERMINAL_JOB_STATUSES | {"Stopping"}
#: Connect and read seconds for one AWS call. Every call here answers in well under a second, and the artifact
#: read streams, so this bounds a stalled connection, not a slow download.
_AWS_CALL_TIMEOUT_S = 60.0
#: Polls never run faster than this, whatever the caller passes, so a loop cannot throttle the account's API quota.
_MIN_POLL_S = 5.0
#: wait()'s default poll: a job runs for tens of minutes to hours and SageMaker moves its status in minutes, so a
#: faster poll learns nothing sooner and spends the account's DescribeTrainingJob quota.
_JOB_POLL_S = 60.0
#: get_logs(follow=True)'s default poll: a training job logs a metric line every few seconds, so ten seconds
#: keeps the tail current without listing the log streams more often than lines arrive.
_LOG_POLL_S = 10.0
#: wait()'s default deadline adds this to MaxRuntimeInSeconds: a job can wait for capacity for hours (SageMaker
#: reports it as Pending), and the artifact upload comes after the runtime ends.
_WAIT_OVERHEAD_S = 7200


def _utcnow() -> datetime:
    """The current UTC time; the one clock every generated name reads, so tests can pin those names."""
    return datetime.now(timezone.utc)


class SynthefyNoriFineTuner:
    """Fine-tune Nori with the SageMaker fine-tuning Algorithm and deploy the result, in the caller's AWS account.

    Every AWS call is one the caller's credentials must allow; nothing runs in Synthefy's account. Progress goes to
    the ``synthefy.fine_tune`` logger.

    Parameters
    ----------
    region_name : str or None, optional
        The AWS region every call goes to. ``None`` uses the region boto3 resolves from the environment or shared
        config; either way the region in use is ``self.region_name``, and an Algorithm ARN from another region is
        refused before SageMaker sees it.
    timeout : float, default 60.0
        Connect and read timeout, in seconds, of each AWS call; not a deadline for a job or an endpoint, which
        :meth:`wait` and :meth:`deploy` take on their own.
    max_retries : int, default 2
        Retries after the first attempt of each AWS call, in botocore's standard retry mode.
    user_agent : str or None, optional
        Appended to boto3's user agent for request attribution; ``"synthefy-python"`` by default.

    Attributes
    ----------
    region_name : str
        The region every call goes to.

    Example::

        from synthefy import FineTuneHyperparameters, FineTuneJobRequest, SynthefyNoriFineTuner

        with SynthefyNoriFineTuner(region_name="us-east-1") as tuner:
            job = tuner.fine_tune(
                FineTuneJobRequest(
                    algorithm="arn:aws:sagemaker:us-east-1:123456789012:algorithm/nori-fine-tuning",
                    training_data="s3://my-bucket/churn/train/",
                    output_path="s3://my-bucket/churn/output/",
                    role_arn="arn:aws:iam::123456789012:role/SageMakerExecution",
                    hyperparameters=FineTuneHyperparameters(target_column="churn", model_name="ft-acme-churn-v1"),
                )
            )
            job = tuner.wait(job.job_name)
            print(tuner.get_training_report(job.job_name).val_mse_after)
            endpoint = tuner.deploy(job)
            with tuner.create_client(endpoint, s3_staging_uri="s3://my-bucket/churn/staging/") as client:
                predictions = client.predict(X_train, y_train, X_test)
            tuner.undeploy(endpoint)
    """

    def __init__(
        self,
        *,
        region_name: Optional[str] = None,
        timeout: float = _AWS_CALL_TIMEOUT_S,
        max_retries: int = 2,
        user_agent: Optional[str] = None,
    ) -> None:
        check_timeout_and_retries(timeout, max_retries)
        self._sagemaker, self._logs, self._s3 = create_aws_clients(
            ("sagemaker", "logs", "s3"),
            region_name=region_name,
            timeout=float(timeout),
            max_retries=int(max_retries),
            user_agent_extra=user_agent or AWS_USER_AGENT_EXTRA,
        )
        #: The region every call goes to: the one passed, or the one boto3 resolved from the environment.
        self.region_name: str = self._sagemaker.meta.region_name
        self._reports: Dict[str, TrainingReport] = {}

    def __enter__(self) -> "SynthefyNoriFineTuner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the three AWS clients; a client that fails to close does not stop the others."""
        for client in (self._sagemaker, self._logs, self._s3):
            try:
                client.close()
            except Exception:
                _LOG.debug("closing an AWS client failed", exc_info=True)

    # ------------------------------------------------------------------ training
    def fine_tune(self, request: FineTuneJobRequest) -> FineTuneJob:
        """Start one training job and return at once, with the job as SageMaker first describes it.

        The request's checks run again first, in case its hyperparameters were changed after it was built. Then the
        Algorithm ARN's region is compared with this fine-tuner's, because SageMaker runs an Algorithm only in its
        own region and says so less clearly, and a channel given as one ``.csv`` object is refused when other keys
        start with its key, because SageMaker's channel would read them too. A training quota SageMaker refuses the
        instance under raises :class:`FineTuningError` naming that quota. When SageMaker accepted the job but the
        first status read fails, :class:`FineTuningError` carries the job's name and ARN: the job runs and bills,
        and ``get_status(job_name)`` reconnects to it.
        """
        request = revalidate_request(request)
        refuse_foreign_region(request.algorithm, self.region_name)
        for argument, uri in (("training_data", request.training_data), ("validation_data", request.validation_data)):
            if uri is not None and not uri.endswith("/"):
                refuse_csv_siblings(self._s3, uri, argument)
        job_name = request.job_name or build_job_name(request.hyperparameters.model_name, _utcnow())
        try:
            response = self._sagemaker.create_training_job(**build_training_job_request(request, job_name))
        except Exception as exc:
            code, message = read_aws_error(exc)
            if code != "ResourceLimitExceeded":
                raise
            raise FineTuningError(
                f"SageMaker refused training job {job_name}: the account's training quota has no "
                f"{request.instance_type} left ({message}). Ask for more of the '{request.instance_type} for "
                "training job usage' quota in Service Quotas (Amazon SageMaker), wait for a running job on that type "
                "to end, or pick another instance_type the Algorithm supports.",
                job_name=job_name,
                failure_reason=message,
            ) from exc
        job_arn = response.get("TrainingJobArn")
        _LOG.info(
            "started training job %s: %s on %s", job_name, request.hyperparameters.base_model, request.instance_type
        )
        try:
            return self.get_status(job_name)
        except Exception as exc:
            raise FineTuningError(
                f"SageMaker started training job {job_name} ({job_arn}), which runs and bills from now on, but "
                f"reading its status failed: {exc}. Call get_status({job_name!r}) to reconnect to it, or "
                f"stop({job_name!r}) to end it; calling fine_tune() again would start a second job.",
                job_name=job_name,
                job_arn=job_arn,
            ) from exc

    def get_status(self, job_name: str) -> FineTuneJob:
        """The job as SageMaker describes it now: one DescribeTrainingJob call."""
        return parse_job_description(self._describe_training_job(job_name))

    def wait(self, job_name: str, *, poll_s: float = _JOB_POLL_S, timeout_s: Optional[float] = None) -> FineTuneJob:
        """Poll the job until it is Completed, Failed or Stopped, and return it.

        A Failed job is returned, not raised: its ``failure_reason`` is the answer the caller waited for. Each status
        change is logged with SageMaker's message. Polls run every ``poll_s`` seconds (5 at least). The default
        deadline is the job's MaxRuntimeInSeconds plus two hours, because a job can wait for capacity for hours
        (SageMaker reports it as Pending). Past the deadline :class:`~synthefy.errors.APITimeoutError` is raised
        and the job keeps running.
        """
        description = self._describe_training_job(job_name)
        if timeout_s is None:
            max_runtime = description["StoppingCondition"]["MaxRuntimeInSeconds"]
            timeout_s = float(max_runtime + _WAIT_OVERHEAD_S)
        deadline = time.monotonic() + timeout_s
        interval = max(poll_s, _MIN_POLL_S)
        last: Optional[Tuple[str, Optional[str], Optional[str]]] = None
        while True:
            job = parse_job_description(description)
            current = (job.status, job.secondary_status, job.status_message)
            if current != last:
                _LOG.info(
                    "training job %s: %s / %s: %s", job_name, job.status, job.secondary_status, job.status_message
                )
                last = current
            if job.status in TERMINAL_JOB_STATUSES:
                return job
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise APITimeoutError(
                    f"training job {job_name} is still {job.status} after {format_seconds(timeout_s)} s; it keeps "
                    f"running and billing on SageMaker. Call get_status({job_name!r}) or wait() again later, or "
                    f"stop({job_name!r}) to end it."
                )
            time.sleep(min(interval, remaining))
            description = self._describe_training_job(job_name)

    def stop(self, job_name: str) -> FineTuneJob:
        """Ask SageMaker to stop the job, then return it as described right after.

        The job ships what it trained so far (the report says ``cut_short``), or nothing when it had not produced a
        model yet. A job that already ended or is already stopping is returned as is.
        """
        try:
            self._sagemaker.stop_training_job(TrainingJobName=job_name)
        except Exception as exc:
            if read_aws_error(exc)[0] != "ValidationException":
                raise
            job = self.get_status(job_name)
            if job.status in _NOTHING_TO_STOP_JOB_STATUSES:
                return job
            raise
        _LOG.info("requested StopTrainingJob for %s", job_name)
        return self.get_status(job_name)

    def get_logs(self, job_name: str, *, follow: bool = False, poll_s: float = _LOG_POLL_S) -> Iterator[str]:
        """The job's log lines from CloudWatch Logs, each once, in stream order.

        Without ``follow``, the lines logged so far. With ``follow``, new lines every ``poll_s`` seconds (5 at least)
        until the job reaches a terminal status, then one last read, so the job's final lines are never cut off.
        Nothing is yielded while the job has not started logging.
        """
        follower = TrainingLogFollower(self._logs, job_name)
        if not follow:
            yield from follower.read_new_lines()
            return
        interval = max(poll_s, _MIN_POLL_S)
        while True:
            ended = self._describe_training_job(job_name)["TrainingJobStatus"] in TERMINAL_JOB_STATUSES
            yield from follower.read_new_lines()
            if ended:
                return
            time.sleep(interval)

    def get_training_curves(self, job_name: str) -> Dict[str, List[Any]]:
        """Every validated step so far as columns ``child``, ``step``, ``train_loss`` and ``val_mse``.

        One row per step, so ``pandas.DataFrame(tuner.get_training_curves(name))`` takes it directly. Read from the
        job's log lines in every state: they carry every step, while SageMaker's metrics are sampled once a minute.
        When no log events remain (expired, or deleted), they are read from the report in the artifact.
        """
        follower = TrainingLogFollower(self._logs, job_name)
        lines = follower.read_new_lines()
        # Retention expires events without deleting their CloudWatch streams.
        # The follower exhausts every page, so an empty result can use the same
        # artifact fallback as a missing stream.
        if lines:
            return parse_curves_from_log_lines(lines)
        job = self.get_status(job_name)
        if job.artifact_uri is None:
            return create_empty_curves()
        return build_curves_from_report(self._read_report(job, job.artifact_uri))

    def get_training_report(self, job_name: str) -> TrainingReport:
        """The ``training_report.json`` of the job's artifact: the before and after scores, curves and cut_short.

        The artifact is streamed once and the report kept per artifact, so a second call reads no S3. A job without
        an artifact, or an artifact without a report, raises :class:`FineTuningError`.
        """
        job = self.get_status(job_name)
        if job.artifact_uri is None:
            reason = f": {job.failure_reason}" if job.failure_reason else ""
            raise FineTuningError(
                f"training job {job_name} is {job.status} and has no model artifact, so it has no training "
                f"report{reason}",
                job_name=job_name,
                status=job.status,
                failure_reason=job.failure_reason,
            )
        return self._read_report(job, job.artifact_uri)

    # ------------------------------------------------------------------ deployment
    def deploy(
        self,
        job: Union[FineTuneJob, str],
        *,
        endpoint_name: Optional[str] = None,
        instance_type: str = DEFAULT_ENDPOINT_INSTANCE_TYPE,
        async_output_uri: Optional[str] = None,
        role_arn: Optional[str] = None,
        wait: bool = True,
        timeout_s: float = ENDPOINT_TIMEOUT_S,
    ) -> FineTunedEndpoint:
        """Serve the job's artifact on an asynchronous endpoint and return it.

        Makes, in order: a model package from the job (named as the job), a SageMaker model on it (named as the
        endpoint), an endpoint config and the endpoint. ``endpoint_name`` defaults to the job's model name,
        ``role_arn`` to the job's execution role, and ``async_output_uri`` to ``<job output path>/<endpoint>/async``.

        Safe to call again after a failure: a package over this artifact, and a model on that package with the same
        execution role, are reused. A Completed job ships; a Stopped job ships what it trained, with a
        ``UserWarning`` when its report says ``cut_short``. A job in any other state raises
        :class:`FineTuningError`, as does a failure after something was made, whose ``created`` names it. An
        endpoint, package or model of the same name that is not this job's, a model left with another execution role,
        or an instance type the package does not serve, raises ``ValueError`` before anything more is made.
        """
        job_name = job.job_name if isinstance(job, FineTuneJob) else job
        description = self._describe_training_job(job_name)
        current = parse_job_description(description)
        artifact = find_deployable_artifact(self._s3, current)
        if not current.model_name:
            raise FineTuningError(
                f"training job {job_name} has no model_name hyperparameter, so it was not run by the Nori "
                "fine-tuning Algorithm",
                job_name=job_name,
                status=current.status,
            )
        endpoint_name = endpoint_name or current.model_name
        check_sagemaker_resource_name(endpoint_name, "endpoint_name")
        output_root = async_output_uri or (
            f"{description['OutputDataConfig']['S3OutputPath'].rstrip('/')}/{endpoint_name}/async"
        )
        parse_s3_uri(output_root, name="async_output_uri")
        output_root = output_root.rstrip("/")
        refuse_existing_endpoint(self._sagemaker, endpoint_name)
        cut_short = current.status == "Stopped" and self._read_report(current, artifact).cut_short
        if cut_short:
            warnings.warn(
                f"training job {job_name} was stopped before its children finished (the report says cut_short); "
                "the endpoint serves what they trained until then",
                UserWarning,
                stacklevel=2,
            )
        config_name = build_endpoint_config_name(endpoint_name, _utcnow())
        package_arn, status = create_deployment(
            self._sagemaker,
            job=current,
            artifact=artifact,
            endpoint_name=endpoint_name,
            config_name=config_name,
            instance_type=instance_type,
            output_root=output_root,
            role_arn=role_arn or description["RoleArn"],
            wait=wait,
            timeout_s=timeout_s,
        )
        return FineTunedEndpoint(
            endpoint_name=endpoint_name,
            model_name=current.model_name,
            instance_type=instance_type,
            status=status,
            model_package_arn=package_arn,
            sagemaker_model_name=endpoint_name,
            endpoint_config_name=config_name,
            async_output_uri=output_root,
            cut_short=cut_short,
            region_name=self.region_name,
        )

    def undeploy(self, endpoint: Union[FineTunedEndpoint, str]) -> None:
        """Delete an endpoint deploy() made, then its endpoint configs, then its SageMaker model; keep the package.

        The endpoint goes first and is waited for, because SageMaker refuses to delete a config or a model an
        endpoint still uses. The model package stays: it bills nothing, and the next deploy() of the same job reuses
        it instead of waiting for SageMaker to validate a new one. A resource already gone is skipped, so undeploy()
        can run again after a partial failure.
        """
        if isinstance(endpoint, FineTunedEndpoint):
            endpoint_name, model_name = endpoint.endpoint_name, endpoint.sagemaker_model_name
        else:
            endpoint_name = model_name = endpoint
        delete_deployment(self._sagemaker, endpoint_name=endpoint_name, model_name=model_name)

    def create_client(
        self,
        endpoint: FineTunedEndpoint,
        *,
        s3_staging_uri: str,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        user_agent: Optional[str] = None,
    ) -> SynthefyNoriClient:
        """A new SageMaker-mode :class:`~synthefy.SynthefyNoriClient` bound to the endpoint, its model name and region.

        ``s3_staging_uri`` is the caller's S3 prefix for request files; ``timeout``, ``max_retries`` and
        ``user_agent`` are :class:`~synthefy.SynthefyNoriClient`'s own arguments, with its defaults. Each call builds
        a client with its own AWS clients, which the caller closes (or uses as a context manager).
        """
        return SynthefyNoriClient(
            mode="sagemaker",
            model=endpoint.model_name,
            endpoint_name=endpoint.endpoint_name,
            region_name=endpoint.region_name,
            s3_staging_uri=s3_staging_uri,
            timeout=timeout,
            max_retries=max_retries,
            user_agent=user_agent,
        )

    # ------------------------------------------------------------------ helpers
    def _describe_training_job(self, job_name: str) -> Dict[str, Any]:
        return self._sagemaker.describe_training_job(TrainingJobName=job_name)

    def _read_report(self, job: FineTuneJob, uri: str) -> TrainingReport:
        """The artifact's report, read once per artifact URI."""
        cached = self._reports.get(uri)
        if cached is None:
            cached = self._reports[uri] = read_training_report(self._s3, uri=uri, job=job)
        return cached
