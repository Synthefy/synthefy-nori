"""Fine-tune Nori on Amazon SageMaker and deploy the result, all in the caller's own AWS account.

:class:`SynthefyNoriFineTuner` drives the Nori fine-tuning Algorithm (a SageMaker Algorithm resource, named by
ARN, or by name in the caller's own account) through a job's lifecycle: it starts one training job, reports its
status, log and curves while it runs, continues a stopped job from its checkpoint folder, reads the report its
artifact carries, deploys the artifact as an asynchronous endpoint, and builds the
:class:`~synthefy.SynthefyNoriClient` that predicts through it with ``mode="sagemaker"`` and the fine-tuned model name.

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
import secrets
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from synthefy._aws import (
    ACCESS_DENIED_CODES,
    AWS_USER_AGENT_EXTRA,
    FINE_TUNING_LOGGER_NAME,
    RESOURCE_NAME_TOKEN_CHARS,
    check_sagemaker_resource_name,
    check_timeout_and_retries,
    create_aws_clients,
    format_seconds,
    is_access_denied,
    is_missing_resource,
    parse_s3_uri,
    read_aws_error,
)
from synthefy._sagemaker_deploy import (
    DEFAULT_ENDPOINT_INSTANCE_TYPE,
    ENDPOINT_TIMEOUT_S,
    UNDEPLOY_TIMEOUT_S,
    build_endpoint_config_name,
    create_deployment,
    delete_deployment,
    find_deployable_artifact,
    refuse_existing_endpoint,
)
from synthefy._training_artifacts import (
    MAX_CHECKPOINT_FOLDER_KEYS,
    CheckpointFolderContents,
    describe_checkpoint_folder,
    read_checkpoint_folder,
    read_training_report,
)
from synthefy._training_jobs import (
    SAME_AS_JOB,
    build_job_name,
    build_request_from_job_description,
    build_training_job_request,
    LEGACY_DEFAULT_STEPS_PER_MODEL,
    FooterTooLong,
    TableRefused,
    get_declared_default_base_model,
    get_offered_base_models,
    get_supported_hyperparameter_names,
    get_request_with_version_default_base_model,
    get_training_plan_from_footers,
    has_resume_hyperparameter,
    has_whole_table_hyperparameter,
    parse_job_description,
    refuse_base_model_not_offered,
    refuse_task_settings_the_version_lacks,
    refuse_foreign_region,
    refuse_lookalike_keys,
    refuse_max_runtime_below_training_cap,
    refuse_table_pass_settings,
    refuse_version_specific_settings,
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
from synthefy.nori_data_models import (
    MAX_TRAINING_HOURS,
    FineTuneJob,
    FineTuneJobRequest,
    FineTunedEndpoint,
    TrainingPlan,
    TrainingReport,
)

_LOG = logging.getLogger(FINE_TUNING_LOGGER_NAME)

#: SageMaker's terminal training job statuses: a job in one of them holds no instance and changes no more.
TERMINAL_JOB_STATUSES = frozenset({"Completed", "Failed", "Stopped"})
#: Statuses in which StopTrainingJob has nothing left to do: the job ended, or a stop is already under way.
_NOTHING_TO_STOP_JOB_STATUSES = TERMINAL_JOB_STATUSES | {"Stopping"}
#: Statuses of a job that may still write to its checkpoint folder.
_RUNNING_JOB_STATUSES = frozenset({"InProgress", "Stopping"})
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


def _generate_resource_name_token() -> str:
    """The random characters every generated name ends with, from one place, so tests can pin those names."""
    return secrets.token_hex(RESOURCE_NAME_TOKEN_CHARS // 2)


def _has_models_left_to_train(report: TrainingReport) -> bool:
    """Whether a job's report leaves a model a resumed job would train: one cut short, failed, or ended or left
    unstarted by ``max_training_hours`` (a job the cap ended completes without being cut short)."""
    ended_by_cap = report.stop_reason == "max_training_hours" or any(
        curve.stop_reason == "max_training_hours" for curve in report.child_curves
    )
    return report.cut_short or bool(report.failed_children) or bool(report.unstarted_children) or ended_by_cap


def _is_unreadable_from_here(exc: BaseException) -> bool:
    """Whether a training plan failed because this client cannot read the channels' files, not because of the files:
    an AWS error (credentials missing or expired, access denied, an object gone, the network), pyarrow missing, or
    a footer longer than the plan reads."""
    is_aws_error = bool(read_aws_error(exc)[0]) or type(exc).__module__.partition(".")[0] == "botocore"
    return is_aws_error or isinstance(exc, (ImportError, FooterTooLong))


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
        #: What every client this fine-tuner makes is configured with, including one for another region (undeploy()).
        self._client_options: Dict[str, Any] = {
            "timeout": float(timeout),
            "max_retries": int(max_retries),
            "user_agent_extra": user_agent or AWS_USER_AGENT_EXTRA,
        }
        self._sagemaker, self._logs, self._s3 = create_aws_clients(
            ("sagemaker", "logs", "s3"), region_name=region_name, **self._client_options
        )
        #: The region every call goes to: the one passed, or the one boto3 resolved from the environment.
        self.region_name: str = self._sagemaker.meta.region_name
        self._reports: Dict[str, TrainingReport] = {}
        #: Whether each Algorithm this fine-tuner described trains each model on the whole table; None for one the
        #: caller may not describe.
        self._reads_whole_table_by_algorithm: Dict[str, Optional[bool]] = {}
        #: The ``base_model`` values each described Algorithm's version offers; None for one that lists none.
        self._offered_base_models_by_algorithm: Dict[str, Optional[Tuple[str, ...]]] = {}
        self._supported_hyperparameters_by_algorithm: Dict[str, Optional[frozenset]] = {}
        #: The ``base_model`` default each described Algorithm's version declares; None for one that declares none.
        self._default_base_model_by_algorithm: Dict[str, Optional[str]] = {}
        #: The error each Algorithm's last failed DescribeAlgorithm gave.
        self._describe_error_by_algorithm: Dict[str, str] = {}
        #: Whether each Algorithm this fine-tuner described accepts the resume hyperparameter.
        self._takes_resume_by_algorithm: Dict[str, bool] = {}

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
    def fine_tune(self, request: FineTuneJobRequest, *, overwrite: bool = False) -> FineTuneJob:
        """Start one training job and return at once, with the job as SageMaker first describes it.

        A request with ``checkpoint_uri`` has the job save each model it finishes, and each model's training state as
        it goes, in that S3 folder, which must be empty: the folder is listed before the job is created, and one that
        holds files raises :class:`FineTuningError` naming its finished models, the step each other model reached and
        the job that wrote them. :meth:`resume` continues that job. ``overwrite=True`` deletes the folder's checkpoint
        files right before the job is created, so the job trains every model from the start; it deletes nothing and
        raises :class:`FineTuningError` when the folder holds a file the job does not write or more than 1,000 files,
        or when the job that wrote its newest record is still running. A folder that holds only files a job does not
        write raises :class:`FineTuningError` saying it is not a checkpoint folder. ``overwrite=True`` without
        ``checkpoint_uri`` raises ``ValueError``.

        One job saves to a folder at a time: right before any file is deleted and the job is created, the account's
        InProgress and Stopping training jobs are listed, and one whose checkpoint folder is this one raises
        :class:`FineTuningError` naming it, with nothing deleted and no job created. Two calls made at the same
        moment can both pass that check. The folder's checks need ``s3:ListBucket`` and ``s3:GetObject`` on it,
        ``sagemaker:ListTrainingJobs`` and ``sagemaker:DescribeTrainingJob``, and ``overwrite=True`` needs
        ``s3:DeleteObject``; a call AWS refuses raises :class:`FineTuningError` naming the permission, and no job is
        created.

        The request's checks run again first, in case its hyperparameters were changed after it was built. Then the
        Algorithm ARN's region is compared with this fine-tuner's, because SageMaker runs an Algorithm only in its
        own region and says so less clearly, and a channel given as one object is refused when other keys start with
        its key, because SageMaker's channel would read them too. A training quota SageMaker refuses the
        instance under raises :class:`FineTuningError` naming that quota. When SageMaker accepted the job but the
        first status read fails, :class:`FineTuningError` carries the job's name and ARN: the job runs and bills,
        and ``get_status(job_name)`` reconnects to it.

        Before submitting, it works out the job's :class:`~synthefy.TrainingPlan` (:meth:`get_training_plan`), logs
        it and attaches it to the returned job; tables the job would refuse raise ``ValueError`` before the job is
        created. Any other failure to work out the plan (an AWS error such as missing credentials or denied access,
        pyarrow missing, a footer above 256 MiB) logs a warning, and the job is submitted without a plan. A plan whose
        training hours leave the job less than the 30 minutes it needs besides training within SageMaker's 120-hour
        limit on a training job, with no ``checkpoint_uri`` to continue from, warns. A ``base_model`` the Algorithm's
        version does not offer raises :class:`FineTuningError` naming the bases it offers. On an Algorithm version that
        trains each model for a fixed number of steps, a setting only a newer version reads raises
        :class:`FineTuningError`. When the Algorithm cannot be described, the job gets only the settings every version
        of the Algorithm reads, and a request that sets ``row_partition_strategy``, ``n_passes_per_row``,
        ``max_training_hours``, ``validation_every_n_steps``, ``n_steps_per_model`` or ``patience=None`` is refused,
        naming each. Unless the Algorithm's version trains each model for a fixed number of steps, a ``max_runtime_s``
        set below ``max_training_hours`` x 3,600 + 1,800 s raises ``ValueError``, since SageMaker would stop the job
        before the cap ends its training.
        """
        if overwrite and request.checkpoint_uri is None:
            raise ValueError(
                "overwrite=True deletes the files in the request's checkpoint_uri, and the request names none; set "
                "checkpoint_uri, or leave overwrite unset"
            )
        request, trains_on_whole_table = self._get_checked_request_and_version(request)
        if trains_on_whole_table is False:
            _LOG.warning(
                "Algorithm %s trains each model for a fixed number of steps (%s), not on the whole table; a newer "
                "version of the Algorithm trains on the whole table",
                request.algorithm,
                request.hyperparameters.n_steps_per_model or LEGACY_DEFAULT_STEPS_PER_MODEL,
            )
        folder: Optional[CheckpointFolderContents] = None
        if request.checkpoint_uri is not None:
            folder = self._read_checkpoint_folder(request.checkpoint_uri, job_name=None)
            if folder.keys and len(folder.foreign_keys) == len(folder.keys):
                bucket, _ = parse_s3_uri(folder.uri, name="checkpoint_uri")
                count = len(folder.keys)
                raise FineTuningError(
                    f"checkpoint_uri {folder.uri} is not a checkpoint folder: it holds {count} "
                    f"file{'' if count == 1 else 's'}, none of them one a fine-tuning job writes, the first "
                    f"s3://{bucket}/{folder.foreign_keys[0]}. Nothing was deleted and no job was created; give the "
                    "request a checkpoint_uri that is an empty folder."
                )
            if folder.keys and not overwrite:
                resume_call = (
                    f"resume({folder.newest_record_job_name!r})"
                    if folder.newest_record_job_name is not None
                    else "resume() with the name of the job that wrote them"
                )
                raise FineTuningError(
                    f"checkpoint_uri {describe_checkpoint_folder(folder)}, and a new job needs an empty folder. To "
                    f"continue the job that wrote them, call {resume_call}; to delete them and train every model from "
                    "the start, call fine_tune(request, overwrite=True); or give the request another checkpoint_uri, "
                    "an empty folder."
                )
            if folder.keys:
                self._refuse_checkpoint_folder_overwrite(folder)
        plan = self._get_plan_before_submitting(request, trains_on_whole_table)
        job_name = request.job_name or build_job_name(
            request.hyperparameters.model_name, _utcnow(), _generate_resource_name_token()
        )
        deleted_checkpoint_file_count = 0
        if folder is not None:
            # Read last, right before the delete and the create: a job started meanwhile is in the listing.
            running = self._find_running_job_on_checkpoint_folder(folder.uri, job_name=None)
            if running is not None:
                raise FineTuningError(
                    f"training job {running[0]} is {running[1]} and saves to checkpoint_uri {folder.uri}, and a "
                    "second job on the folder would write the same files; nothing was deleted and no job was "
                    f"created. Wait for it to end, or stop({running[0]!r}) it, then call again; or give the request "
                    "another checkpoint_uri.",
                    job_name=running[0],
                    status=running[1],
                )
            if folder.keys:
                self._delete_checkpoint_files(folder)
                deleted_checkpoint_file_count = len(folder.keys)
        return self._create_training_job(
            request,
            job_name,
            trains_on_whole_table=trains_on_whole_table,
            resume=False,
            plan=plan,
            deleted_checkpoint_file_count=deleted_checkpoint_file_count,
        )

    def resume(
        self,
        job: Union[FineTuneJob, str],
        *,
        instance_type: Optional[str] = None,
        max_training_hours: Optional[float] = SAME_AS_JOB,
    ) -> FineTuneJob:
        """Start a training job that continues ``job`` from its checkpoint folder, and return it as SageMaker first
        describes it.

        The new job runs ``job``'s request again under a new job name: the same Algorithm, tables, hyperparameters
        (``model_name`` included), role, output path and checkpoint folder. It keeps every model the folder holds
        finished and continues each other model from its last saved state. ``instance_type`` replaces the job's
        instance type; ``max_training_hours`` replaces the job's cap (None: no cap), and the job's runtime limit is
        then derived from it as a new request derives it; otherwise the job's runtime limit is kept. The request's
        checks run as in :meth:`fine_tune`. The job's tags are not copied to the new job.

        Before submitting, it works out the request's :class:`~synthefy.TrainingPlan` as :meth:`fine_tune` does,
        logs it and attaches it to the returned job. The plan is the whole request's, every model trained from its
        first step, so it is the most the new job takes: that job trains only what the folder does not hold.
        Tables the job would refuse raise ``ValueError`` before the job is created; any other failure to work out
        the plan logs a warning, and the job is submitted without a plan.

        Raises :class:`FineTuningError` naming the job and what to do when ``job`` is still running (InProgress or
        Stopping), has no checkpoint folder, or its folder is empty or holds no file a fine-tuning job writes; when
        another InProgress or Stopping training job of the account saves to the folder, such as the job an earlier
        ``resume()`` of ``job`` started; when ``job`` was created with managed spot training, a VPC or a volume KMS
        key, which a request does not carry; when ``job`` completed and its training report cannot be read, or says
        every model trained to its own end, none cut short, failed, or ended or left unstarted by
        ``max_training_hours``, which leaves nothing to continue; and when the Algorithm cannot be described, so its
        version is unknown. An Algorithm version that trains each model for a fixed number of steps has no
        ``max_training_hours`` and refuses one with :class:`FineTuningError`. An ``instance_type`` or
        ``max_training_hours`` a request cannot take raises ``ValueError``. The checks need ``s3:ListBucket`` and
        ``s3:GetObject`` on the folder, ``sagemaker:ListTrainingJobs`` and ``sagemaker:DescribeTrainingJob``; the
        plan needs ``s3:ListBucket`` and ``s3:GetObject`` on the tables.
        """
        job_name = job.job_name if isinstance(job, FineTuneJob) else job
        description = self._describe_training_job(job_name)
        described = parse_job_description(description)

        def build_refusal(message: str) -> FineTuningError:
            return FineTuningError(
                message, job_name=job_name, status=described.status, failure_reason=described.failure_reason
            )

        if described.status in _RUNNING_JOB_STATUSES:
            raise build_refusal(
                f"training job {job_name} is {described.status}, and a second job on its checkpoint folder would "
                f"write the same files; wait({job_name!r}) for it to end, or stop({job_name!r}) it, then call "
                "resume() again"
            )
        uri = (description.get("CheckpointConfig") or {}).get("S3Uri")
        if uri is None:
            raise build_refusal(
                f"training job {job_name} has no checkpoint_uri, so it saved nothing to continue from; start a new "
                "job with fine_tune(), with a checkpoint_uri so a later job can continue it"
            )
        # Before the folder and report reads: an override the request cannot take is the caller's mistake.
        request = build_request_from_job_description(
            description, instance_type=instance_type, max_training_hours=max_training_hours
        )
        folder = self._read_checkpoint_folder(uri, job_name=job_name)
        if not folder.keys:
            raise build_refusal(
                f"the checkpoint folder {uri} of training job {job_name} is empty, so there is nothing to continue; "
                "start a new job with fine_tune(), which trains every model from the start"
            )
        if len(folder.foreign_keys) == len(folder.keys):
            raise build_refusal(
                f"the checkpoint folder {uri} of training job {job_name} holds no file a fine-tuning job writes, so "
                "there is nothing to continue; start a new job with fine_tune() and a checkpoint_uri that is an empty "
                "folder"
            )
        if folder.newest_record_job_name is not None and folder.newest_record_job_name != job_name:
            status = self._get_newest_record_job_status(folder.newest_record_job_name, folder.uri, job_name=job_name)
            if status in _RUNNING_JOB_STATUSES:
                raise build_refusal(
                    f"cannot resume training job {job_name}: {describe_checkpoint_folder(folder)}, and training job "
                    f"{folder.newest_record_job_name} is {status}; wait for it to end, or "
                    f"stop({folder.newest_record_job_name!r}) it, then call resume({folder.newest_record_job_name!r})"
                )
        if described.status == "Completed" and described.artifact_uri is not None:
            try:
                report = self._read_report(described, described.artifact_uri)
            except Exception as exc:
                code, message = read_aws_error(exc)
                advice = (
                    "let the identity that runs this code call s3:GetObject on it, then call resume() again"
                    if is_access_denied(exc)
                    else "call get_training_report() to see the error, then call resume() again"
                )
                raise build_refusal(
                    f"cannot resume training job {job_name}: it completed, and its training report in "
                    f"{described.artifact_uri} could not be read ({code or type(exc).__name__}: {message}), so "
                    f"whether it left a model to train is unknown and no job was created; {advice}"
                ) from exc
            if not _has_models_left_to_train(report):
                raise build_refusal(
                    f"training job {job_name} completed, and its training report says no model was cut short, "
                    "failed, or ended or left unstarted by max_training_hours, so there is nothing to continue; "
                    f"deploy({job_name!r}) deploys its model, and fine_tune() with another checkpoint_uri trains a "
                    "new one"
                )
        # Before the shared checks: the rebuilt request sets every setting, which they refuse for an unknown version.
        if self._is_whole_table_algorithm(request.algorithm) is None:
            raise build_refusal(
                f"cannot resume training job {job_name}: Algorithm {request.algorithm} could not be described "
                f"({self._describe_error_by_algorithm[request.algorithm]}), so its version is unknown, and whether "
                "it continues the job is unknown too; let the identity that runs this code call "
                "sagemaker:DescribeAlgorithm on the Algorithm, then call resume() again"
            )
        request, trains_on_whole_table = self._get_checked_request_and_version(request)
        plan = self._get_plan_before_submitting(request, trains_on_whole_table, is_resumed=True)
        # Read last, right before the create: the job an earlier resume() started is in the listing.
        running = self._find_running_job_on_checkpoint_folder(uri, job_name=job_name)
        if running is not None:
            raise build_refusal(
                f"cannot resume training job {job_name}: training job {running[0]} is {running[1]} and saves to the "
                f"same checkpoint folder {uri}, and a second job on the folder would write the same files; wait for "
                f"it to end, or stop({running[0]!r}) it, then call resume({running[0]!r})"
            )
        new_job_name = build_job_name(request.hyperparameters.model_name, _utcnow(), _generate_resource_name_token())
        _LOG.info(
            "training job %s continues training job %s: %s", new_job_name, job_name, describe_checkpoint_folder(folder)
        )
        return self._create_training_job(
            request,
            new_job_name,
            trains_on_whole_table=trains_on_whole_table,
            resume=self._takes_resume_by_algorithm.get(request.algorithm, False),
            plan=plan,
        )

    def get_training_plan(self, request: FineTuneJobRequest) -> TrainingPlan:
        """What the job would read and what it would take, before it is submitted: the rows each model trains on,
        the steps, the share of the table read, the training hours and their estimated cost.

        It reads only the footers of the channels' Parquet files, no row, and describes the
        Algorithm once, so the caller's credentials need ``s3:ListBucket`` and ``s3:GetObject`` on the channels and
        ``sagemaker:DescribeAlgorithm``. The width of a featurized row is estimated from the column count, so
        ``rows_per_step`` and what follows from it are estimates (``rows_per_step_is_estimate``); the job's own plan,
        in its log and its training report, uses the measured width. The same request goes to :meth:`fine_tune`.

        A job of tasks (``tasks``, ``task_column``, ``time_column``, ``context_rows`` or ``query_rows``) has no plan
        before it runs: its job reads the tables whole and logs its plan then, so this raises ``ValueError`` once the
        tables pass the checks below.

        The request's checks run first, as in :meth:`fine_tune`: among them, a channel given as one object when other
        keys start with its key raises ``ValueError``, since SageMaker's channel would read them too. Raises
        ``ValueError`` when the job would refuse the tables: a channel with no Parquet file or with a file of another
        extension, a ``target_column`` or ``feature_columns`` entry the training table lacks, a validation table with
        other columns, a file that names a column twice, a ``.parquet`` or ``.pq`` file that cannot be read as Parquet
        or whose pandas metadata cannot be applied, or too few rows for each model; and when a file's footer is above
        256 MiB, a table :meth:`fine_tune` submits without a plan. A ``base_model`` the Algorithm's version does not
        offer raises :class:`FineTuningError` naming the bases it offers.

        When the Algorithm cannot be described, the job gets only the settings every version of the Algorithm reads,
        and a request that sets ``row_partition_strategy``, ``n_passes_per_row``, ``max_training_hours``,
        ``validation_every_n_steps``, ``n_steps_per_model`` or ``patience=None`` is refused, naming each. The plan is
        then the newest version's, and one of its warnings says so.
        """
        request, trains_on_whole_table = self._get_checked_request_and_version(request)
        plan = get_training_plan_from_footers(self._s3, request, trains_on_whole_table=trains_on_whole_table)
        if plan is None:
            raise ValueError(
                "a job of tasks reads its tables whole and logs its plan once it runs; "
                "there is no plan before it is submitted"
            )
        return plan

    def get_status(self, job_name: str) -> FineTuneJob:
        """The job as SageMaker describes it now: one DescribeTrainingJob call."""
        return parse_job_description(self._describe_training_job(job_name))

    def wait(self, job_name: str, *, poll_s: float = _JOB_POLL_S, timeout_s: Optional[float] = None) -> FineTuneJob:
        """Poll the job until it is Completed, Failed or Stopped, and return it.

        A Failed job is returned, not raised: its ``failure_reason`` is the answer the caller waited for. Each status
        change is logged with SageMaker's message, and a Completed job's training report summary
        (:meth:`~synthefy.TrainingReport.get_summary`) is logged when the job ends. Polls run every ``poll_s`` seconds
        (5 at least). The default deadline is the job's MaxRuntimeInSeconds plus two hours, because a job can wait for
        capacity for hours (SageMaker reports it as Pending). Past the deadline
        :class:`~synthefy.errors.APITimeoutError` is raised and the job keeps running.
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
                if job.status == "Completed" and job.artifact_uri is not None:
                    self._log_report_summary(job, job.artifact_uri)
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
        model yet. With ``checkpoint_uri``, the models it finished and the states it saved stay in that folder, and
        :meth:`resume` continues the job from them. A job that already ended or is already stopping is returned as is.
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
        When no log events remain (expired, or deleted), they are read from the report in the artifact. A job that
        resumed from a ``checkpoint_uri`` logs only the children it trained; its report's ``child_curves`` holds every
        child, with ``restored`` marking the others.
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
        config_name = build_endpoint_config_name(endpoint_name, _utcnow(), _generate_resource_name_token())
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

    def undeploy(self, endpoint: Union[FineTunedEndpoint, str], *, timeout_s: float = UNDEPLOY_TIMEOUT_S) -> None:
        """Delete an endpoint deploy() made, then its endpoint configs, then its SageMaker model; keep the package.

        The endpoint goes first and is waited for, because SageMaker refuses to delete a config or a model an
        endpoint still uses. The model package stays: it bills nothing, and the next deploy() of the same job reuses
        it instead of waiting for SageMaker to validate a new one. A resource already gone is skipped, so undeploy()
        can run again after a partial failure. A :class:`~synthefy.FineTunedEndpoint` is deleted in its own region,
        the way :meth:`create_client` predicts through it; a name is deleted in this fine-tuner's region.

        SageMaker cannot delete an endpoint while it is Creating, Updating, SystemUpdating or RollingBack, so
        undeploy() waits up to ``timeout_s`` for it to leave that status, then deletes it; an endpoint still in it
        after ``timeout_s`` raises :class:`APITimeoutError`, and undeploy() can be called again later.
        """
        if isinstance(endpoint, FineTunedEndpoint):
            endpoint_name, model_name = endpoint.endpoint_name, endpoint.sagemaker_model_name
            region_name = endpoint.region_name
        else:
            endpoint_name = model_name = endpoint
            region_name = self.region_name
        if region_name == self.region_name:
            delete_deployment(
                self._sagemaker, endpoint_name=endpoint_name, model_name=model_name, settle_timeout_s=timeout_s
            )
            return
        # Deleting by name through this fine-tuner's client would hit same-named resources in its region and leave
        # the handle's endpoint running, so the deletes go through a client for the handle's region.
        (sagemaker,) = create_aws_clients(("sagemaker",), region_name=region_name, **self._client_options)
        try:
            delete_deployment(sagemaker, endpoint_name=endpoint_name, model_name=model_name, settle_timeout_s=timeout_s)
        finally:
            sagemaker.close()

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
    def _get_checked_request_and_version(
        self, request: FineTuneJobRequest
    ) -> Tuple[FineTuneJobRequest, Optional[bool]]:
        """The request with its checks run again, and whether its Algorithm trains each model on the whole table, None
        when the Algorithm cannot be described. Refuses, before any table is read, an Algorithm in another region, a
        channel given as one object when other keys start with its key, a ``base_model`` the Algorithm's version does
        not offer (an unset one first takes the version's declared default), a job-of-tasks setting the described
        version does not read, a setting the Algorithm's version does not read or, when the version is unknown, a
        setting its versions read differently, and, unless the version trains each model for a fixed number of steps,
        a ``max_runtime_s`` that ends the job before ``max_training_hours`` does."""
        set_by_caller = request.hyperparameters.model_fields_set
        request = revalidate_request(request)
        refuse_foreign_region(request.algorithm, self.region_name)
        for argument, uri in (("training_data", request.training_data), ("validation_data", request.validation_data)):
            if uri is not None and not uri.endswith("/"):
                refuse_lookalike_keys(self._s3, uri, argument)
        trains_on_whole_table = self._is_whole_table_algorithm(request.algorithm)
        request = get_request_with_version_default_base_model(
            request, set_by_caller, self._default_base_model_by_algorithm.get(request.algorithm)
        )
        refuse_base_model_not_offered(
            request.hyperparameters, request.algorithm, self._offered_base_models_by_algorithm.get(request.algorithm)
        )
        refuse_task_settings_the_version_lacks(
            request.hyperparameters,
            request.algorithm,
            self._supported_hyperparameters_by_algorithm.get(request.algorithm),
        )
        if trains_on_whole_table is None:
            refuse_version_specific_settings(
                request.hyperparameters,
                request.algorithm,
                set_by_caller=set_by_caller,
                reason=self._describe_error_by_algorithm[request.algorithm],
            )
        elif trains_on_whole_table is False:
            refuse_table_pass_settings(request.hyperparameters, request.algorithm, set_by_caller=set_by_caller)
        if trains_on_whole_table is not False:
            refuse_max_runtime_below_training_cap(request, version_known=trains_on_whole_table is True)
        return request, trains_on_whole_table

    def _is_whole_table_algorithm(self, algorithm: str) -> Optional[bool]:
        """Whether ``algorithm`` trains each model on the whole table (True) or for a fixed number of steps (False);
        None when it cannot be described. One DescribeAlgorithm per Algorithm, kept; an access denial, which lasts,
        is kept with its error, and any other failure is not, so the next call describes the Algorithm again. A
        described Algorithm's offered ``base_model`` values are kept too."""
        if algorithm not in self._reads_whole_table_by_algorithm:
            try:
                description = self._sagemaker.describe_algorithm(AlgorithmName=algorithm)
            except Exception as exc:
                _LOG.warning(
                    "could not describe Algorithm %s, so its version is unknown and the job gets only the settings "
                    "every version reads: %s",
                    algorithm,
                    exc,
                )
                self._describe_error_by_algorithm[algorithm] = str(exc)
                if read_aws_error(exc)[0] == "AccessDeniedException":
                    self._reads_whole_table_by_algorithm[algorithm] = None
                return None
            self._offered_base_models_by_algorithm[algorithm] = get_offered_base_models(description)
            self._supported_hyperparameters_by_algorithm[algorithm] = get_supported_hyperparameter_names(description)
            self._default_base_model_by_algorithm[algorithm] = get_declared_default_base_model(description)
            self._reads_whole_table_by_algorithm[algorithm] = has_whole_table_hyperparameter(description)
            self._takes_resume_by_algorithm[algorithm] = has_resume_hyperparameter(description)
        return self._reads_whole_table_by_algorithm[algorithm]

    def _get_plan_before_submitting(
        self, request: FineTuneJobRequest, trains_on_whole_table: Optional[bool], *, is_resumed: bool = False
    ) -> Optional[TrainingPlan]:
        """The plan :meth:`fine_tune` and :meth:`resume` log and attach; None, with a warning, when the plan cannot
        be worked out. The plan is advice, so only a table the job would refuse (:class:`TableRefused`) stops the
        request. A job with a job of tasks gets None once its tables pass the checks: it works out its plan once it
        has read the table. ``is_resumed`` says the job continues a checkpoint folder: the plan is the whole
        request's, the most that job takes, and the log line says so."""
        try:
            plan = get_training_plan_from_footers(self._s3, request, trains_on_whole_table=trains_on_whole_table)
        except TableRefused:
            raise
        except Exception as exc:
            if _is_unreadable_from_here(exc):
                _LOG.warning(
                    "could not read the Parquet footers of the channels, so the job is submitted without a plan: %s",
                    exc,
                )
            else:
                _LOG.warning(
                    "could not work out the training plan, so the job is submitted without a plan", exc_info=exc
                )
            return None
        if plan is None:
            _LOG.info("a job of tasks logs its plan once it has read the tables")
            return None
        if is_resumed:
            _LOG.info(
                "training plan of the whole request, the most this job takes (it trains only what the checkpoint "
                "folder does not hold):\n%s",
                plan.get_summary(),
            )
        else:
            _LOG.info("training plan:\n%s", plan.get_summary())
        hours = plan.training_hours
        if plan.max_training_hours is not None and hours is not None:
            hours = min(hours, plan.max_training_hours)
        if hours is not None and hours > MAX_TRAINING_HOURS and request.checkpoint_uri is None:
            warnings.warn(
                f"the plan trains for {hours:,.1f} hours, which with the 30 minutes the job needs besides training "
                "passes SageMaker's limit of 120 hours on one training job; the job then stops before every model "
                "trains and saves nothing; set checkpoint_uri so resume() continues the job from the models saved, "
                "or lower n_passes_per_row, n_steps_per_model or bag",
                UserWarning,
                stacklevel=3,
            )
        return plan

    def _describe_training_job(self, job_name: str) -> Dict[str, Any]:
        return self._sagemaker.describe_training_job(TrainingJobName=job_name)

    def _create_training_job(
        self,
        request: FineTuneJobRequest,
        job_name: str,
        *,
        trains_on_whole_table: Optional[bool],
        resume: bool,
        plan: Optional[TrainingPlan],
        deleted_checkpoint_file_count: int = 0,
    ) -> FineTuneJob:
        """Create the training job and return it as first described, with ``plan`` attached. A training quota
        SageMaker refuses the instance under raises :class:`FineTuningError` naming that quota; a first status read
        that fails raises :class:`FineTuningError` carrying the job's name and ARN. When ``overwrite=True`` deleted
        ``deleted_checkpoint_file_count`` files from the checkpoint folder first, any refusal of the job raises
        :class:`FineTuningError` saying so."""
        try:
            response = self._sagemaker.create_training_job(
                **build_training_job_request(
                    request, job_name, trains_on_whole_table=trains_on_whole_table, resume=resume
                )
            )
        except Exception as exc:
            code, message = read_aws_error(exc)
            deleted = (
                f" The {deleted_checkpoint_file_count} files in checkpoint_uri {request.checkpoint_uri} were deleted; "
                "call fine_tune(request) again, without overwrite=True."
                if deleted_checkpoint_file_count
                else ""
            )
            if code != "ResourceLimitExceeded":
                if not deleted:
                    raise
                raise FineTuningError(
                    f"SageMaker refused training job {job_name} ({code or type(exc).__name__}: {message}).{deleted}",
                    job_name=job_name,
                    failure_reason=message,
                ) from exc
            raise FineTuningError(
                f"SageMaker refused training job {job_name}: the account's training quota has no "
                f"{request.instance_type} left ({message}). Ask for more of the '{request.instance_type} for "
                "training job usage' quota in Service Quotas (Amazon SageMaker), wait for a running job on that type "
                f"to end, or pick another instance_type the Algorithm supports.{deleted}",
                job_name=job_name,
                failure_reason=message,
            ) from exc
        job_arn = response.get("TrainingJobArn")
        _LOG.info(
            "started training job %s: %s on %s", job_name, request.hyperparameters.base_model, request.instance_type
        )
        try:
            return self.get_status(job_name).model_copy(update={"training_plan": plan})
        except Exception as exc:
            raise FineTuningError(
                f"SageMaker started training job {job_name} ({job_arn}), which runs and bills from now on, but "
                f"reading its status failed: {exc}. Call get_status({job_name!r}) to reconnect to it, or "
                f"stop({job_name!r}) to end it; calling fine_tune() or resume() again would start a second job.",
                job_name=job_name,
                job_arn=job_arn,
            ) from exc

    def _read_checkpoint_folder(self, uri: str, *, job_name: Optional[str]) -> CheckpointFolderContents:
        """What the checkpoint folder holds (:func:`read_checkpoint_folder`); a listing or a record read that fails,
        or a record that is gone by the time it is read, raises :class:`FineTuningError`, which names
        ``s3:ListBucket`` and ``s3:GetObject`` when S3 denied access."""
        try:
            return read_checkpoint_folder(self._s3, uri)
        except Exception as exc:
            code, message = read_aws_error(exc)
            advice = (
                "; let the identity that runs this code call s3:ListBucket on the folder's bucket and s3:GetObject "
                "on the folder, then call again"
                if is_access_denied(exc)
                else ""
            )
            raise FineTuningError(
                f"could not read the checkpoint folder {uri} ({code or type(exc).__name__}: {message}), so nothing "
                f"was deleted and no job was created{advice}",
                job_name=job_name,
            ) from exc

    def _find_running_job_on_checkpoint_folder(self, uri: str, *, job_name: Optional[str]) -> Optional[Tuple[str, str]]:
        """The name and status of an InProgress or Stopping training job of the account whose checkpoint folder is
        ``uri`` (a trailing ``/`` aside), or None: a job that has saved nothing yet is named by no record in the
        folder. A listing or a describe that fails raises :class:`FineTuningError`, which names
        ``sagemaker:ListTrainingJobs`` and ``sagemaker:DescribeTrainingJob`` when SageMaker denied access."""
        folder_uri = uri.rstrip("/")
        try:
            for status in sorted(_RUNNING_JOB_STATUSES):
                page: Dict[str, Any] = {"StatusEquals": status, "MaxResults": 100}
                while True:
                    response = self._sagemaker.list_training_jobs(**page)
                    for summary in response.get("TrainingJobSummaries") or []:
                        try:
                            description = self._describe_training_job(summary["TrainingJobName"])
                        except Exception as exc:
                            if is_missing_resource(exc):
                                continue
                            raise
                        job_uri = (description.get("CheckpointConfig") or {}).get("S3Uri")
                        job_status = description.get("TrainingJobStatus")
                        if job_uri is None or job_uri.rstrip("/") != folder_uri:
                            continue
                        if job_status in _RUNNING_JOB_STATUSES:
                            return summary["TrainingJobName"], job_status
                    if not response.get("NextToken"):
                        break
                    page["NextToken"] = response["NextToken"]
        except Exception as exc:
            code, message = read_aws_error(exc)
            advice = (
                "; let the identity that runs this code call sagemaker:ListTrainingJobs and "
                "sagemaker:DescribeTrainingJob, then call again"
                if is_access_denied(exc)
                else ""
            )
            raise FineTuningError(
                f"could not list the training jobs that save to the checkpoint folder {uri} "
                f"({code or type(exc).__name__}: {message}), so nothing was deleted and no job was created{advice}",
                job_name=job_name,
            ) from exc
        return None

    def _get_newest_record_job_status(
        self, writer_job_name: str, uri: str, *, job_name: Optional[str]
    ) -> Optional[str]:
        """The status of ``writer_job_name``, the job that wrote the newest record in the checkpoint folder ``uri``;
        None when SageMaker does not know it."""
        try:
            return self._describe_training_job(writer_job_name)["TrainingJobStatus"]
        except Exception as exc:
            if is_missing_resource(exc):
                return None
            raise FineTuningError(
                f"could not read the status of training job {writer_job_name}, which wrote the newest record in "
                f"{uri}, so nothing was deleted and no job was created: {exc}",
                job_name=job_name,
            ) from exc

    def _refuse_checkpoint_folder_overwrite(self, folder: CheckpointFolderContents) -> None:
        """Raise :class:`FineTuningError` when ``fine_tune(..., overwrite=True)`` must not empty the folder: it holds
        a file the job does not write or more files than one listing reads, or the job that wrote its newest record
        is InProgress or Stopping."""
        bucket, _ = parse_s3_uri(folder.uri, name="checkpoint_uri")
        if folder.foreign_keys:
            raise FineTuningError(
                "overwrite=True deletes only the files a fine-tuning job writes, and "
                f"{describe_checkpoint_folder(folder)}, among them s3://{bucket}/{folder.foreign_keys[0]}; nothing "
                "was deleted and no job was created. Move or delete the files the job does not write, or give the "
                "request another checkpoint_uri."
            )
        if folder.truncated:
            raise FineTuningError(
                f"checkpoint_uri {folder.uri} holds more than {MAX_CHECKPOINT_FOLDER_KEYS:,} files, more than "
                "overwrite=True reads before it deletes, so nothing was deleted and no job was created. Empty the "
                "folder another way, or give the request another checkpoint_uri."
            )
        if folder.newest_record_job_name is not None:
            status = self._get_newest_record_job_status(folder.newest_record_job_name, folder.uri, job_name=None)
            if status in _RUNNING_JOB_STATUSES:
                raise FineTuningError(
                    f"{describe_checkpoint_folder(folder)}, and training job {folder.newest_record_job_name} is "
                    f"{status}, so overwrite=True would delete files a running job writes; nothing was deleted and no "
                    f"job was created. Wait for it to end, or stop({folder.newest_record_job_name!r}) it, then call "
                    "again.",
                    job_name=folder.newest_record_job_name,
                    status=status,
                )

    def _delete_checkpoint_files(self, folder: CheckpointFolderContents) -> None:
        """Delete every file the folder holds with one DeleteObjects call, for ``fine_tune(..., overwrite=True)``,
        once :meth:`_refuse_checkpoint_folder_overwrite` let it."""
        bucket, _ = parse_s3_uri(folder.uri, name="checkpoint_uri")
        keys = list(folder.keys)
        try:
            response = self._s3.delete_objects(
                Bucket=bucket, Delete={"Objects": [{"Key": key} for key in keys], "Quiet": True}
            )
        except Exception as exc:
            code, message = read_aws_error(exc)
            advice = (
                "; let the identity that runs this code call s3:DeleteObject on the folder, then call again"
                if is_access_denied(exc)
                else ""
            )
            raise FineTuningError(
                f"deleting the {len(keys)} files in {folder.uri}, the first s3://{bucket}/{keys[0]}, failed "
                f"({code or type(exc).__name__}: {message}), and some of them may be deleted, so no job was "
                f"created{advice}"
            ) from exc
        errors = response.get("Errors") or []
        if errors:
            first = errors[0]
            advice = (
                "; let the identity that runs this code call s3:DeleteObject on the folder, then call again"
                if first.get("Code") in ACCESS_DENIED_CODES
                else ""
            )
            raise FineTuningError(
                f"S3 refused to delete s3://{bucket}/{first.get('Key')} ({first.get('Code')}: {first.get('Message')})"
                f", so no job was created, and {len(keys) - len(errors)} of the folder's {len(keys)} files are "
                f"deleted{advice}"
            )
        _LOG.info("deleted the %s files in %s for overwrite=True", len(keys), folder.uri)

    def _log_report_summary(self, job: FineTuneJob, uri: str) -> None:
        """Log the report's summary; a report that cannot be read is logged too, since the job itself ended well."""
        try:
            report = self._read_report(job, uri)
        except Exception as exc:  # the caller waited for the job, which completed; the report is a bonus here
            _LOG.warning("training job %s: could not read its training report: %s", job.job_name, exc)
            return
        _LOG.info("training job %s:\n%s", job.job_name, report.get_summary())

    def _read_report(self, job: FineTuneJob, uri: str) -> TrainingReport:
        """The artifact's report, read once per artifact URI."""
        cached = self._reports.get(uri)
        if cached is None:
            cached = self._reports[uri] = read_training_report(self._s3, uri=uri, job=job)
        return cached
