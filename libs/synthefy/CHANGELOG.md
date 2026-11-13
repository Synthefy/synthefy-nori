# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [9.0.0] - 2026-11-11

The fine-tuning hyperparameters say how much of the table each model reads, in
place of a step count and a time budget per model. A caller that passes `steps`
or `budget_s` is refused, so the release that carries this is a major version.
`mode="local"` with 9.0.0 needs a synthefy-nori release after 0.23.0:
synthefy-nori 0.23.0 requires `synthefy>=7,<9`.

### Changed (breaking)

- `FineTuneHyperparameters` drops `steps` and `budget_s`. A model takes the
  training steps that read each row of its share of the table
  `n_passes_per_row` times (unset by default: each row once, so the job reads
  the whole table once, unless `patience` stops the model sooner), and at least
  100; `n_steps_per_model` sets an exact number of steps and is refused
  together with `n_passes_per_row`.
  Replace `steps=N` with `n_steps_per_model=N`; remove `budget_s` and use
  `max_training_hours` for a time cap.
- `patience` counts validation scores: a model stops after `patience` scores
  in a row without a better one (25 by default), and a score comes every
  `validation_every_n_steps` steps.
  Replace a `patience` of N training steps with N divided by
  `validation_every_n_steps`, rounded up.
- The default `validation_every_n_steps` keeps the wait `patience` sets at
  100 training steps or fewer, so it no longer grows with the plan (a
  50,000-step plan waited 12,500 steps): a model is scored every hundredth of
  its steps, but at most every 100 / `patience` steps (4 at patience 25), which
  makes a long plan's upper-bound hours about 45% longer. A
  `validation_every_n_steps` you set is kept, and the wait is `patience` times
  it, however long. `TrainingPlan.patience_steps` gives the wait, which the
  summary prints. The 100 steps are provisional.
- `max_training_hours` (24.0 by default, at most 119.5) caps the whole job from
  its start, the read of the table included, and
  `FineTuneJobRequest.max_runtime_s` defaults to `max_training_hours` x 3,600 +
  1,800 s, and 432,000 s when `max_training_hours` is `None`. A `max_runtime_s`
  you set is sent as it is, and, unless the Algorithm's version trains each
  model for a fixed number of steps, `fine_tune()` and `get_training_plan()`
  refuse one below that sum; a request copied with new hyperparameters derives
  `max_runtime_s` from them.
  `max_training_hours` is refused when it is not finite or above 119.5. A job
  that `resume()` starts may change `max_training_hours`.
- Each model trains on at least 1,000 rows, so a table needs about 1,400 rows
  for `bag` 1 without a validation channel, and under `"shard"` about 1,150
  rows per model (1,050 with a validation channel); a smaller table trains
  fewer models.
- Under `"folds"` with `bag` above 1, each model trains on all but one fold,
  so a table needs about 1,270 rows with `bag` 8 (1,150 with a validation
  channel) and more with a smaller `bag`; a smaller table is refused.

### Added

- `base_model="nori-1-5"` fine-tunes Nori 1.5 next to `nori-6m` and `nori-100m`, and it is the default. The
  model it trains keeps no context cache, like its base. Its plan gives hours and cost on the `ml.g5` types
  (6.0 s a step, measured on `ml.g5.12xlarge`) and none on `ml.g6` or `ml.g6e`. `fine_tune()` and
  `get_training_plan()` raise `FineTuningError` for a `base_model` the Algorithm's version does not list,
  naming the bases it offers. An unset `base_model` takes the version's declared default, so a version
  published before `nori-1-5` still trains `nori-6m` for a caller who never chose a base.
- A job of tasks fine-tunes one model on several regression tasks:
  `FineTuneHyperparameters.task_column` names each row's task in one table
  (numbers or text, or `key=value` folder names), and `tasks`, a list of
  `TaskTable`, gives tables in different files their own target and feature
  columns. `time_column` makes each task of a table a time series, sorted by
  it, whose training steps predict the rows right after their context.
  `context_rows` and `query_rows` set how many rows a step takes, as a number
  or as `{rows: weight}`, for the job or per table; a time series needs
  `query_rows`. A job of tasks reads its tables into memory whole, refuses a
  task whose target never changes and a time repeated within a task, has no
  plan before it runs (`get_training_plan()` checks the tables, then raises
  `ValueError`; `fine_tune()` attaches no plan) and refuses `checkpoint_uri`.
  `target_column` is now optional: set it or `tasks`. `TrainingReport.tasks`
  lists each task's table, rows and scores (`TaskScore`). The settings are
  unset by default and never sent unset, so a request without them runs on
  every Algorithm version.
- `FineTuneHyperparameters.row_partition_strategy`: `"shard"` (the default)
  gives each model its own 1/`bag` of the training rows; `"folds"` trains each
  model on every training row outside its validation fold.
- `FineTuneHyperparameters.validation_every_n_steps`: the training steps
  between two validation scores of a model; by default the steps per model
  divided by 100, at least 1.
- `patience=None` lets each model take every step of its plan, and
  `max_training_hours=None` turns the cap off; both are sent to the job as
  `"none"`.
- `TrainingPlan`: what a job reads and what it takes (rows per model, steps,
  passes per row, `patience`, estimated hours and dollars, warnings), with
  `get_summary()`; the hours and dollars are a ceiling when `patience` can
  stop a model sooner. `TrainingReport.training_plan` holds the job's plan;
  `FineTuneJob.training_plan` holds a plan `fine_tune()` worked out before
  submitting, and is `None` otherwise.
- `TrainingReport` fields `stop_reason`, `n_passes_per_row_completed`,
  `seconds_per_step`, `fraction_of_rows_as_targets`, `unstarted_children`,
  `n_rows_unread`, `bag` and `row_partition_strategy`, and
  `ChildCurve.stop_reason`. A report from an earlier job still parses, with
  `stop_reason`, `n_passes_per_row_completed`, `fraction_of_rows_as_targets`
  and `n_rows_unread` null.
- `TrainingReport.get_summary()`, which `wait()` logs when a job completes.
- `SynthefyNoriFineTuner.resume(job, instance_type=None, max_training_hours=...)`
  starts a new job with `job`'s request on its checkpoint folder, which keeps
  the models the folder holds finished and continues the others from their
  last saved step; `instance_type` and `max_training_hours` may change, and
  the job's tags are not copied. It makes the request's training plan as
  `fine_tune()` does, logs it and returns it as the new job's
  `training_plan`: the plan of the whole request, the most the new job
  takes. It refuses a job still running, a job
  without `checkpoint_uri` or whose folder holds no checkpoint file, a job
  created with managed spot training, a VPC or a volume KMS key, and a
  completed job that left no model to train or whose training report cannot
  be read. It needs `s3:ListBucket` and `s3:GetObject` on the folder and
  `sagemaker:ListTrainingJobs`.
- `fine_tune(request, overwrite=True)` deletes the checkpoint files in the
  request's `checkpoint_uri` folder, then creates the job. It deletes nothing
  and refuses when the folder holds a file a fine-tuning job does not write,
  more than 1,000 files, or the files of a job still running; it needs
  `s3:DeleteObject` on the folder. A message names a permission only when AWS
  denied access.
- `SynthefyNoriFineTuner.get_training_plan(request)`: the job's plan for a
  request from the footers of its Parquet files, before submitting it, with
  rows per step estimated from the feature columns. `get_training_plan` refuses
  the same one-object channel `fine_tune()` refuses. `fine_tune()` logs that
  plan, returns it as `FineTuneJob.training_plan`, and warns when the plan's
  training hours leave the job less than the 30 minutes it needs besides
  training within SageMaker's 120-hour limit on a training job and the
  request has no `checkpoint_uri`.
- `fine_tune()` refuses a target or feature column the training table lacks,
  a folder channel with no Parquet file or with a file of another extension,
  and a file that cannot be read as Parquet, before creating the job.
- On an Algorithm version that trains each model for a fixed number of steps,
  `fine_tune()` refuses `row_partition_strategy`, `n_passes_per_row`,
  `max_training_hours`, `validation_every_n_steps` and a `patience=None` the
  request sets, by name, and otherwise trains each model for
  `n_steps_per_model` steps (100 by default) or 900 seconds, whichever ends
  first, and the plan counts the steps that fit; `patience` counts validation
  scores on that version too, 25 by default. A job on that version is sent a
  `max_runtime_s` the request sets as it is, and by default ceil(bag / N) x
  1,350 + 1,800 s on an instance type with N GPUs, at most 432,000 s. When the
  Algorithm cannot be described, the job gets only the settings every version of
  the Algorithm reads, and a request that sets `row_partition_strategy`,
  `n_passes_per_row`, `max_training_hours`, `validation_every_n_steps`,
  `n_steps_per_model` or `patience=None` is refused, naming each.
- The `aws` extra installs `pyarrow` 16.0.0 or later, which reads the Parquet
  footers for a training plan.
- A job given `checkpoint_uri` saves a model in training every 15 minutes or
  500 steps and when it stops; `resume()` continues that model from its last
  saved step. The folder holds at most about 0.4 GB
  per model for `nori-6m` (6.1 GB for `nori-100m`), and the execution role
  needs `s3:DeleteObject` on it.
  `ChildCurve.n_skipped_state_saves` counts the saves of a model that were due
  and did not happen, and `ChildCurve.last_saved_step` names its newest saved
  step. A job that cannot continue a saved state fails and leaves that model's
  files in the folder unchanged.

### Changed

- `fine_tune()` refuses a `checkpoint_uri` folder that holds any file before
  creating the job, naming the models and steps it holds and the job that
  wrote it: call `resume()` with that job, pass `overwrite=True`, or give
  another `checkpoint_uri`. A folder that holds only files a fine-tuning job
  does not write is refused as not a checkpoint folder.
- `fine_tune()` with `checkpoint_uri`, and `resume()`, refuse while another
  `InProgress` or `Stopping` training job of the account saves to the folder,
  with nothing deleted and no job created.
- `fine_tune()` with `checkpoint_uri` needs `s3:ListBucket` on the bucket,
  `s3:GetObject` on the folder and `sagemaker:ListTrainingJobs` for the
  identity that runs the client, and refuses without them: add
  `sagemaker:ListTrainingJobs` to a caller policy copied from an earlier
  README.
- A job that `resume()` starts keeps the earlier job's `model_name`, and a job
  on a folder another job wrote fails unless it continues that job.
- The `APITimeoutError` that `deploy()` raises when the endpoint is not
  `InService` within `timeout_s` says the endpoint is left in place, that
  `undeploy()` waits for it and then deletes it, and names the other endpoint
  instance types to deploy on when AWS is short of the chosen one.

### Fixed

- `SynthefyNoriFineTuner.undeploy()` deletes an endpoint that is still
  `Creating`, `Updating`, `SystemUpdating` or `RollingBack`: when SageMaker
  refuses the delete for that reason, it waits for the endpoint to leave that
  status and asks again, up to its new `timeout_s` argument in all (3,600 s by
  default). An endpoint still in that status after `timeout_s` raises
  `APITimeoutError`, which says to call `undeploy()` again later.
- `undeploy()` of an endpoint already `Deleting`, after an earlier call that
  timed out waiting for it or a delete made elsewhere, waits for the endpoint
  to be gone and then deletes its endpoint configs and model.

## [8.0.0] - 2026-10-28

The major version is for the Parquet-only fine-tuning channels: a 7.x caller
that passes a `.csv` channel is refused. `mode="local"` with 8.0.0 needs a
synthefy-nori release after 0.22.0: synthefy-nori 0.22.0 requires
`synthefy>=7,<8`.

### Changed (breaking)

- Fine-tuning channels are Parquet only. `FineTuneJobRequest` accepts a channel
  given as an S3 folder ending in `/` of Parquet files or as one `.parquet` or
  `.pq` object, and refuses a `.csv` object, or any other object, before any AWS
  call; the refusal of a `.csv` object says how to write the table as Parquet
  (`pandas.DataFrame.to_parquet(path, index=False)`). The job fails before it
  reads any row when a channel holds a CSV file or a file of another format.
  The channels go to SageMaker as `application/x-parquet`.

### Added

- `FineTuneJobRequest.checkpoint_uri`: an S3 folder ending in `/` where a
  fine-tuning job saves each child it finishes. After a stop or a failure,
  `fine_tune()` with the same request (`model_name` and `instance_type` may
  change) trains only the children not saved there, and the report marks the
  others with `ChildCurve.restored`. A folder resumes only on the Algorithm
  version that wrote it and with the same table and hyperparameters: a job
  refuses a folder written by another version, with other hyperparameters or
  for another table, naming what differs, and an edited table needs an empty
  folder. Run one job per folder at a time. An Algorithm version that does not
  save checkpoints ignores the folder and trains every child.

### Changed

- `SynthefyNoriFineTuner.fine_tune` refuses a channel given as one object when
  other keys start with its key, whatever the object's format, since SageMaker
  would read those too; the message says the channel "names one object".
- `FineTuneJobRequest` refuses a channel given as one object whose name starts
  with `_` or `.`: the job skips such files, like the markers Spark and Hadoop
  write next to a table, and would find no rows.
- `max_runtime_s` covers the bag on an instance type with N GPUs: by default
  ceil(1.5 x ceil(bag / N) x budget_s) + 1800, and at least
  ceil(bag / N) x budget_s.
- With `steps` of 1,000 or more, a child is scored every `steps` / 500 steps,
  rounded down (the report's `validation_interval`); `patience` is rounded up
  to a multiple of that, and a child's curve has one point per score.
- The descriptions of `steps`, `bag`, `budget_s` and `patience` in
  `FineTuneHyperparameters` say what each setting means to the job: `steps`
  sets how many rows a child reads, the `bag` children train one per GPU at a
  time, `budget_s` is the time cap of one child, and `patience` counts training
  steps without a better validation score.

### Added

- `TrainingReport` fields `validation_interval` (default `1`), `read_seconds`
  (`0.0`), `rows_visited` (`0`), `fraction_visited` (`0.0`), `steps_per_pass`
  (`0`), `workers` (`1`), `gpus` (`1`) and `failed_children` (empty), and
  `ChildCurve` fields `rows_visited` (`0`), `failed` (`False`) and `failure`
  (empty); a report written without one of these fields reads it as its
  default. `child_curves` holds a curve for every child that started or
  failed.
- Fine-tuning on `ml.g5.12xlarge` and `ml.g6e.12xlarge` (4 GPUs) and on
  `ml.g5.48xlarge` and `ml.g6e.48xlarge` (8 GPUs), one child per GPU at a time.
  `MULTI_GPU_INSTANCE_TYPES` maps each to its number of GPUs, and
  `LARGE_GPU_INSTANCE_TYPES`, the types `nori-100m` needs, holds the two
  `ml.g6e` ones as well.
- The README's fine-tuning section gives the rows a step reads on every
  training instance type, the hours and dollars of one pass over a large table
  on the types measured or estimated, and the host memory a job needs ("Large
  tables: what a job reads and what it costs").

## [7.4.0] - 2026-10-20

### Added

- **Nori 1.5** (`model="nori-1-5"`, gateway slug `synthefy/nori-1-5`), the ~21M Model v3
  release, on the hosted API (`mode="remote"`), on the SageMaker endpoint
  (`mode="sagemaker"`) and locally (`mode="local"`, which needs synthefy-nori 0.22.0 or
  later).
- Nori 1.5 builds no key/value cache over the context rows, so with it `memory_policy=`
  may set only `elements_budget` and `allow_subsample`, the two fields that act without a
  cache. A preset or any other field raises `ValueError` before a request is sent, in every
  mode; the hosted API refuses the same settings with the same message.

### Changed

- `NoriTSForecaster` refuses a covariate whose values a generated time feature
  would overwrite, such as a column named `year`, `running_index` or
  `hour_of_day_sin`, with a `ValueError` that names the column. Before, the
  generated feature silently replaced the caller's values. A covariate named
  `_is_train`, the forecaster's internal train/horizon marker, is refused the
  same way.
- When a `CalendarFeature` generator gives one calendar feature several cycle
  lengths in `seasonal_features` (for example `{"hour_of_day": [24, 12]}`),
  every period keeps its own sine/cosine pair: `hour_of_day_sin_0`,
  `hour_of_day_cos_0`, `hour_of_day_sin_1`, `hour_of_day_cos_1`. Before, each
  period overwrote the previous one's `hour_of_day_sin` and `hour_of_day_cos`,
  so only the last survived. A single period keeps the `<feature>_sin` / `<feature>_cos` names.

### Fixed

- `NoriTSForecaster.predict` orders every series by time before it builds the
  model context and the positional features, so rows passed out of order
  forecast the same as sorted ones.
- `NoriTSForecaster` infers a series' cadence from its complete history before
  `context_length` limits the rows the model sees.
- `MultimodalPreprocessor` accepts a custom text embedding with a single
  coordinate when `svd_dim` is set; that coordinate is kept as the feature.
- `TimeSeriesDataFrame.from_iterable_dataset` reads a single-pass iterable,
  such as a generator, once before validating it, so conversion no longer finds
  it exhausted and validation errors name the right record.
- `DataFramePreprocessor` treats a pandas `category` column as a declared
  categorical, so a high-cardinality `category` column keeps its most frequent
  levels instead of raising. Tuple column labels, including a `MultiIndex`,
  keep a one-dimensional `feature_names_in_` and survive pickling.

## [7.3.0] - 2026-10-15

### Added

- **`SynthefyNoriFineTuner`** fine-tunes a released Nori model (`nori-6m` or
  `nori-100m`) on your own CSV table through the Nori fine-tuning Algorithm on
  Amazon SageMaker and deploys the result. `fine_tune()` starts a training job from a `FineTuneJobRequest`
  (its `FineTuneHyperparameters` are the Algorithm's hyperparameters), `get_status()`, `wait()`,
  `get_logs()` and `get_training_curves()` follow it, `stop()` stops it,
  `get_training_report()` reads its `TrainingReport` without extracting the
  model, `deploy()` / `undeploy()` create and remove an asynchronous endpoint
  serving it, described by a `FineTunedEndpoint`, and `create_client()` builds the
  SageMaker-mode `SynthefyNoriClient` that predicts through that endpoint.
  Requests are checked before any AWS call. Install it with
  `pip install "synthefy[aws]"`.
- SageMaker mode accepts a fine-tuned model's name (`model="ft-acme-churn-v1"`:
  `ft-` then lowercase letters, digits and hyphens, 63 characters at most), sends it to the
  endpoint and verifies the answering model as it does for a released name. The
  constructor still refuses every other name, a mistyped `"nori-6n"` included; a
  fine-tuned name the endpoint does not serve is refused by the endpoint as
  `SageMakerInvocationError`. Multi-target and large-context requests are refused
  only for the released Thinking name (`nori-30m-thinking-medium` or its
  `synthefy/` slug); a fine-tuned name is left to the deployment.
- `SynthefyNoriClient.last_model_version`: the `model_version` the deployment
  reported on the most recent remote or SageMaker prediction, cleared before
  every call and left `None` by a call that raises, so a returned result can be
  tied to the model a `TrainingReport` describes.
- `FineTuningError`, raised when a fine-tuning job or the deployment of its
  result cannot give you a model. It carries `job_name`, `job_arn` (set when
  SageMaker accepted the job, so a failed first status read keeps its receipt),
  `status`, `failure_reason` and `created`, the resources a failed deployment
  had already made.
- The `aws` extra requires boto3 1.42.54 or later, the first release whose
  `InvokeEndpointAsync` takes `S3OutputPathExtension` and `Filename`, which every
  SageMaker prediction sends.
- `SageMakerInvocationError`, raised when a queued SageMaker inference writes a
  failure object instead of predictions. It carries the `invocation_id` and the
  `failure_location` of that object.

### Changed

- SageMaker mode now requires a customer-owned `s3_staging_uri`, writes a compact
  tensor request there, and uses `InvokeEndpointAsync` instead of the retired
  response-stream path. `predict()` remains blocking while it polls the output
  and failure objects, supports requests up to 1 GiB, and cleans up only its own
  invocation objects.
- One SageMaker endpoint now serves all released Nori models, including 100M,
  and the client verifies the returned model identity.

### Fixed

- `undeploy()` deletes a `FineTunedEndpoint` in the endpoint's own region, the way
  `create_client()` predicts through it, so a handle from another region deletes
  that region's endpoint and not same-named resources in the fine-tuner's region.
- Generated training job and endpoint config names end with six random characters
  after the UTC timestamp (`nori-<model_name[:35]>-<timestamp>-<token>` and
  `<endpoint_name[:40]>-<timestamp>-<token>`), so two launches in the same second,
  or two long names cut to the same characters, get different names. `undeploy()`
  still finds the endpoint configs 7.2.0 named without the token.
- The README's several-models example resets `max_runtime_s` when it copies a
  request with other hyperparameters, so the copy derives its runtime from them.

## [7.2.0] - 2026-10-13

### Added

- **`model_version` on `NoriPredictResponse`**, the `sha256:<hex>` digest of the
  released specification that answered: the served model's checkpoint sha256,
  the name of its inference config, its Thinking budget, and the capabilities
  that budget implies. Equal values mean the same released model specification
  served the two responses, whichever host ran them, so a result can be pinned
  to a model rather than to a name that may later be re-pointed. It tracks that
  record alone, not the serving build. A deployment that predates the field
  leaves the attribute `None`.

## [7.1.3] - 2026-09-14

### Fixed

- Clarified that the `exact` memory preset disables cache quantization rather
  than guaranteeing identical predictions across chunk sizes or execution paths.
  Prediction behavior and policy defaults are unchanged.

## [7.1.2] - 2026-09-02

### Added

- Added the opt-in `target_rank[cap=N]` large-context policy to local, Baseten,
  and SageMaker requests. Policy lists can use bounded random or tail holdouts
  to select a per-table winner, and capability reports verify the honored
  policy and holdout strategy.

## [7.1.1] - 2026-08-29

### Fixed

- Preserved the one-row local `NoriRegressor.predict_point(...)` shape contract when the installed `synthefy-nori` package returns a single prediction. Local clients now receive a one-element prediction vector instead of a scalar for batch size 1, matching multi-row calls.

## [7.1.0] - 2026-08-27

### Added

- Large-context policy lists now work in local, Baseten, and SageMaker modes.
  `large_context_holdout="random"` scores IID rows, while `"tail"` keeps future
  chronological rows out of validation contexts. The capability report echoes
  the honored strategy. Lists are bounded to eight installed built-ins.
- Added one-call multi-target regression to `SynthefyNoriClient.predict` in
  local, Baseten remote, and SageMaker modes. A matrix-valued `y_train` returns
  joint means or bounded joint samples, defaults to the copula strategy, and
  accepts the same grouped `MultiTargetPredictionPolicy` as `NoriRegressor`.
  Hosted responses echo the honored strategy so older deployments fail closed.

### Fixed

- `MemoryReport.est_cache_gb`, `resident_gb` and `query_chunk` carry the library's
  bounds again (`>= 0`, `>= 0`, `> 0`). The mirror had dropped them.
- In local mode, a matrix-valued `y_train` (or `multi_target_prediction_strategy=` /
  `multi_target_prediction_policy=`) on a `synthefy-nori` that predates multi-target
  regression now raises `ImportError` with an upgrade hint before the library is
  constructed, instead of a shape error or `TypeError` deep inside it. Every keyword
  forwarded to `NoriRegressor(...)` is checked against its signature the same way.


## [7.0.4] - 2026-08-25

### Added

- Added large-context selection to `SynthefyNoriClient.predict` in local,
  Baseten remote, and SageMaker modes. Hosted modes accept a string naming every
  installed built-in policy, including `boost`, `safeboost`, and parameter
  strings; custom policies remain local-only. `last_large_context_report`
  echoes the resolved policy and proves how many internal Nori calls ran.
  Hosted policy execution stops before a 65th internal call; direct local use
  remains unlimited by default. Hosted calls retain no customer context between
  requests; Snowflake SPCS rejects the unsupported fifth-options shape explicitly.

### Removed (breaking)

- Removed the exported `HOSTED_LARGE_CONTEXT_POLICIES` snapshot. It was not
  used for validation and could not stay synchronized with the installed
  server registry. Code importing that constant must upgrade alongside this
  client change and pass policy specs directly to `predict` instead.

## [7.0.3] - 2026-08-20

### Added

- Every `discretize=` strategy now works against a hosted deployment, not just
  `"snap-mean"`. The request carries `discretize` and `categorical_levels` to
  the server, which holds the predictive distribution the other strategies need,
  and the response echoes the strategy that ran so a deployment too old to know
  the field is distinguishable from one that honored it. The call is unchanged:
  same arguments, same response shape, and a request that does not discretize
  sends the body it always sent.

- Added bounded large-context selection to `SynthefyNoriClient.predict` in
  local, Baseten remote, and SageMaker modes. The shared contract accepts
  `random`, `cluster_route`, and `cluster_route_g4` plus a bounded threshold
  and seed; `last_large_context_report` proves what ran and how many internal
  Nori calls it made. Hosted calls are one-shot and retain no customer context
  between requests; Snowflake SPCS rejects the unsupported fifth-options shape
  explicitly.

## [7.0.2] - 2026-08-17

### Added

- Added the `nori-100m` size to `NORI_VARIANTS`, so `model="nori-100m"` (and the
  raw gateway slug `"synthefy/nori-100m"`) select the ~98.3M checkpoint in both
  hosted and local modes. The weights are public at
  [`Synthefy/Nori-100M`](https://huggingface.co/Synthefy/Nori-100M).

## [7.0.1] - 2026-08-15

### Added

- Added `categorical_columns=` to `SynthefyNoriClient.predict`: `"auto"`
  resolves remaining non-numeric DataFrame columns, a sequence declares an
  exact categorical schema, and `None` disables inference. The same fitted
  schema contract is shared with `NoriRegressor` and the one-shot helpers.
- Added `future_df=` to `NoriTSForecaster.predict_df` for explicit forecast
  horizons with numeric known-future covariates, such as planned promotions,
  holidays, weather, or scheduled prices. The horizon must contain exactly the
  same series and covariates as history, and future target values are rejected
  to prevent leakage.
- Added `target_column=` to the same DataFrame workflow so callers can preserve
  domain names such as `sales` instead of renaming the target to `target`.
  It accepts one target column per call and clearly rejects multiple targets.
  Forecasts continue to execute through the configured
  `SynthefyNoriClient`, including remote, SageMaker, and local modes.
- Added `reuse_context_cache` to the public `MemoryPolicy` mirror, keeping the
  lightweight client aligned with the `synthefy-nori 0.17.3` hosted and local
  contract and allowing retained context reuse to be disabled explicitly.

### Changed

- Ordinal categorical query values that were absent from context now use the
  bounded `K` `other` code instead of `-1`; missing values remain null/NaN.
  Automatically inferred high-cardinality and temporal columns now raise with
  an explicit categorical/text/numeric choice instead of being dropped with a
  warning. Explicit high-cardinality categoricals retain top-K levels plus
  `other`.
- `predict_df` now accepts exactly one of `prediction_length=` or `future_df=`.
  The existing generated-horizon behavior remains available through
  `prediction_length=`.


## [7.0.0] - 2026-08-12

### Removed

- Removed the retired `SynthefyAPIClient` and `SynthefyAsyncAPIClient`, the
  `ForecastV2Request` and `ForecastV2Response` types, their top-level exports,
  and the obsolete `/v2/forecast` examples and tests. This product has no
  customer-compatibility migration requirement; Nori forecasting uses
  `synthefy.nori_ts.NoriTSForecaster`.

### Changed

- Removed execution `mode="auto"` from `SynthefyNoriClient`. Callers must select
  `"remote"`, `"sagemaker"`, or `"local"` explicitly; installed packages and
  ambient credentials never change request routing.
- Local execution is installed with `synthefy-nori` rather than the retired
  `synthefy[local]` extra; runtime error messages and README commands now name
  the consolidated package relationship.


## [6.3.0]

### Added

- Added `SynthefyNoriClient(mode="sagemaker", model=...,
  endpoint_name=..., region_name=...)` for Amazon SageMaker real-time endpoints.
  It sends the same Nori request/response contract through boto3
  `InvokeEndpointWithResponseStream`, signed through boto3's standard AWS credential
  chain; the public API accepts no raw AWS keys.
- Added the optional `synthefy[aws]` dependency extra.
- Added stubbed SageMaker Runtime tests for named-endpoint invocation, credential-
  chain construction, transport argument guards, and preservation of a container's
  original status/message from SageMaker `ModelError`.

### Changed

- Hosted response capability checks are now shared by the Baseten and SageMaker
  transports, so distribution output and `memory_policy` cannot silently degrade on
  either backend.
- `model` is required and non-null on every Nori transport. SageMaker accepts the
  three released specifications: `nori-6m`, `nori-30m`, and
  `nori-30m-thinking-medium`.
- All SageMaker variants use response streaming with heartbeat chunks, allowing
  large 30M requests to run beyond regular invocation's 60-second limit while the
  synchronous client still returns one final typed result.
- SageMaker requests fail locally before invocation when their encoded body exceeds
  AWS Marketplace's 25,000,000-byte endpoint limit; oversized tables require
  the planned explicit S3-backed asynchronous API rather than semantic-changing splits.

## [6.2.2]

### Changed

- `SynthefyNoriClient.predict(text_columns=...)` now runs named sentence encoders
  on CUDA/ROCm when available, then Apple MPS, with CPU as the fallback. Pass the
  new `text_device=` argument to override automatic selection.

### Fixed

- Kept the hosted Nori wire contract aligned with the server: missing feature cells are sent
  as JSON `null` and nullable predictions are converted back to `NaN`.
- Moved `NoriPredictRequest` and `NoriPredictResponse` to `synthefy.data_models`, while
  preserving their package-level and `synthefy.nori_client` import paths.
- Centralized request serialization in `NoriPredictRequest.to_wire()` so the client and
  serving contract tests preserve the same optional-field and partial-policy semantics.

## [6.2.1]

### Changed

- Raised the `local` extra floor to `synthefy-nori>=0.13.1`, ensuring
  `pip install "synthefy[local]"` includes the release where recoverable SVD
  fallbacks emit `SvdFallbackWarning` instead of silently changing the pipeline.
  The client preserves the warning category/message and strict-mode exception.

## [6.2.0]

### Added

- **Prediction intervals on `SynthefyNoriClient.predict`** via `output_type=` and
  `quantiles=` — shared selectors use the same meanings as
  `synthefy-nori`'s `NoriRegressor.predict`. Nori's forward pass already produces
  a full predictive distribution, so intervals cost nothing extra; previously the
  client could only return its mean.
  - `output_type="mean"` (default, unchanged) and `"median"` — one value per
    query row.
  - `output_type="quantiles"` with `quantiles=[0.1, 0.5, 0.9]` — returns
    `(n_levels, n_query)`, level-major, so
    `lo, mid, hi = client.predict(..., output_type="quantiles", quantiles=[...])`
    unpacks directly (matching `NoriRegressor`). Levels come back in the order
    you passed them.
  - `output_type="full"` — the whole quantile bank as
    `{"quantiles": (n_query, K), "taus": (K,), "mean": (n_query,)}`, for CRPS /
    interval scoring and calibration.
  - `as_pandas=True` returns a `DataFrame` for the distribution output types: one
    row per query row (indexed by `X_test`), one column per level, named
    `"<target>[<level>]"` — the same convention the forecasting client uses.
- **Local mode routes through `NoriRegressor`** for any non-default
  `output_type`. The functional `synthefy_nori.predict` cannot express it (it
  forwards `**kwargs` to the constructor), so local intervals were previously
  unreachable. `output_type="mean"` still goes through the functional path, so
  the default local behavior is unchanged.
- **Remote mode** sends `output_type`/`quantiles` and reads the quantile block
  back. The hosted endpoint echoes the `output_type` it honored and the client
  raises `ValueError` on a mismatch: a deployment that predates distribution
  output answers with the distribution mean, which is indistinguishable from a
  genuine `"median"` result, so the handshake turns a silently wrong answer into
  a clear error naming the fix. Server-side support: `synthefy-nori-internal`
  `baseten/` (see the PR that pairs with this one).

### Notes

- An ordinary `predict(...)` call is byte-for-byte unchanged: the new fields are
  omitted from the request body unless explicitly requested, so nothing changes
  for existing deployments.
- `output_type`/`quantiles=` cannot be combined with
  `discretize=`/`categorical_levels=` (discrete labels vs. a distribution
  summary are different answers) — raises `ValueError`, mirroring
  `NoriRegressor`.

## [6.1.0]

### Added

- **`memory_policy=` on `predict()` — the serving-memory policy, at parity with the local package.**
  Either a preset name (`"exact"`, `"max_context"`, `"off"`) or an object of fields, e.g.
  `{"cache_dtype": "int8"}`. Nori does in-context regression, so your table is *input*: one
  prediction keeps a per-layer key/value cache over every context row, and that cache — not
  the ~6M-parameter model — is what exhausts GPU memory on a big table. This decides what to
  do about it. Omit it for defaults that suit almost every request.

  Works in **both** modes. Remote, the policy is validated server-side and an incoherent one
  is rejected before any inference is paid for. Local, it needs `synthefy-nori >= 0.13.0` and
  raises `ImportError` with an upgrade hint on older builds.

- **`SynthefyNoriClient.last_memory_report`** — what the server actually did about the policy
  on the most recent `predict`: which fallback rung ran, the estimated and resident cache
  sizes, the query chunk, any dropped context rows, plus any fields the server clamped and
  coherence notes about the policy you sent. Worth reading, because the rung is decided by the
  replica's free VRAM rather than by your request, so it is not knowable client-side.

  Remote mode only. In local mode the policy is honoured but no report exists: the client goes
  through `synthefy_nori.predict`, which builds an estimator internally and discards it, and
  the report lives on that estimator. Use `NoriRegressor` and read `memory_report_` directly if
  you need it locally.

- **`MemoryPolicy` and `MemoryReport` pydantic models** (`synthefy.nori_data_models`), exported
  from `synthefy`. `NoriPredictRequest.memory_policy` is typed `str | MemoryPolicy`, so a plain
  dict is **validated before any request is sent** — an unknown field, a bad type or an
  out-of-range value is caught locally instead of costing a round trip. Which *combinations* are
  incoherent stays server-side, deliberately: duplicating that behaviour would drift in a way a
  schema comparison cannot detect.

  Only the fields you actually set go on the wire (`exclude_unset`), so the client never pins the
  server's defaults — a later change to a default reaches existing clients rather than being
  silently overridden by them.

  The models are a copy of the library's, policed by 20 parity tests today and by the cross-repo
  sync check specced in SynthefyPFN#119.

- **`memory_report` on `NoriPredictResponse`**, mirroring the hosted contract.

- **`memory_policy=` also accepts a `MemoryPolicy` instance** — the pydantic model from `synthefy-nori`,
  which *is* the schema. This client does not redeclare the policy's fields, so there is nothing
  here to drift from the library: a preset name or dict is validated server-side (where the rules
  live), and anyone with `synthefy-nori` installed can pass the real model and get validation at
  construction plus IDE completion. Duck-typed on `model_dump`, so the client keeps no dependency
  on the model package. The fields `resolve()` decides (`rung`, `est_cache_gb`, …) are stripped
  before sending; a policy that has already been resolved keeps its `rung` and is rejected
  server-side, which is correct and not re-implemented here.

### Changed

- A `predict()` call that sets `memory_policy=` and gets **no** `memory_report` back now raises
  `ValueError`. A deployment predating the field ignores it and returns default-memory
  predictions that are numerically valid, so nothing in `predictions` reveals the policy was
  dropped — the server echoes the report precisely so this is detectable, and believing a
  policy took effect when it did not is worse than an error.

- A request that does **not** set `memory_policy=` is byte-for-byte what it was before: the field is
  omitted from the payload entirely rather than sent as `null`.

## [6.0.0]

### Removed (breaking)

- **`DEDICATED_BASE_URL` and `DEDICATED_ENDPOINT` are gone.** `from synthefy.nori_client import
  DEDICATED_BASE_URL, DEDICATED_ENDPOINT` now raises `ImportError`. Nori is addressed by gateway
  slug: `SynthefyNoriClient(api_key=..., model="nori-6m")` sends `{"model": "synthefy/nori-6m"}`
  to `https://inference.baseten.co/predict`, and the gateway resolves the slug to a deployment.

  A hardcoded `model-<id>.api.baseten.co` host cannot stay correct. Each Nori variant is its own
  Baseten model with its own id, so one constant can address at most one of them, and an id does
  not survive a model being deleted and re-created. The gateway slug is stable across both.

  The gateway is also the only path Synthefy meters, rate-limits and grants per key, so it is the
  one supported way to reach hosted Nori.

### Migration

```python
# before
from synthefy.nori_client import DEDICATED_BASE_URL, DEDICATED_ENDPOINT
client = SynthefyNoriClient(
    api_key=key, base_url=DEDICATED_BASE_URL, endpoint=DEDICATED_ENDPOINT,
    model=None, auth_scheme="Api-Key",
)

# after
client = SynthefyNoriClient(api_key=key, model="nori-6m")
```

`base_url`, `endpoint`, `auth_scheme` and `model=None` all still exist as generic overrides for
pointing the client at a host of your own. Only the two Synthefy-specific constants are removed.

## [5.0.0]

### Changed (breaking)

- **`model=` is now REQUIRED** on `SynthefyNoriClient` — there is no default. Every request
  names a size: `model="nori-6m"` (~6M base) or `model="nori-30m"` (~29.2M). Omitting `model`
  raises `ValueError`. (`model=None` still targets a dedicated deployment endpoint.) This keeps a
  model identifier from ever silently changing which model it serves.
- **Removed the bare `nori` / `synthefy/nori` selectors.** Only size-explicit names/slugs resolve:
  `"nori-6m"` / `"synthefy/nori-6m"` and `"nori-30m"` / `"synthefy/nori-30m"` (plus the hosted-only
  `nori-30m-thinking*`). The `GATEWAY_MODEL` constant is removed.

### Migration

- Add an explicit size: `SynthefyNoriClient(api_key=..., model="nori-30m")` (was the ~6M default;
  pass `model="nori-6m"` to keep the smaller base).

## [4.3.0]

### Changed

- **Default categorical encoding is now ordinal** (was one-hot).
  `SynthefyNoriClient.predict` maps each non-numeric DataFrame column to a
  single column of integer codes in sorted-category order — the same
  convention as the Nori model's server-side `OrdinalEncoder` path (unseen
  test value → `-1`, missing → NaN, imputed server-side). A 35-dataset
  benchmark across three model sizes found ordinal at least as accurate as
  one-hot on average, with one-hot substantially worse on wide,
  categorical-heavy tables (it never widens the feature matrix).
  Pass `categorical_encoding="onehot"` to restore the previous behavior.

### Added

- `categorical_encoding` parameter on `predict` (`"ordinal"` (default) or
  `"onehot"`).

## [4.2.2]

### Fixed

- `SynthefyNoriClient` remote retries no longer surface a stale earlier-attempt
  exception. When an earlier attempt raised a transient error (e.g. a connection
  error or timeout) but the final attempt returned a retryable response (e.g. a
  5xx), the client raised the stale `APIConnectionError`/`APITimeoutError`
  instead of the true final error. `_post_with_retries` now resets its per-attempt
  state each iteration, so the final attempt's outcome is what's raised (e.g. a
  5xx maps to `InternalServerError`).

## [4.2.1]

### Added

- `SynthefyNoriClient.predict` now **one-hot encodes non-numeric columns** when
  both `X_train` and `X_test` are DataFrames, instead of raising. The encoding is
  fit on `X_train` and applied to `X_test` (a category seen only in `X_test`
  becomes an all-zeros indicator group), producing a fully numeric, model-ready
  matrix client-side — no server change and no reliance on server-side category
  detection. Numeric columns (including `bool`) pass through unchanged. A missing
  value (NaN) in a categorical column gets its own one-hot indicator
  (`dummy_na`), kept only when missingness actually occurs.
- New `max_categorical_cardinality` argument to `predict` (default `100`):
  non-numeric columns with more than this many distinct training values — plus
  datetime columns and columns with no non-missing values — are dropped with a
  `UserWarning` rather than exploding the feature matrix.
- `category` columns whose categories are numeric are kept as a numeric
  magnitude (not one-hot exploded), NaN-safe for integer categories.

### Changed

- A non-numeric column in a DataFrame `X_train`/`X_test` pair no longer raises;
  it is one-hot encoded (see above). Passing a non-numeric column with a
  non-DataFrame `X_test` still raises, since one-hot alignment needs column
  names on both sides. A column that is numeric in one of `X_train`/`X_test` but
  not the other raises a clear type-mismatch `ValueError`; duplicate column
  names, name/value one-hot collisions, and `timedelta` columns also raise with
  actionable messages (convert timedeltas to a number or string first).

## [4.2.0]

### Added

- `SynthefyNoriClient.predict` now accepts **pandas** inputs in addition to
  Python lists and numpy arrays: a `DataFrame` for `X_train`/`X_test`, and a
  `Series` or single-column `DataFrame` for `y_train`. All feature columns must
  be numeric — non-numeric (categorical/text/datetime) columns raise a clear
  `ValueError` directing the caller to encode them first. Missing values (NaN)
  are allowed and imputed server-side; no need to fill them in beforehand.
- When both `X_train` and `X_test` are DataFrames, `X_test` is aligned to
  `X_train`'s columns **by name**, so column order no longer has to match.
  Mismatched column *sets* raise `ValueError`.
- `predict(..., as_pandas=True)` returns a pandas `Series` instead of the default
  `list[float]`: one value per `X_test` row, named after `y_train` (its `Series`
  name or single-column `DataFrame` label, else `"prediction"`) and indexed by
  `X_test`'s index when it is a pandas object. Mirrors AutoGluon's `as_pandas`;
  defaults to `False` so existing callers are unaffected.

## [4.1.3]

### Changed

- Bumped the `local` extra's floor to `synthefy-nori>=0.9.0` (was `>=0.8.0`) so
  `pip install "synthefy[local]"` pulls in the latest local-inference package.
  `synthefy-nori>=0.9.0` still supports Python >=3.9, matching the base
  package's floor, so no environment marker is needed. No code or public API
  changes.

## [4.1.2]

### Changed

- Bumped the `local` extra's floor to `synthefy-nori>=0.8.0` (was `>=0.6.0`) so
  `pip install "synthefy[local]"` pulls in the latest local-inference package.
  `synthefy-nori>=0.8.0` still supports Python >=3.9, matching the base
  package's floor, so no environment marker is needed. No code or public API
  changes.

## [4.1.1]

### Changed

- Bumped the `local` extra's floor to `synthefy-nori>=0.6.0` (was `>=0.5.0`) so
  `pip install "synthefy[local]"` pulls in the latest local-inference package.
  `synthefy-nori>=0.6.0` still supports Python >=3.9, matching the base
  package's floor, so no environment marker is needed. No code or public API
  changes.

## [4.1.0]

### Fixed

- **Remote gateway authentication.** Requests to the default Baseten inference
  gateway (`https://inference.baseten.co/predict`) now send
  `Authorization: Bearer <key>` instead of `Authorization: Api-Key <key>`. The
  gateway accepts only the `Bearer` scheme, so every default-configured remote
  `predict(...)` call previously failed with HTTP `403` ("please check the
  api-key you provided") even when the key was valid. Dedicated deployments
  continue to use `Api-Key` (see below).

### Added

- New `auth_scheme` constructor argument on `SynthefyNoriClient`
  (`{"Bearer", "Api-Key"}`, default `"Bearer"`). The default fixes gateway
  auth out of the box; pass `auth_scheme="Api-Key"` when targeting a dedicated
  deployment. Invalid values raise `ValueError`.

## [4.0.1]

### Changed

- Bumped the `local` extra's floor to `synthefy-nori>=0.5.0` (was `>=0.1.0`) so
  `pip install "synthefy[local]"` pulls in the latest local-inference package.
  `synthefy-nori>=0.5.0` still supports Python >=3.9, matching the base
  package's floor, so no environment marker is needed. No code or public API
  changes.

## [4.0.0]

The tabular in-context regression product is now **Synthefy Nori**. This is a
breaking release: the client class, models, module, and optional local-inference
package were all renamed from `tabular` to `nori`. The forecasting client
(`SynthefyAPIClient` / `SynthefyAsyncAPIClient`) is unchanged, and the
`predict(...)` signature is identical — only names changed.

### Changed (BREAKING)

- Renamed `SynthefyTabularClient` → `SynthefyNoriClient`. There is no
  backward-compatible alias; the old name no longer imports.
- Renamed the request/response models `TabularPredictRequest` →
  `NoriPredictRequest` and `TabularPredictResponse` → `NoriPredictResponse`.
- Renamed the module `synthefy.tabular_client` → `synthefy.nori_client`. Imports
  such as `from synthefy.nori_client import DEDICATED_BASE_URL, DEDICATED_ENDPOINT`
  must be updated.
- The `local` extra now installs `synthefy-nori>=0.1.0` (was
  `synthefy-tabular>=0.2.3`). Local inference now imports from the `synthefy_nori`
  package. `pip install "synthefy[local]"` is unchanged.
- The default hosted gateway model identifier is now `synthefy/nori` (was
  `synthefy/synthefy-tabular`). The dedicated deployment `base_url`/`endpoint` are
  unchanged.

## [3.1.2]

### Changed

- Bumped the `local` extra's floor to `synthefy-tabular>=0.2.3` (was `>=0.2.2`)
  so `pip install "synthefy[local]"` pulls in the latest local-inference
  package. Still supports Python >=3.9, so no environment marker is needed.

## [3.1.1]

### Changed

- Documentation only (PyPI long description): the README now reflects the tabular
  client. The intro, feature list, and installation instructions cover
  `SynthefyTabularClient` (hosted and local modes) alongside forecasting. No code
  or API changes — released solely to refresh the immutable PyPI project page.

## [3.1.0]

### Added

- `SynthefyTabularClient`: a standalone, synchronous client for Synthefy Tabular
  in-context regression. Supply labeled context rows (`X_train`, `y_train`) and
  query rows (`X_test`) and receive one prediction per query row in a single
  forward pass — no training step. Accepts Python lists or numpy arrays,
  validates shapes, and is exported from the top-level `synthefy` package.
  - A single `mode` argument selects how predictions run: `"remote"` (default,
    hosted Baseten endpoint), `"local"` (in-process, no network, no API key), or
    `"auto"` (local if the optional package is installed, else remote).
  - Remote mode authenticates with a Baseten API key sent as
    `Authorization: Api-Key <key>`, taken from the `api_key` argument or the
    `BASETEN_API_KEY` environment variable, and defaults to the Baseten inference
    gateway (`https://inference.baseten.co/predict`, model
    `synthefy/synthefy-tabular`). To target a dedicated deployment, pass
    `base_url`/`endpoint` and `model=None`.
  - Reuses the package's existing error types: HTTP 400 maps to `BadRequestError`
    (carrying the server's `error` string) and 401 to `AuthenticationError`, with
    the same retry/backoff behavior as the forecasting client.
  - Local mode runs via the optional `synthefy-tabular` package, exposed through
    the new `local` extra: `pip install "synthefy[local]"` (supports Python >=3.9
    via `synthefy-tabular>=0.2.2`, matching the base package's floor).
- `TabularPredictRequest` and `TabularPredictResponse` pydantic models, exported
  from the top-level `synthefy` package.

## [3.0.0]

- Baseline release of the Synthefy forecasting client (`SynthefyAPIClient`,
  `SynthefyAsyncAPIClient`).
