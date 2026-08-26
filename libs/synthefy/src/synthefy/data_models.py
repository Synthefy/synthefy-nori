from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synthefy.nori_data_models import (
    LargeContextPolicy,
    LargeContextReport,
    MAX_LARGE_CONTEXT_SEED,
    MAX_LARGE_CONTEXT_THRESHOLD,
    MemoryPolicyInput,
    MemoryReport,
)


class NoriPredictRequest(BaseModel):
    """Request payload for a Synthefy Nori prediction.

    Mirrors the hosted inference contract exactly::

        {"X_train": [[...], ...], "y_train": [...], "X_test": [[...], ...],
         "task": "regression"}

    Parameters
    ----------
    X_train : List[List[float | None]]
        Labeled context rows. Shape ``(n_context, n_features)``.
    y_train : List[float]
        Target value for each context row. Length ``n_context``.
    X_test : List[List[float | None]]
        Query rows to predict. Shape ``(n_query, n_features)``.
    task : str, default "regression"
        The prediction task. Currently only ``"regression"`` is supported.
    memory_policy : str or dict, optional
        Serving-memory policy, at parity with the local package's
        ``NoriRegressor(memory_policy=...)``: a preset name (``"exact"``,
        ``"max_context"``, ``"off"``) or an object of policy fields. Omitted from
        the wire payload entirely when unset, so a request that does not use it is
        byte-for-byte what it always was.
    output_type : str or None, optional
        What to return from the predictive distribution (``"mean"``,
        ``"median"``, ``"quantiles"``, ``"full"``). ``None`` means
        "the server default", which is ``"mean"``; it is **omitted from the
        request body** so the default request is byte-for-byte what earlier
        client versions sent.
    quantiles : List[float] or None, optional
        Tau levels in ``(0, 1)`` for ``output_type="quantiles"``, in the caller's
        order. Omitted from the body when ``None``.
    large_context_policy : optional
        A policy-name string in hosted modes; local mode also accepts a callable.
        The client forwards it unchanged, and the installed policy registry is the
        source of truth for supported built-ins and parameters.
    large_context_threshold : int or None, optional
        Context row count strictly above which the policy engages. Valid only
        with ``large_context_policy``.
    large_context_seed : int or None, optional
        Deterministic selection/routing seed. Valid only with a policy.
    """

    # Coerce assignments just as construction does, so assigning a policy dict
    # cannot leave the field holding a value that contradicts its declared type.
    model_config = ConfigDict(validate_assignment=True)

    X_train: List[List[Optional[float]]]
    y_train: List[float]
    X_test: List[List[Optional[float]]]
    task: str = "regression"
    memory_policy: Optional[MemoryPolicyInput] = None
    output_type: Optional[str] = None
    quantiles: Optional[List[float]] = None
    large_context_policy: Optional[LargeContextPolicy] = None
    large_context_threshold: Optional[int] = Field(default=None, strict=True, ge=1, le=MAX_LARGE_CONTEXT_THRESHOLD)
    large_context_seed: Optional[int] = Field(default=None, strict=True, ge=0, le=MAX_LARGE_CONTEXT_SEED)

    @model_validator(mode="after")
    def _large_context_parameters_need_a_policy(self):
        if self.large_context_policy is None and (
            self.large_context_threshold is not None or self.large_context_seed is not None
        ):
            raise ValueError("large_context_threshold/large_context_seed require large_context_policy")
        return self

    def to_wire(self) -> Dict[str, Any]:
        """Serialize the request without pinning optional server defaults.

        In particular, a partial ``MemoryPolicy`` must carry only the fields the caller set.
        Sending the model's other defaults would make an older client override future server
        defaults. ``SynthefyNoriClient`` and serving contract tests both use this method so the
        wire representation has one implementation.
        """
        payload = self.model_dump(exclude={"memory_policy"}, exclude_none=True)
        if self.memory_policy is not None:
            payload["memory_policy"] = (
                self.memory_policy
                if isinstance(self.memory_policy, str)
                else self.memory_policy.model_dump(exclude_unset=True)
            )
        return payload


class NoriPredictResponse(BaseModel):
    """Response payload from a Synthefy Nori prediction.

    Parameters
    ----------
    task : str
        The task echoed back by the server (e.g. ``"regression"``).
    predictions : List[float | None]
        One point prediction per row of ``X_test``: the summary named by the
        request's ``output_type``, or the distribution mean when a
        distribution output (``"quantiles"``/``"full"``) was requested. A
        server-side non-finite result is represented as ``None`` on this wire
        model and converted back to ``NaN`` by
        :meth:`synthefy.nori_client.SynthefyNoriClient.predict`.
    model : str or None, optional
        The model identity that produced the response. SageMaker returns this so
        the client can verify that a named endpoint served the requested model and
        fail closed instead of returning valid-looking predictions from the wrong
        model specification.
    memory_report : dict, optional
        Present only when the request set ``memory_policy``: what the server actually did
        about it. See
        :attr:`synthefy.nori_client.SynthefyNoriClient.last_memory_report`.
    output_type : str or None, optional
        The output type the server actually honored. A deployment that predates
        distribution output omits this field, which is how the client detects
        that its ``output_type`` was silently ignored (see
        :meth:`synthefy.nori_client.SynthefyNoriClient._predict_remote`) instead
        of handing back means labeled as something else.
    quantiles : List[List[float | None]] or None, optional
        The predictive quantile function, **one row per query row**:
        ``(n_query, K)`` ascending values in original-``y`` units. For
        ``output_type="quantiles"`` ``K`` is the number of requested levels (in
        the requested order, not sorted); for ``"full"`` it is the checkpoint's
        whole quantile bank. Present only when a distribution output was
        requested. Entries are nullable because JSON has no ``NaN``: the server
        sends ``null`` for a non-finite value and the client maps it back to
        ``NaN``.
    taus : List[float] or None, optional
        The ``K`` quantile levels matching ``quantiles``' columns.
    large_context_report : LargeContextReport or None, optional
        Capability handshake and resolved execution report. Present only when
        the request selected a large-context policy.
    """

    task: str
    predictions: List[Optional[float]]
    model: Optional[str] = None
    memory_report: Optional[MemoryReport] = None
    output_type: Optional[str] = None
    quantiles: Optional[List[List[Optional[float]]]] = None
    taus: Optional[List[float]] = None
    large_context_report: Optional[LargeContextReport] = None
