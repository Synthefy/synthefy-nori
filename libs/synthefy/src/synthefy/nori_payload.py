"""Compact tensor-file codec shared with the host-neutral Nori application."""

from __future__ import annotations

import json
import struct
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from synthefy.data_models import NoriPredictRequest

CONTENT_TYPE = "application/vnd.synthefy.nori.tensorfile"
FORMAT_VERSION = "synthefy-nori-request-v1"
MAX_PAYLOAD_BYTES = 1024**3

_MAGIC = b"NORI_REQ"
_PREFIX = struct.Struct("<8sI")
_HEADER_BYTES = 64 * 1024
_TENSOR_DTYPES = {
    "X_train": np.dtype("<f4"),
    "y_train": np.dtype("<f8"),
    "X_test": np.dtype("<f4"),
}
_METADATA_FIELDS = tuple(name for name in NoriPredictRequest.model_fields if name not in _TENSOR_DTYPES)


def _metadata_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_unset=True)
    return value


def save_request(path: str | Path, request: Any, *, model: str) -> None:
    """Write an array-backed request without nested lists or payload-sized byte copies."""
    arrays = {
        field: np.ascontiguousarray(getattr(request, field), dtype=dtype) for field, dtype in _TENSOR_DTYPES.items()
    }
    offset = _HEADER_BYTES
    tensors = {}
    for field, array in arrays.items():
        tensors[field] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "offset": offset,
            "nbytes": array.nbytes,
        }
        offset += array.nbytes
    if offset > MAX_PAYLOAD_BYTES:
        raise ValueError(
            "The serialized SageMaker request exceeds the 1 GB asynchronous inference "
            "payload limit. Reduce the context/query rows."
        )

    metadata = {"model": model}
    for field in _METADATA_FIELDS:
        value = getattr(request, field, None)
        if value is not None:
            metadata[field] = _metadata_value(value)
    header = json.dumps(
        {"format": FORMAT_VERSION, "metadata": metadata, "tensors": tensors},
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(header) > _HEADER_BYTES - _PREFIX.size:
        raise ValueError("Nori request metadata exceeds the tensor-file header limit.")

    with Path(path).open("wb") as destination:
        destination.write(_PREFIX.pack(_MAGIC, len(header)))
        destination.write(header)
        destination.write(b"\0" * (_HEADER_BYTES - _PREFIX.size - len(header)))
        for array in arrays.values():
            array.tofile(destination)
