"""Typed contracts used by all Dataset-v2 adapters and manifests."""

from .schemas import DenseEvaluationSidecar, FrameRecord, PhysicalItem, QueryRecord
from .validation import ValidationError, assert_mask_free, validate_physical_item, validate_query_record

__all__ = [
    "DenseEvaluationSidecar",
    "FrameRecord",
    "PhysicalItem",
    "QueryRecord",
    "ValidationError",
    "assert_mask_free",
    "validate_physical_item",
    "validate_query_record",
]
