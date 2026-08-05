"""Schema-validated exact-core data helpers for the SigLIP-2 track."""

from .loader import ExactBatch, make_exact_batches, select_epoch_rows
from .manifest import (
    ALLOWED_DATASETS,
    group_rows_by_pair,
    load_exact_core_rows,
    load_exact_pair_rows,
    ordered_id_sha256,
    paired_caption_rows,
    read_jsonl,
)
from .transforms import SynchronizedResize, assert_pair_geometry

__all__ = [
    "ALLOWED_DATASETS",
    "ExactBatch",
    "SynchronizedResize",
    "assert_pair_geometry",
    "group_rows_by_pair",
    "load_exact_core_rows",
    "load_exact_pair_rows",
    "make_exact_batches",
    "ordered_id_sha256",
    "paired_caption_rows",
    "read_jsonl",
    "select_epoch_rows",
]
