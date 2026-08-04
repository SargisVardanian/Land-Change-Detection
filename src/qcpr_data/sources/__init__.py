"""Source adapters: physical normalization only, never scientific policy."""

from .common import RegistrySourceAdapter, iter_jsonl, normalize_pair_row, normalize_sequence_row

__all__ = ["RegistrySourceAdapter", "iter_jsonl", "normalize_pair_row", "normalize_sequence_row"]
