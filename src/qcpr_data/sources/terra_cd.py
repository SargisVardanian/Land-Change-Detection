"""TERRA-CD pair adapter; label-derived text remains evaluation-only."""

from .common import RegistrySourceAdapter

ADAPTER = RegistrySourceAdapter("TERRA-CD", "unacquired")


def iter_items(*args, **kwargs):
    return ADAPTER.pairs(*args, **kwargs)
