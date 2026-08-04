"""SECOND-CC physical pair adapter."""

from .common import RegistrySourceAdapter

ADAPTER = RegistrySourceAdapter("second_cc", "existing")


def iter_items(*args, **kwargs):
    return ADAPTER.pairs(*args, **kwargs)
