"""RSRCC adapter for audited query/parent registries."""

from .common import RegistrySourceAdapter

ADAPTER = RegistrySourceAdapter("RSRCC", "unacquired")


def iter_items(*args, **kwargs):
    return ADAPTER.pairs(*args, **kwargs)
