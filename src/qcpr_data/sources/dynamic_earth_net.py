"""DynamicEarthNet sequence adapter."""

from .common import RegistrySourceAdapter

ADAPTER = RegistrySourceAdapter("DynamicEarthNet", "unacquired")


def iter_items(*args, **kwargs):
    return ADAPTER.pairs(*args, **kwargs)
