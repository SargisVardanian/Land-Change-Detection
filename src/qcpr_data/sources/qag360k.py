"""QAG-360K query adapter; parent physical identity is required."""

from .common import RegistrySourceAdapter

ADAPTER = RegistrySourceAdapter("QAG-360K", "unacquired")


def iter_items(*args, **kwargs):
    return ADAPTER.pairs(*args, **kwargs)
