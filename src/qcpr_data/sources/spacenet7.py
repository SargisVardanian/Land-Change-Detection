"""SpaceNet-7 sequence adapter."""

from .common import RegistrySourceAdapter

ADAPTER = RegistrySourceAdapter("SpaceNet-7", "unacquired")


def iter_items(*args, **kwargs):
    return ADAPTER.pairs(*args, **kwargs)
