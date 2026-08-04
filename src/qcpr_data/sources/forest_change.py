"""Forest-Change physical pair adapter."""

from .common import RegistrySourceAdapter

ADAPTER = RegistrySourceAdapter("Forest-Change", "pilot")


def iter_items(*args, **kwargs):
    return ADAPTER.pairs(*args, **kwargs)
