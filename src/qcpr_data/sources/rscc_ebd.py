"""RSCC-EBD physical-only pair adapter."""

from .common import RegistrySourceAdapter

ADAPTER = RegistrySourceAdapter("RSCC-EBD", "stage2-pilot")


def iter_items(*args, **kwargs):
    return ADAPTER.pairs(*args, **kwargs)
