"""LEVIR-MCI physical pair adapter."""

from .common import RegistrySourceAdapter

ADAPTER = RegistrySourceAdapter("levir_mci", "existing")


def iter_items(*args, **kwargs):
    return ADAPTER.pairs(*args, **kwargs)
