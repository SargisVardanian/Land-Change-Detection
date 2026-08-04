"""Dubai-CC adapter; it remains access/licence gated until official assets exist."""

from .common import RegistrySourceAdapter

ADAPTER = RegistrySourceAdapter("DUBAI-CC", "unacquired")


def iter_items(*args, **kwargs):
    return ADAPTER.pairs(*args, **kwargs)
