from __future__ import annotations

from pathlib import Path

from .backends.base import RetrievalBackend
from .backends.fake import FakeRetrievalBackend
from .backends.noop import NoOpRetrievalBackend
from .backends.pair_analog_prithvi import PairAnalogPrithviBackend
from .backends.static_region_prithvi import StaticRegionPrithviBackend
from .backends.text_bitemporal_remoteclip import TextBitemporalRemoteCLIPBackend

_BACKENDS: dict[str, type[RetrievalBackend]] = {
    "fake": FakeRetrievalBackend,
    "noop": NoOpRetrievalBackend,
    "pair_analog_prithvi": PairAnalogPrithviBackend,
    "static_region_prithvi": StaticRegionPrithviBackend,
    "text_bitemporal_remoteclip": TextBitemporalRemoteCLIPBackend,
}


def register_retrieval_backend(name: str, backend_cls: type[RetrievalBackend]) -> None:
    _BACKENDS[name] = backend_cls


def available_retrieval_backends() -> list[str]:
    return sorted(_BACKENDS)


def get_retrieval_backend(
    backend_name: str,
    model_dir: str | Path,
    device: str,
) -> RetrievalBackend:
    backend_cls = _BACKENDS.get(backend_name)
    if backend_cls is None:
        raise ValueError(f"Unknown retrieval backend: {backend_name}")
    return backend_cls(model_dir=model_dir, device=device)
