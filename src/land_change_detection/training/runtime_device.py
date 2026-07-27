from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


def resolve_runtime_device(device: torch.device | str) -> torch.device:
    """Canonicalize a runtime device without weakening explicit device checks."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested, but CUDA is unavailable")
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def assert_runtime_tensor_devices(outputs: Mapping[str, Tensor], device: torch.device | str) -> torch.device:
    """Return the canonical device after verifying every output is exactly on it."""
    resolved = resolve_runtime_device(device)
    for name, tensor in outputs.items():
        if tensor.device != resolved:
            raise RuntimeError(f"{name} is on {tensor.device}, expected {resolved}")
    return resolved
