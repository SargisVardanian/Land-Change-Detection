from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


def _cpu_byte_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore checkpoint RNG state after arbitrary torch.load map_location.

    ``torch.load(..., map_location="cuda")`` can move saved RNG tensors onto
    CUDA. PyTorch's RNG restoration APIs require CPU ``torch.ByteTensor``
    values, so every state is normalized before it is applied.
    """

    if not state:
        return

    cpu_state = state.get("torch_cpu")
    if cpu_state is not None:
        torch.set_rng_state(_cpu_byte_tensor(cpu_state))

    if not torch.cuda.is_available():
        return

    saved_cuda = state.get("torch_cuda")
    if saved_cuda is None:
        return

    if isinstance(saved_cuda, Tensor):
        cuda_states: list[Any] = [saved_cuda]
    elif isinstance(saved_cuda, Sequence):
        cuda_states = list(saved_cuda)
    else:
        raise TypeError(
            "torch_cuda RNG state must be a tensor or a sequence of tensors, "
            f"got {type(saved_cuda).__name__}"
        )

    if not cuda_states:
        return

    current_device = torch.cuda.current_device()
    state_index = min(current_device, len(cuda_states) - 1)
    torch.cuda.set_rng_state(
        _cpu_byte_tensor(cuda_states[state_index]),
        device=current_device,
    )
