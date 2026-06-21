from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

try:
    from safetensors.torch import load_file, save_file
except ModuleNotFoundError:  # pragma: no cover
    load_file = None
    save_file = None


def save_tensor_shard(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if save_file is not None:
        save_file(tensors, str(path), metadata=metadata or {})
        return
    torch.save({"tensors": tensors, "metadata": metadata or {}}, path)


def load_tensor_shard(path: Path) -> dict[str, torch.Tensor]:
    if load_file is not None:
        return load_file(str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload["tensors"])


def write_index(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_index(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class IndexedShardReader:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index = read_index(index_path)
        self.root = index_path.parent
        self._cache: dict[str, dict[str, torch.Tensor]] = {}

    def _load(self, relative_path: str) -> dict[str, torch.Tensor]:
        if relative_path not in self._cache:
            self._cache[relative_path] = load_tensor_shard((self.root / relative_path).resolve())
        return self._cache[relative_path]

    def get(self, shard_path: str, key: str) -> torch.Tensor:
        shard = self._load(shard_path)
        return shard[key]
