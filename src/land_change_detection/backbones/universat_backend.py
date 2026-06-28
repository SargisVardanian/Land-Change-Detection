from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LevirRGBSpec:
    channel_order: tuple[str, str, str] = ("red", "green", "blue")
    normalization: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "status": ("assumed_imagenet_rgb", "assumed_imagenet_rgb", "assumed_imagenet_rgb"),
        }
    )
    nominal_wavelengths_nm: dict[str, Any] = field(
        default_factory=lambda: {
            "red": None,
            "green": None,
            "blue": None,
            "status": "unknown_for_LEVIR_CC",
        }
    )
    gsd: dict[str, Any] = field(default_factory=lambda: {"value": None, "unit": "m/pixel", "status": "unknown"})
    temporal_order: str = "relative_before_after"
    sensor_name: str = "unknown_RGB_aerial_or_remote_sensing_source"

    def to_universat_metadata(self) -> dict[str, Any]:
        return {
            "sensor_name": self.sensor_name,
            "channels": list(self.channel_order),
            "normalization": self.normalization,
            "nominal_wavelengths_nm": self.nominal_wavelengths_nm,
            "gsd": self.gsd,
            "temporal_order": self.temporal_order,
            "calendar_dates": None,
        }


@dataclass(frozen=True)
class VisualFeatureGrid:
    global_embedding: Tensor
    local_tokens: Tensor
    grid_height: int
    grid_width: int
    original_height: int
    original_width: int
    metadata: dict


@dataclass(frozen=True)
class UniverSatBackendConfig:
    source_dir: str | Path
    checkpoint_dir: str | Path
    source_commit: str = "f6df2eec54955b0f7524cc95fe21a5e80c0239d9"
    output_grid: int = 36
    visual_dim: int = 768
    retrieval_dim: int = 512
    freeze: bool = True
    local_files_only: bool = True
    sensor_spec: LevirRGBSpec = field(default_factory=LevirRGBSpec)
    metadata: dict = field(default_factory=dict)


class UniverSatJointBackend(nn.Module):
    """Joint temporal UniverSat wrapper for the public `[B, T, C, H, W]` path."""

    def __init__(self, config: UniverSatBackendConfig, model: nn.Module | None = None):
        super().__init__()
        self.config = config
        self.source_dir = Path(config.source_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.model = model if model is not None else self._load_external_model()
        self.local_projection = nn.Linear(config.visual_dim, config.retrieval_dim)
        self.global_pool = nn.Sequential(nn.LayerNorm(config.retrieval_dim), nn.Linear(config.retrieval_dim, config.retrieval_dim))
        if config.freeze:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def _load_external_model(self) -> nn.Module:
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Missing UniverSat source checkout: {self.source_dir}")
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Missing UniverSat checkpoint directory: {self.checkpoint_dir}")
        sys.path.insert(0, str(self.source_dir))
        sys.path.insert(0, str(self.source_dir / "src"))
        errors: list[str] = []
        for module_name, builder_name in (
            ("hubconf", "UniverSat"),
            ("hubconf", "from_pretrained"),
            ("universat", "from_pretrained"),
            ("universat", "UniverSat"),
            ("model", "from_pretrained"),
            ("models", "from_pretrained"),
        ):
            try:
                module = importlib.import_module(module_name)
                builder = getattr(module, builder_name)
                if module_name == "hubconf" and builder_name == "UniverSat":
                    return builder.from_pretrained(
                        str(self.checkpoint_dir),
                        local_files_only=self.config.local_files_only,
                    )
                if builder_name == "from_pretrained":
                    return builder(str(self.checkpoint_dir), local_files_only=self.config.local_files_only)
                return builder(checkpoint_path=str(self.checkpoint_dir))
            except Exception as exc:  # pragma: no cover - depends on external checkout layout.
                errors.append(f"{module_name}.{builder_name}: {exc}")
        raise RuntimeError("Could not construct UniverSat from the pinned checkout. Tried: " + "; ".join(errors))

    @property
    def model_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @property
    def dtype(self) -> str:
        try:
            return str(next(self.model.parameters()).dtype)
        except StopIteration:
            return "unknown"

    def _call_universat(self, pair: Tensor) -> Tensor:
        sensor_metadata = self.config.sensor_spec.to_universat_metadata()
        relative_dates = torch.arange(pair.shape[1], device=pair.device, dtype=torch.long).unsqueeze(0).expand(pair.shape[0], -1)
        encode_payload = {"spot": pair, "spot_dates": relative_dates}
        attempts = (
            lambda: self.model.encode(
                encode_payload,
                patch_size=10.0,
                output_grid=self.config.output_grid,
            ),
            lambda: self.model.encode(
                encode_payload,
                patch_size=10.0,
                output_grid=self.config.output_grid,
                wavelengths={"spot": [665.0, 560.0, 490.0]},
                input_res={"spot": 1.0},
                subpatches={"spot": 1},
            ),
            lambda: self.model(pair, output_grid=self.config.output_grid, **sensor_metadata),
            lambda: self.model(pair, output_grid=self.config.output_grid),
            lambda: self.model(pair),
        )
        last_error: Exception | None = None
        for attempt in attempts:
            try:
                output = attempt()
                break
            except Exception as exc:  # pragma: no cover - external source API may vary.
                last_error = exc
        else:
            raise RuntimeError(f"UniverSat forward failed for joint temporal input: {last_error}") from last_error
        if isinstance(output, dict):
            for key in ("tokens", "last_hidden_state", "features", "x"):
                value = output.get(key)
                if isinstance(value, Tensor):
                    output = value
                    break
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, Tensor):
            raise TypeError(f"UniverSat output must resolve to a Tensor, got {type(output)!r}.")
        return output

    def forward(self, t1: Tensor, t2: Tensor) -> VisualFeatureGrid:
        if t1.shape != t2.shape:
            raise ValueError(f"t1 and t2 must have the same shape, got {tuple(t1.shape)} and {tuple(t2.shape)}.")
        if t1.ndim != 4:
            raise ValueError("t1 and t2 must have shape [B, C, H, W].")
        original_height, original_width = int(t1.shape[-2]), int(t1.shape[-1])
        pair = torch.stack([t1, t2], dim=1)
        with torch.set_grad_enabled(not self.config.freeze):
            tokens = self._call_universat(pair)
        if tokens.ndim == 4:
            tokens = tokens.flatten(1, 2) if tokens.shape[1] == self.config.output_grid else tokens.flatten(2).transpose(1, 2)
        if tokens.ndim != 3:
            raise ValueError(f"Expected UniverSat tokens [B, N, C], got {tuple(tokens.shape)}.")
        expected_tokens = self.config.output_grid * self.config.output_grid
        if tokens.shape[1] != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} spatial tokens, got {tokens.shape[1]}.")
        if tokens.shape[-1] != self.config.visual_dim:
            raise ValueError(f"Expected visual dim {self.config.visual_dim}, got {tokens.shape[-1]}.")
        local_tokens = F.normalize(self.local_projection(tokens), dim=-1)
        global_embedding = F.normalize(self.global_pool(local_tokens.mean(dim=1)), dim=-1)
        return VisualFeatureGrid(
            global_embedding=global_embedding,
            local_tokens=local_tokens,
            grid_height=self.config.output_grid,
            grid_width=self.config.output_grid,
            original_height=original_height,
            original_width=original_width,
            metadata={
                "backend": "universat_joint_public",
                "repo_id": "g-astruc/UniverSat",
                "source_dir": str(self.source_dir),
                "checkpoint_dir": str(self.checkpoint_dir),
                "source_commit": self.config.source_commit,
                "output_grid": self.config.output_grid,
                "sensor_spec": asdict(self.config.sensor_spec),
                "universat_modality_adapter": "registered_rgb_vhr_adapter_for_unknown_LEVIR_RGB",
                "relative_dates": [0, 1],
                "frozen": self.config.freeze,
                **self.config.metadata,
            },
        )
