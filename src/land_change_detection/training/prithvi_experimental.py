from __future__ import annotations

from dataclasses import dataclass, asdict
import importlib
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PrithviExperimentalTrainingConfig:
    manifest_path: str
    output_dir: str
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 1e-3
    num_classes: int = 12
    allow_fallback_baseline: bool = True
    require_terratorch: bool = False
    run_tag: str = "prithvi_terratorch_experimental"
    checkpoint_path: str | None = None
    terratorch_config_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrithviExperimentalStatus:
    mode: str
    terratorch_available: bool
    manifest_rows: int
    output_dir: str
    run_tag: str
    notes: list[str]
    checkpoint_path: str | None = None
    terratorch_config_path: str | None = None
    runtime_ready: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrithviRuntimeResolution:
    checkpoint_path: str | None
    terratorch_config_path: str | None
    runtime_ready: bool
    notes: list[str]
    config_valid: bool = False
    config_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrithviRuntimeBundle:
    checkpoint_path: str | None
    terratorch_config_path: str | None
    terratorch_available: bool
    config_valid: bool
    runtime_ready: bool
    config_summary: dict[str, Any] | None
    notes: list[str]
    module_summary: dict[str, Any] | None = None
    factory_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def terratorch_available() -> bool:
    try:
        import terratorch  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def count_manifest_rows(path: str | Path) -> int:
    total = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            total += 1
    return total


def resolve_prithvi_runtime(
    *,
    checkpoint_path: str | None = None,
    terratorch_config_path: str | None = None,
    model_dir: str | Path = "artifacts/models/semantic/Prithvi-EO-2.0-300M-TL",
) -> PrithviRuntimeResolution:
    notes: list[str] = []
    model_root = Path(model_dir)

    resolved_checkpoint = _resolve_first_existing(
        checkpoint_path,
        [
            model_root / "model.pt",
            model_root / "checkpoint.pt",
            model_root / "pytorch_model.bin",
            model_root / "model.safetensors",
        ],
    )
    if resolved_checkpoint is None:
        notes.append("No Prithvi checkpoint candidate was found.")

    resolved_config = _resolve_first_existing(
        terratorch_config_path,
        [
            model_root / "terratorch_config.yaml",
            model_root / "terratorch_config.yml",
            model_root / "config.yaml",
            model_root / "config.yml",
        ],
    )
    if resolved_config is None:
        notes.append("No TerraTorch config candidate was found.")
        config_valid = False
        config_summary = None
    else:
        config_valid, config_summary, config_notes = inspect_terratorch_config(resolved_config)
        notes.extend(config_notes)

    runtime_ready = (
        resolved_checkpoint is not None
        and resolved_config is not None
        and config_valid
        and terratorch_available()
    )
    if not terratorch_available():
        notes.append("TerraTorch is not installed in the current environment.")
    return PrithviRuntimeResolution(
        checkpoint_path=str(resolved_checkpoint) if resolved_checkpoint else None,
        terratorch_config_path=str(resolved_config) if resolved_config else None,
        runtime_ready=runtime_ready,
        notes=notes,
        config_valid=config_valid,
        config_summary=config_summary,
    )


def build_prithvi_runtime_bundle(
    *,
    checkpoint_path: str | None = None,
    terratorch_config_path: str | None = None,
    model_dir: str | Path = "artifacts/models/semantic/Prithvi-EO-2.0-300M-TL",
) -> PrithviRuntimeBundle:
    resolution = resolve_prithvi_runtime(
        checkpoint_path=checkpoint_path,
        terratorch_config_path=terratorch_config_path,
        model_dir=model_dir,
    )
    module_summary = inspect_terratorch_module()
    factory_summary = inspect_terratorch_factories()
    notes = list(resolution.notes)
    if module_summary["available"] and not module_summary["required_symbols_present"]:
        notes.append("TerraTorch module is importable, but expected runtime symbols were not all discovered.")
    if factory_summary["available"] and not factory_summary["factory_candidates"]:
        notes.append("TerraTorch module is importable, but no expected factory candidates were discovered.")
    return PrithviRuntimeBundle(
        checkpoint_path=resolution.checkpoint_path,
        terratorch_config_path=resolution.terratorch_config_path,
        terratorch_available=terratorch_available(),
        config_valid=resolution.config_valid,
        runtime_ready=resolution.runtime_ready,
        config_summary=resolution.config_summary,
        notes=notes,
        module_summary=module_summary,
        factory_summary=factory_summary,
    )


def _resolve_first_existing(explicit: str | None, candidates: list[Path]) -> Path | None:
    if explicit:
        explicit_path = Path(explicit)
        return explicit_path if explicit_path.exists() else None
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def inspect_terratorch_config(path: str | Path) -> tuple[bool, dict[str, Any] | None, list[str]]:
    config_path = Path(path)
    notes: list[str] = []
    try:
        payload = _parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, None, [f"Failed to parse TerraTorch config: {type(exc).__name__}: {exc}"]

    required = ["model_name", "task", "num_input_channels", "num_classes"]
    missing = [key for key in required if key not in payload]
    if missing:
        notes.append(f"TerraTorch config is missing required keys: {', '.join(missing)}")
        return False, payload, notes

    try:
        payload["num_input_channels"] = int(payload["num_input_channels"])
        payload["num_classes"] = int(payload["num_classes"])
    except Exception:
        notes.append("TerraTorch config has non-integer num_input_channels or num_classes.")
        return False, payload, notes

    if payload["num_input_channels"] != 6:
        notes.append("TerraTorch config is not aligned with the current 6-band Prithvi contract.")
    return True, payload, notes


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _coerce_scalar(value.strip())
    return data


def _coerce_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value.strip("\"'")


def inspect_terratorch_module(module_name: str = "terratorch") -> dict[str, Any]:
    required_symbols = ["__file__"]
    candidate_submodules = [
        "terratorch",
        "terratorch.tasks",
        "terratorch.models",
        "terratorch.registry",
    ]
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "available": False,
            "module_name": module_name,
            "error": f"{type(exc).__name__}: {exc}",
            "required_symbols_present": False,
            "candidate_submodules": [],
        }

    discovered = []
    for name in candidate_submodules:
        try:
            importlib.import_module(name)
            discovered.append(name)
        except Exception:
            continue
    return {
        "available": True,
        "module_name": module_name,
        "module_file": getattr(module, "__file__", None),
        "required_symbols_present": all(hasattr(module, symbol) for symbol in required_symbols),
        "candidate_submodules": discovered,
    }


def inspect_terratorch_factories(module_name: str = "terratorch") -> dict[str, Any]:
    candidate_modules = [
        "terratorch",
        "terratorch.models",
        "terratorch.tasks",
        "terratorch.registry",
    ]
    candidate_names = [
        "EncoderDecoderFactory",
        "BACKBONE_REGISTRY",
        "MODEL_FACTORY_REGISTRY",
        "build_model",
        "create_model",
        "build_task",
        "create_task",
    ]
    discovered: list[dict[str, str]] = []
    module_errors: dict[str, str] = {}
    any_available = False
    for mod_name in candidate_modules:
        try:
            module = importlib.import_module(mod_name)
            any_available = True
        except Exception as exc:
            module_errors[mod_name] = f"{type(exc).__name__}: {exc}"
            continue
        for name in candidate_names:
            if hasattr(module, name):
                discovered.append({"module": mod_name, "name": name})
    return {
        "available": any_available,
        "module_name": module_name,
        "factory_candidates": discovered,
        "module_errors": module_errors,
    }
