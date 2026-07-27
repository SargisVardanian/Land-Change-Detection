from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether the current environment is ready for TerraTorch/Prithvi experiments.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _safe_import_version(module_name: str) -> tuple[bool, str | None, str | None]:
    try:
        module = __import__(module_name)
    except Exception as exc:  # pragma: no cover - exercised through CLI behavior
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, getattr(module, "__version__", None), None


def main() -> int:
    args = parse_args()
    diagnostics: dict[str, object] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    torch_ok, torch_version, torch_error = _safe_import_version("torch")
    diagnostics["torch"] = {"available": torch_ok, "version": torch_version, "error": torch_error}
    terratorch_ok, terratorch_version, terratorch_error = _safe_import_version("terratorch")
    diagnostics["terratorch"] = {"available": terratorch_ok, "version": terratorch_version, "error": terratorch_error}
    tifffile_ok, tifffile_version, tifffile_error = _safe_import_version("tifffile")
    diagnostics["tifffile"] = {"available": tifffile_ok, "version": tifffile_version, "error": tifffile_error}

    try:
        import torch

        diagnostics["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover
        diagnostics["cuda"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    diagnostics["ready_for_experimental_prithvi"] = bool(
        diagnostics["torch"]["available"] and diagnostics["terratorch"]["available"] and diagnostics["tifffile"]["available"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
