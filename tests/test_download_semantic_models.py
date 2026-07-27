from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "download_semantic_models.py"
    spec = importlib.util.spec_from_file_location("download_semantic_models", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_model_targets_runtime_only(tmp_path: Path):
    module = _load_module()
    targets = module.build_model_targets(tmp_path, include_research=False)
    assert set(targets) == {"mfaytin/mask2former-satellite"}
    assert targets["mfaytin/mask2former-satellite"] == tmp_path / "mask2former-satellite"


def test_build_model_targets_with_research(tmp_path: Path):
    module = _load_module()
    targets = module.build_model_targets(tmp_path, include_research=True)
    assert "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL" in targets
    assert "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL" in targets
    assert targets["ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"] == tmp_path / "Prithvi-EO-2.0-300M-TL"
