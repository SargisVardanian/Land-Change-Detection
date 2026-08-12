from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_qcpr_siglip2_native_highres import _read_highres_manifest


def test_highres_manifest_requires_authorized_runtime_contract(tmp_path: Path) -> None:
    path = tmp_path / "highres.json"
    path.write_text(
        json.dumps({"HIGHRES_RUNTIME_STRESS_READY": True, "items": [{"item_id": "x"}]}),
        encoding="utf-8",
    )
    assert _read_highres_manifest(path) == [{"item_id": "x"}]

    path.write_text(
        json.dumps({"HIGHRES_RUNTIME_STRESS_READY": False, "items": [{"item_id": "x"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not authorized"):
        _read_highres_manifest(path)
