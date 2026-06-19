from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_dinov2_small import parse_args as parse_download_args
from scripts.setup_rs_change_project import ensure_layout


def test_download_dinov2_defaults_to_project_dot_cache(monkeypatch):
    monkeypatch.setenv("RS_PROJECT_ROOT", "/mnt/weka/tester/rs_change_project")
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setattr(sys, "argv", ["download_dinov2_small.py"])

    args = parse_download_args()

    assert args.output_dir == Path("/mnt/weka/tester/rs_change_project/models/dinov2-small")
    assert args.cache_dir == Path("/mnt/weka/tester/rs_change_project/.cache/huggingface")


def test_setup_layout_creates_models_and_dot_cache(tmp_path: Path):
    paths = ensure_layout(tmp_path)

    assert tmp_path / ".cache" in paths
    assert tmp_path / ".cache" / "huggingface" in paths
    assert tmp_path / "models" in paths
    assert (tmp_path / ".cache" / "huggingface").exists()
    assert (tmp_path / "models").exists()
