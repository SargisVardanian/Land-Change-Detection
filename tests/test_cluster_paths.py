from __future__ import annotations

from pathlib import Path

import pytest

from land_change_detection.cluster import YSUClusterLayout, ensure_symlink


def test_default_cluster_layout_paths():
    layout = YSUClusterLayout(repo_root=Path("/repo"))

    assert layout.weka_root == Path("/mnt/weka/svardanyan/land-change-detection")
    assert layout.data_root == Path("/mnt/weka/svardanyan/land-change-detection/data")
    assert layout.artifacts_root == Path("/mnt/weka/svardanyan/land-change-detection/artifacts")
    assert layout.conda_base == Path("/home/svardanyan/miniconda3")


def test_shell_exports_include_expected_roots():
    layout = YSUClusterLayout(
        repo_root=Path("/repo"),
        cluster_user="alice",
        project_slug="lcd",
        weka_base=Path("/weka/custom"),
        home_base=Path("/home/custom"),
        conda_env_name="geo",
    )

    exports = layout.shell_exports()

    assert exports["LCDC_CLUSTER_USER"] == "alice"
    assert exports["LCDC_WEKA_ROOT"] == "/weka/custom"
    assert exports["LCDC_DATA_ROOT"] == "/weka/custom/data"
    assert exports["LCDC_HOME_ROOT"] == "/home/custom"
    assert exports["LCDC_CONDA_ENV"] == "geo"


def test_ensure_symlink_creates_link(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"

    ensure_symlink(link, target)

    assert link.is_symlink()
    assert link.resolve() == target.resolve()


def test_ensure_symlink_rejects_existing_directory(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.mkdir()

    with pytest.raises(IsADirectoryError):
        ensure_symlink(link, target, force=True)
