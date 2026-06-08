from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "write_ysu_ssh_config.py"
    spec = importlib.util.spec_from_file_location("write_ysu_ssh_config", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_upsert_host_block_adds_new_alias():
    module = _load_module()
    updated = module.upsert_host_block("", "ysu-hpc", module.render_host_block("ysu-hpc", "cluster.ysu.am", "alice"))
    assert "Host ysu-hpc" in updated
    assert "User alice" in updated


def test_upsert_host_block_replaces_existing_alias():
    module = _load_module()
    existing = (
        "Host ysu-hpc\n"
        "    HostName old.example\n"
        "    User olduser\n"
        "\n"
        "Host github.com\n"
        "    User git\n"
    )
    updated = module.upsert_host_block(
        existing,
        "ysu-hpc",
        module.render_host_block("ysu-hpc", "cluster.ysu.am", "newuser"),
    )
    assert "HostName cluster.ysu.am" in updated
    assert "User newuser" in updated
    assert "old.example" not in updated
    assert "Host github.com" in updated
