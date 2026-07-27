from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "ysu_cluster_preflight.py"
    spec = importlib.util.spec_from_file_location("ysu_cluster_preflight", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_inspect_ssh_config_missing(tmp_path: Path):
    module = _load_module()
    exists, alias_present = module.inspect_ssh_config(tmp_path / "config", "ysu-hpc")
    assert exists is False
    assert alias_present is False


def test_inspect_ssh_config_alias_present(tmp_path: Path):
    module = _load_module()
    config = tmp_path / "config"
    config.write_text("Host ysu-hpc\n    HostName cluster.ysu.am\n", encoding="utf-8")
    exists, alias_present = module.inspect_ssh_config(config, "ysu-hpc")
    assert exists is True
    assert alias_present is True


def test_run_preflight_records_dns_failure(tmp_path: Path):
    module = _load_module()
    config = tmp_path / "config"
    config.write_text("", encoding="utf-8")
    result = module.run_preflight(
        host="definitely.invalid.example.codex",
        ssh_alias="ysu-hpc",
        conda_alias="ysu-hpc-conda",
        user="alice",
        terminal_target="conda",
        ssh_config=config,
        timeout=1,
    )
    assert result.host_resolves is False
    assert any("DNS did not return an address" in note for note in result.notes)


def test_recommended_ssh_target_uses_terminal_suffix():
    module = _load_module()
    assert module.recommended_ssh_target("cluster.ysu.am", "alice", "conda") == "alice+conda@cluster.ysu.am"
