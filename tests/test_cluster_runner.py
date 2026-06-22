from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = REPO_ROOT / "cluster" / "ysu"
FORBIDDEN_PYTHON_PATHS = (
    ".venv/bin/" + "python",
    "/mnt/weka/shared-cache/miniforge3/bin/" + "python",
    "/opt/conda/bin/" + "python",
)
SHARED_BASE_PYTHON = FORBIDDEN_PYTHON_PATHS[1]
CONDA_BASE_PYTHON = FORBIDDEN_PYTHON_PATHS[2]


def test_no_test_references_legacy_python_paths():
    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        content = path.read_text(encoding="utf-8")
        assert not any(marker in content for marker in FORBIDDEN_PYTHON_PATHS), path


def test_no_cluster_script_hardcodes_legacy_python_paths():
    for path in sorted(CLUSTER_DIR.glob("*")):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        assert SHARED_BASE_PYTHON not in content, path
        assert CONDA_BASE_PYTHON not in content, path
        assert "source ~/.bashrc" not in content, path


def test_cluster_shell_scripts_pass_bash_n():
    for path in sorted(CLUSTER_DIR.glob("*.sh")) + sorted(CLUSTER_DIR.glob("*.sbatch")):
        result = subprocess.run(["bash", "-n", str(path)], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"{path}: {result.stderr}"


def test_staged_runner_prints_simple_patch_dependency_chain(tmp_path: Path):
    counter_path = tmp_path / "counter.txt"
    sbatch_stub = tmp_path / "sbatch"
    sbatch_stub.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                f"counter_file='{counter_path}'",
                "count=0",
                "if [ -f \"$counter_file\" ]; then count=$(cat \"$counter_file\"); fi",
                "count=$((count + 1))",
                "echo \"$count\" > \"$counter_file\"",
                "printf 'job-%s\\n' \"$count\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sbatch_stub.chmod(0o755)
    env = dict(os.environ)
    env["SBATCH_BIN"] = str(sbatch_stub)
    env["RS_PROJECT_ROOT"] = str(tmp_path / "rs_change_project")
    result = subprocess.run(
        ["bash", "cluster/ysu/run_levir_cc_baseline_end_to_end.sh", "all"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines == [
        "preprocess: job-1",
        "validate-manifests: job-2",
        "random: job-3",
        "overfit: job-4",
        "train: job-5",
        "eval: job-6",
        "render: job-7",
    ]


def test_staged_runner_prints_dinov2_dependency_chain(tmp_path: Path):
    counter_path = tmp_path / "counter.txt"
    sbatch_stub = tmp_path / "sbatch"
    sbatch_stub.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                f"counter_file='{counter_path}'",
                "count=0",
                "if [ -f \"$counter_file\" ]; then count=$(cat \"$counter_file\"); fi",
                "count=$((count + 1))",
                "echo \"$count\" > \"$counter_file\"",
                "printf 'job-%s\\n' \"$count\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sbatch_stub.chmod(0o755)
    env = dict(os.environ)
    env["SBATCH_BIN"] = str(sbatch_stub)
    env["RS_PROJECT_ROOT"] = str(tmp_path / "rs_change_project")
    env["PRESET"] = "dinov2_t2_only"
    result = subprocess.run(
        ["bash", "cluster/ysu/run_levir_cc_baseline_end_to_end.sh", "all"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines == [
        "preprocess: job-1",
        "validate-manifests: job-2",
        "random: job-3",
        "validate-models: job-4",
        "smoke: job-5",
        "cache-dino: job-6",
        "cache-text: job-7",
        "overfit: job-8",
        "train: job-9",
        "eval: job-10",
        "render: job-11",
    ]
