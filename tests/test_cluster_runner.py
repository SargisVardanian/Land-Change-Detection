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
        "audit: job-8",
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
        "audit: job-12",
    ]


def test_submit_runner_prints_simple_patch_chain(tmp_path: Path):
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
    env["PATH"] = f"{tmp_path}:{env.get('PATH', '')}"
    env["RS_PROJECT_ROOT"] = str(tmp_path / "rs_change_project")
    result = subprocess.run(
        ["bash", "cluster/ysu/submit_levir_cc_baseline.sh"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines[:7] == [
        "Submitting LEVIR-CC baseline for preset=simple_patch_smoke run_name=simple_patch_smoke",
        "preprocess_levir_cc: job-1",
        "validate_levir_cc_pair_manifest: job-2",
        "eval_levir_cc_random_retrieval: job-3",
        "overfit_levir_cc_retrieval_100: job-4",
        "train_levir_cc_retrieval: job-5",
        "eval_levir_cc_retrieval: job-6",
    ]
    assert "render_levir_cc_text_query_grid: job-7" in lines
    assert "audit_levir_cc_milestone: job-8" in lines


def test_submit_runner_prints_dinov2_chain_with_model_validation(tmp_path: Path):
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
    env["PATH"] = f"{tmp_path}:{env.get('PATH', '')}"
    env["RS_PROJECT_ROOT"] = str(tmp_path / "rs_change_project")
    env["PRESET"] = "dinov2_t2_only"
    result = subprocess.run(
        ["bash", "cluster/ysu/submit_levir_cc_baseline.sh"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines[:10] == [
        "Submitting LEVIR-CC baseline for preset=dinov2_t2_only run_name=dinov2_t2_only",
        "preprocess_levir_cc: job-1",
        "validate_levir_cc_pair_manifest: job-2",
        "eval_levir_cc_random_retrieval: job-3",
        "validate_retrieval_model_assets: job-4",
        "smoke_levir_cc_dino_remoteclip_batch: job-5",
        "cache_levir_cc_dinov2_features: job-6",
        "cache_levir_cc_remoteclip_text: job-7",
        "overfit_levir_cc_retrieval_100: job-8",
        "train_levir_cc_retrieval: job-9",
    ]
    assert "eval_levir_cc_retrieval: job-10" in lines
    assert "render_levir_cc_text_query_grid: job-11" in lines
    assert "audit_levir_cc_milestone: job-12" in lines
