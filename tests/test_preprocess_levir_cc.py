from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from PIL import Image


def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _run_preprocess(root: Path, project_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/preprocess_levir_cc.py",
            "--root",
            str(root),
            "--project-root",
            str(project_root),
            *extra,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )


def _make_zip(path: Path, files: dict[str, bytes | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)


def _make_tar(path: Path, files: dict[str, bytes | str], gz: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w:gz" if gz else "w"
    with tarfile.open(path, mode) as archive:
        for name, payload in files.items():
            data = payload.encode("utf-8") if isinstance(payload, str) else payload
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _archive_payload() -> dict[str, bytes | str]:
    captions = [
        {"id": "pair_a", "caption": "new building appears", "transition_label": "urban_growth"},
        {"id": "pair_a", "caption": "new building appears", "transition_label": "urban_growth"},
        {"id": "pair_b", "caption": "water shrinks", "transition_label": "water_change"},
        {"id": "pair_c", "caption": "forest loss", "transition_label": "forest_loss"},
    ]
    return {
        "hf_snapshot/train/A/pair_a.png": _png_bytes((255, 0, 0)),
        "hf_snapshot/train/B/pair_a.png": _png_bytes((0, 255, 0)),
        "hf_snapshot/val/A/pair_b.png": _png_bytes((10, 20, 30)),
        "hf_snapshot/val/B/pair_b.png": _png_bytes((30, 20, 10)),
        "hf_snapshot/test/A/pair_c.png": _png_bytes((0, 0, 255)),
        "hf_snapshot/test/B/pair_c.png": _png_bytes((255, 255, 0)),
        "hf_snapshot/LevirCCcaptions.json": json.dumps(captions),
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_preprocess_levir_cc_valid_zip_layout_and_idempotent_hashes(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    raw_root = project_root / "datasets" / "raw" / "LEVIR-CC"
    _make_zip(raw_root / "levir_cc_snapshot.zip", _archive_payload())

    first = _run_preprocess(raw_root, project_root)
    assert first.returncode == 0, first.stdout + first.stderr
    report_a = json.loads((project_root / "reports" / "levir_cc_preprocess_report.json").read_text(encoding="utf-8"))

    second = _run_preprocess(raw_root, project_root)
    assert second.returncode == 0, second.stdout + second.stderr
    report_b = json.loads((project_root / "reports" / "levir_cc_preprocess_report.json").read_text(encoding="utf-8"))

    assert report_a["used_archives"] is True
    assert report_a["manifest_hashes"] == report_b["manifest_hashes"]
    assert report_a["raw_archive_sha256"] == report_b["raw_archive_sha256"]
    assert report_a["extracted_file_count"] == 7
    train_rows = _read_jsonl(project_root / "indexes" / "levir_cc_train.jsonl")
    assert len(train_rows) == 2
    assert train_rows[0]["caption"] == "new building appears"
    assert train_rows[1]["caption"] == "new building appears"
    assert train_rows[0]["caption_index"] == 0
    assert train_rows[1]["caption_index"] == 1


def test_preprocess_levir_cc_valid_tar_layout(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    raw_root = project_root / "datasets" / "raw" / "LEVIR-CC"
    _make_tar(raw_root / "levir_cc_snapshot.tar", _archive_payload())

    result = _run_preprocess(raw_root, project_root)
    assert result.returncode == 0, result.stdout + result.stderr
    pairs = _read_jsonl(project_root / "indexes" / "levir_cc_pairs.jsonl")
    assert len(pairs) == 3


def test_preprocess_levir_cc_rejects_archive_path_traversal(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    raw_root = project_root / "datasets" / "raw" / "LEVIR-CC"
    payload = _archive_payload()
    payload["../escape.txt"] = "bad"
    _make_zip(raw_root / "bad.zip", payload)

    result = _run_preprocess(raw_root, project_root)
    assert result.returncode != 0
    assert "path_traversal" in (result.stdout + result.stderr)


def test_preprocess_levir_cc_rejects_incomplete_pairs(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    raw_root = project_root / "datasets" / "raw" / "LEVIR-CC"
    payload = _archive_payload()
    payload.pop("hf_snapshot/test/B/pair_c.png")
    _make_tar(raw_root / "missing_pair.tgz", payload, gz=True)

    result = _run_preprocess(raw_root, project_root)
    assert result.returncode != 0
    assert "missing_pair_half" in result.stdout


def test_preprocess_levir_cc_verify_only_changes_no_files(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    raw_root = project_root / "datasets" / "raw" / "LEVIR-CC"
    _make_zip(raw_root / "levir_cc_snapshot.zip", _archive_payload())
    initial = _run_preprocess(raw_root, project_root)
    assert initial.returncode == 0, initial.stdout + initial.stderr

    manifest_before = (project_root / "datasets" / "processed" / "LEVIR-CC" / "levir_cc_extraction_manifest.json").read_text(encoding="utf-8")
    report_before = (project_root / "reports" / "levir_cc_preprocess_report.json").read_text(encoding="utf-8")
    verify = _run_preprocess(raw_root, project_root, "--verify-only")
    assert verify.returncode == 0, verify.stdout + verify.stderr
    manifest_after = (project_root / "datasets" / "processed" / "LEVIR-CC" / "levir_cc_extraction_manifest.json").read_text(encoding="utf-8")
    report_after = (project_root / "reports" / "levir_cc_preprocess_report.json").read_text(encoding="utf-8")
    assert manifest_before == manifest_after
    assert report_before == report_after


def test_preprocess_levir_cc_verify_only_requires_existing_extraction(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    raw_root = project_root / "datasets" / "raw" / "LEVIR-CC"
    _make_zip(raw_root / "levir_cc_snapshot.zip", _archive_payload())
    result = _run_preprocess(raw_root, project_root, "--verify-only")
    assert result.returncode != 0
    assert "verify-only" in (result.stdout + result.stderr)
