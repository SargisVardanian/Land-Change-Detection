import json

from qcpr_data.sources.rsrcc import validate_physical_manifest


def test_validate_physical_manifest_checks_paths_and_hashes(tmp_path):
    image = tmp_path / "train/images/000/a.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    import hashlib
    digest = hashlib.sha256(b"image").hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "dataset": "google/RSRCC",
        "local_split": "train",
        "relative": "images/000/a.png",
        "sha256": digest,
    }) + "\n", encoding="utf-8")
    result = validate_physical_manifest(manifest, repository_root=tmp_path)
    assert result["passed"] is True
    assert result["asset_count"] == 1
