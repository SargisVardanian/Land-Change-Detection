from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_stage2_candidate_release_is_additive_and_hold_gated(tmp_path: Path) -> None:
    core = tmp_path / "core"
    for directory in (core / "registries", core / "manifests", core / "reports", core / "hashes"):
        directory.mkdir(parents=True)
    _write_jsonl(core / "registries/pair_registry.jsonl", [{"canonical_pair_id": "levir_mci:train:0"}])
    _write_jsonl(core / "registries/caption_registry.jsonl", [])
    _write_jsonl(core / "registries/relevance_registry.jsonl", [])
    _write_jsonl(core / "registries/dense_label_registry.jsonl", [])
    for split in ("train", "development", "test"):
        (core / f"manifests/retrieval_semantic_{split}_v2.jsonl").write_text("")
    (core / "manifests/dense_evaluation_v2.jsonl").write_text("")

    image_dir = tmp_path / "images"; label_dir = tmp_path / "labels"
    image_dir.mkdir(); label_dir.mkdir()
    image_a = image_dir / "a.png"; image_b = image_dir / "b.png"; pre = label_dir / "pre.png"; post = label_dir / "post.png"
    image_a.write_bytes(b"a"); image_b.write_bytes(b"b"); pre.write_bytes(b"pre"); post.write_bytes(b"post")
    rscc = {
        "canonical_pair_id": "rscc_ebd:EVENT:0", "source_scene_group_id": "rscc_ebd:event:EVENT", "source_event_id": "EVENT",
        "split": "train", "t1_path": str(image_a), "t2_path": str(image_b),
        "dense_label_sidecar": {"pre_mask_path": str(pre), "post_mask_path": str(post)},
    }
    rscc_rows = [
        dict(rscc, canonical_pair_id=f"rscc_ebd:EVENT:{index}", split=split)
        for index, split in enumerate(("train", "development", "test"))
    ]
    rscc_path = tmp_path / "rscc.jsonl"; _write_jsonl(rscc_path, rscc_rows)
    qvq_path = tmp_path / "qvq.jsonl"; _write_jsonl(qvq_path, [{"canonical_pair_id": row["canonical_pair_id"]} for row in rscc_rows])
    qvq_audit = tmp_path / "qvq_audit.json"; qvq_audit.write_text(json.dumps({"caption_count": 1, "training_enabled": False, "human_audit_passed": False}))
    structured = tmp_path / "structured"
    structured.mkdir()
    structured_row = {
        "schema_version": "temporal-caption-manifest-v1", "query_id": "s2looking:train:0:q", "canonical_pair_id": "s2looking:train:0",
        "pair_id": "s2looking:train:0", "dataset_name": "s2looking", "split": "train", "t1_path": str(image_a), "t2_path": str(image_b),
        "text": "new buildings appeared", "normalized_text": "new buildings appeared", "captions": ["new buildings appeared"],
        "normalized_caption_groups": ["new buildings appeared"], "caption_source": "official_dense_label_semantics", "generator": "official_label_template_v1",
        "semantic_group_id": "s2looking:official_relation:appeared", "verification_status": "structured_source_verified", "training_enabled": True,
        "query_scope": "semantic_group", "provenance": {"dense_labels_remain_sidecar_only": True},
    }
    for split in ("train", "development", "test"):
        _write_jsonl(structured / f"retrieval_semantic_structured_{split}.jsonl", [structured_row] if split == "train" else [])
    _write_jsonl(structured / "semantic_group_registry_structured.jsonl", [{"semantic_group_id": structured_row["semantic_group_id"], "split": "train", "pair_ids": [structured_row["canonical_pair_id"], "s2looking:train:1"], "pair_count": 2}])
    (structured / "structured_semantic_audit.json").write_text(json.dumps({"status": "STRUCTURED_SOURCE_SEMANTIC_READY", "training_enabled": True, "output_row_count": 1}))
    (structured / "structured_semantic_loader_contract.json").write_text(json.dumps({"passed": True}))
    (structured / "independent_structured_semantic_verification.json").write_text(json.dumps({
        "status": "INDEPENDENT_STRUCTURED_SEMANTIC_VERIFIED",
        "verified_rows": 1,
        "mask_free_output": True,
    }))
    stage2 = tmp_path / "stage2"; stage2.mkdir(); (stage2 / "source_registry.json").write_text(json.dumps({"status": "DATA_QUALITY_HOLD"}))
    output = tmp_path / "release"
    subprocess.run([
        sys.executable, "scripts/build_qcpr_stage2_candidate_release.py", "--repo", str(Path.cwd()), "--core-root", str(core),
        "--rscc-pairs", str(rscc_path), "--rscc-qvq", str(qvq_path), "--rscc-qvq-audit", str(qvq_audit),
        "--structured-dir", str(structured), "--stage2-audit-root", str(stage2), "--output-root", str(output),
    ], check=True)
    release = json.loads((output / "dataset_v2_stage2_release.json").read_text())
    assert release["status"] == "DATA_QUALITY_HOLD"
    assert release["training_authorized"] is False
    assert release["new_real_physical_source"]["pair_count"] == 3
    assert (output / "manifests/physical_pairs_rscc_ebd.jsonl").is_file()
    assert json.loads((output / "registries/pair_registry.jsonl").read_text().splitlines()[-1])["canonical_pair_id"] == rscc_rows[-1]["canonical_pair_id"]
    assert "mask_path" not in (output / "manifests/retrieval_semantic_train_v2.jsonl").read_text()
    assert (core / "registries/pair_registry.jsonl").read_text() == '{"canonical_pair_id": "levir_mci:train:0"}\n'
