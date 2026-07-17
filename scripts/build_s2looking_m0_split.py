#!/usr/bin/env python3
"""Build the immutable M0 directional split from time-corrected S2Looking pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

INSPECTED_BASE_IDS = frozenset({"1507", "4138", "1078", "1003", "2164"})
SPLIT_SEED = "qcpr-v31-m0-a936f116"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_nonempty(path: str) -> bool:
    with Image.open(path) as image:
        return bool(np.asarray(image.convert("L")).any())


def stable_key(row: dict) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{row['pair_id']}".encode()).hexdigest()


def base_id(row: dict) -> str:
    return str(row.get("original_id") or str(row["pair_id"]).rsplit(":", 1)[-1])


def is_eligible(row: dict) -> bool:
    targets = list(row.get("directional_targets", []))
    directions = {str(target.get("direction")) for target in targets}
    return (
        row.get("dataset_name") == "s2looking"
        and directions == {"appeared", "disappeared"}
        and str(row.get("quality_tier")) in {"core", "hard"}
        and float(row.get("directional_overlap_iou", 1.0)) <= 0.10
        and all(mask_nonempty(str(target["mask_path"])) for target in targets)
    )


def copy_with_split(row: dict, split: str) -> dict:
    result = json.loads(json.dumps(row))
    result["split"] = split
    result.setdefault("source_metadata", {})["m0_split"] = split
    result["source_metadata"]["m0_split_seed"] = SPLIT_SEED
    result["source_metadata"]["pre_a936f116_mask_local_artifacts"] = "INVALID"
    return result


def write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def select(rows: list[dict], *, count: int, forbidden: set[str]) -> list[dict]:
    available = [row for row in rows if str(row["pair_id"]) not in forbidden]
    if len(available) < count:
        raise RuntimeError(f"need {count} eligible rows, found {len(available)}")
    return sorted(available, key=stable_key)[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible = [row for row in rows if is_eligible(row)]
    by_split = {name: [row for row in eligible if row.get("split") == name] for name in ("train", "val", "test")}

    inspected = [row for row in eligible if base_id(row) in INSPECTED_BASE_IDS]
    found = {base_id(row) for row in inspected}
    if found != INSPECTED_BASE_IDS:
        raise RuntimeError(f"inspected IDs absent or ineligible: {sorted(INSPECTED_BASE_IDS - found)}")

    used = {str(row["pair_id"]) for row in inspected}
    development = inspected + select(by_split["val"], count=32 - len(inspected), forbidden=used)
    used |= {str(row["pair_id"]) for row in development}
    test = select(by_split["test"], count=32, forbidden=used)
    used |= {str(row["pair_id"]) for row in test}
    train = select(by_split["train"], count=128, forbidden=used)

    if any(base_id(row) in INSPECTED_BASE_IDS for row in test):
        raise RuntimeError("an inspected pair leaked into untouched test")
    all_ids = [str(row["pair_id"]) for group in (train, development, test) for row in group]
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("M0 splits overlap by pair ID")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    train_sha = write_jsonl(output / "m0_train_manifest.jsonl", [copy_with_split(row, "train") for row in train])
    dev_sha = write_jsonl(output / "m0_dev_manifest.jsonl", [copy_with_split(row, "val") for row in development])
    test_sha = write_jsonl(output / "m0_test_manifest.jsonl", [copy_with_split(row, "test") for row in test])
    audit = {
        "schema_version": "qcpr_v31_m0_split_v1",
        "source_manifest": str(source),
        "source_manifest_sha256": sha256_file(source),
        "chronology": "Image2(before)->Image1(after); Label1=appeared; Label2=disappeared",
        "pre_a936f116_mask_local_artifacts": "INVALID",
        "split_seed": SPLIT_SEED,
        "eligibility": {"tier": ["core", "hard"], "both_directional_masks_nonempty": True, "directional_overlap_iou_max": 0.10},
        "pair_counts": {"train": len(train), "development": len(development), "test": len(test)},
        "manifest_sha256": {"train": train_sha, "development": dev_sha, "test": test_sha},
        "pair_ids": {"train": [str(row["pair_id"]) for row in train], "development": [str(row["pair_id"]) for row in development], "test": [str(row["pair_id"]) for row in test]},
        "inspected_pair_ids_forced_to_development": sorted(base_id(row) for row in development if base_id(row) in INSPECTED_BASE_IDS),
        "leakage_check": "PASS",
    }
    (output / "m0_split_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

