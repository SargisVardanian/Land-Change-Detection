from __future__ import annotations

import argparse
import json
from pathlib import Path


def _infer_split(path: Path) -> str:
    normalized = [part.lower() for part in path.parts]
    if "train" in normalized:
        return "train"
    if "val" in normalized or "valid" in normalized or "validation" in normalized:
        return "val"
    if "test" in normalized:
        return "test"
    return "unknown"


def _discover_levir_cc_pairs(root: Path) -> list[dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        lowered = path.stem.lower()
        key = lowered
        if lowered.endswith(("_before", "_after", "_a", "_b", "_t1", "_t2")):
            key = lowered.rsplit("_", 1)[0]
        row = groups.setdefault(key, {})
        full = str(path)
        if lowered.endswith(("_before", "_a", "_t1")):
            row["before_path"] = full
        elif lowered.endswith(("_after", "_b", "_t2")):
            row["after_path"] = full
    rows = []
    for key, value in groups.items():
        if {"before_path", "after_path"} <= set(value):
            before_path = Path(value["before_path"])
            after_path = Path(value["after_path"])
            split = _infer_split(before_path)
            other_split = _infer_split(after_path)
            if split != other_split and other_split != "unknown":
                raise SystemExit(f"Split mismatch for {key}: before={split}, after={other_split}")
            rows.append(dict(sample_id=key, split=split if split != "unknown" else other_split, **value))
    return rows


def _caption_map(root: Path) -> dict[str, list[dict[str, str]]]:
    mapping: dict[str, list[dict[str, str]]] = {}
    for source in sorted(root.rglob("*.json")):
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("images", []) if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            sample_id = str(row.get("id") or row.get("image_id") or row.get("sample_id") or row.get("filename") or "")
            if sample_id:
                mapping.setdefault(sample_id.lower(), []).append(
                    {
                        "caption": str(row.get("caption") or row.get("text") or row.get("change_caption") or ""),
                        "transition_label": str(
                            row.get("transition_label") or row.get("class") or row.get("change_type") or sample_id
                        ),
                    }
                )
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LEVIR-CC pair manifest with before/after image paths and captions.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = _discover_levir_cc_pairs(args.root)
    if not pairs:
        raise SystemExit(f"No LEVIR-CC before/after pairs discovered under {args.root}")
    caption_map = _caption_map(args.root)
    rows = []
    split_to_pair_ids: dict[str, set[str]] = {}
    for sample in pairs:
        pair_id = str(sample["sample_id"])
        split = str(sample.get("split") or "unknown")
        split_to_pair_ids.setdefault(split, set()).add(pair_id)
        caption_entries = caption_map.get(pair_id.lower()) or [{"caption": "", "transition_label": pair_id}]
        for caption_index, caption_entry in enumerate(caption_entries):
            row_sample_id = pair_id if len(caption_entries) == 1 else f"{pair_id}#cap{caption_index:03d}"
            rows.append(
                {
                    "sample_id": row_sample_id,
                    "pair_id": pair_id,
                    "dataset_name": "LEVIR-CC",
                    "before_path": sample["before_path"],
                    "after_path": sample["after_path"],
                    "caption": caption_entry.get("caption", ""),
                    "split": split,
                    "metadata": {
                        "source_root": str(args.root),
                        "transition_label": caption_entry.get("transition_label", pair_id),
                        "retrieval_role": "text_to_pair_retrieval",
                        "curriculum_stage": "stage_1_text_to_pair",
                    },
                }
            )
    non_unknown = {split: pair_ids for split, pair_ids in split_to_pair_ids.items() if split != "unknown"}
    pair_ids = set().union(*non_unknown.values()) if non_unknown else set()
    pair_id_count = sum(len(pair_ids_for_split) for pair_ids_for_split in non_unknown.values())
    if pair_id_count != len(pair_ids):
        raise SystemExit("Detected LEVIR-CC pair_id leakage across train/val/test splits.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "caption_row_count": len(rows),
                "unique_pair_count": len({row["pair_id"] for row in rows}),
                "splits": {split: len(pair_ids_for_split) for split, pair_ids_for_split in split_to_pair_ids.items()},
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
