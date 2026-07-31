#!/usr/bin/env python3
"""Resolve all ChangeChat records against retained LEVIR pairs with split safety."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else value.get("data", value.get("items", []))


def ids(row: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("pair_id", "image_id", "source_pair_id", "image", "image_name"):
        value = row.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            text = str(item or "").strip()
            if text:
                out.add(text)
                out.add(Path(text).stem)
    return out


def conversations(row: dict[str, Any]) -> tuple[str, str]:
    turns = row.get("conversations") or row.get("conversation") or []
    human = [str(t.get("value", "")) for t in turns if isinstance(t, dict) and str(t.get("from", "")).casefold() in {"human", "user"}]
    answer = [str(t.get("value", "")) for t in turns if isinstance(t, dict) and str(t.get("from", "")).casefold() in {"gpt", "assistant"}]
    return (human[-1] if human else "", answer[-1] if answer else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instructions", type=Path, required=True)
    ap.add_argument("--retained-pairs", type=Path, required=True)
    ap.add_argument("--legacy-pairs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--conflicts", type=Path, required=True)
    args = ap.parse_args()
    retained = [json.loads(x) for x in args.retained_pairs.open(encoding="utf-8") if x.strip()]
    retained_ids = {str(r["canonical_pair_id"]): r for r in retained}
    alias: dict[str, set[str]] = defaultdict(set)
    for canonical, row in retained_ids.items():
        source = str(row.get("source_pair_id") or "")
        for value in {canonical, source, source.rsplit(":", 1)[-1]}:
            if value:
                alias[value].add(canonical)
    legacy = [json.loads(x) for x in args.legacy_pairs.open(encoding="utf-8") if x.strip()]
    legacy_by_id = {str(r.get("sample_id") or r.get("pair_id")): r for r in legacy}
    conflict_ids = {f"levir_mci:{str(r.get('split','train'))}:{key}" for key, r in legacy_by_id.items() if f"levir_mci:{str(r.get('split','train'))}:{key}" not in retained_ids}
    mapped: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    counts = defaultdict(int)
    for ordinal, row in enumerate(read_json(args.instructions)):
        matches: set[str] = set()
        for value in ids(row):
            matches.update(alias.get(value, set()))
        image_names = sorted(v for v in ids(row) if v.endswith(".png"))
        if len(matches) == 1:
            canonical = next(iter(matches)); pair = retained_ids[canonical]
            prompt, answer = conversations(row)
            mapped.append({"instruction_id": f"changechat:{ordinal}", "canonical_pair_id": canonical, "instruction": prompt, "response": answer, "original_record": row, "split": pair.get("split"), "verification_status": "rule_based_unverified" if not row.get("gpt") else "generated_unverified"})
            counts["mapped"] += 1
            continue
        legacy_id = next((v for v in ids(row) if v.startswith("train_") or v.startswith("val_") or v.startswith("test_")), None)
        reason = "EXCLUDED_SPLIT_CONFLICT" if legacy_id and any(legacy_id in cid for cid in conflict_ids) else ("AMBIGUOUS" if len(matches) > 1 else "UNMAPPED")
        record = {"instruction_ordinal": ordinal, "legacy_pair_id": legacy_id, "image_names": image_names, "reason": reason, "candidate_canonical_ids": sorted(matches), "original_record": row}
        conflicts.append(record); counts[reason.lower()] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True); args.conflicts.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in mapped: f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    with args.conflicts.open("w", encoding="utf-8") as f:
        for row in conflicts: f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    report = {"counts": dict(counts), "input_records": len(mapped) + len(conflicts), "mapped": len(mapped), "excluded_split_conflict": counts.get("excluded_split_conflict", 0), "unmapped": counts.get("unmapped", 0), "ambiguous": counts.get("ambiguous", 0), "passed": counts.get("ambiguous", 0) == 0}
    args.output.with_name("changechat_full_mapping.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
