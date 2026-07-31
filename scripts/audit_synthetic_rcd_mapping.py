#!/usr/bin/env python3
"""Audit Synthetic RCD original-A mapping without index-only assumptions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha(path: Path) -> str | None:
    if not path.is_file(): return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--rcd-root", type=Path, required=True); ap.add_argument("--original-root", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--pilot", type=int, default=500); args = ap.parse_args()
    train = args.rcd_root / "train"; entries = [x.strip() for x in (train / "train_A.txt").read_text().splitlines() if x.strip()]
    original_files = list(args.original_root.rglob("*.png")) if args.original_root.exists() else []
    by_name: dict[str, list[Path]] = {}
    for path in original_files: by_name.setdefault(path.name, []).append(path)
    mapping: list[dict[str, Any]] = []; missing = 0; ambiguous = 0
    for index, name in enumerate(entries):
        candidates = by_name.get(Path(name).name, [])
        if not candidates: missing += 1
        if len(candidates) > 1: ambiguous += 1
        mapping.append({"ordinal": index, "synthetic_a_name": name, "candidate_original_paths": [str(p) for p in candidates], "candidate_sha256": [sha(p) for p in candidates], "status": "mapped" if len(candidates) == 1 else ("missing" if not candidates else "ambiguous")})
    pilot = mapping[:args.pilot]
    report = {"input_entries": len(entries), "original_png_count": len(original_files), "mapping_coverage": (len(entries) - missing) / len(entries) if entries else 0.0, "missing_a_images": missing, "ambiguous_mappings": ambiguous, "pilot_entries": len(pilot), "real_a_ready": missing == 0 and ambiguous == 0, "mode": "SYNTHETIC_RCD_REAL_A" if missing == 0 and ambiguous == 0 else "SYNTHETIC_RCD_SYNTHETIC_A_ONLY", "blocker": "original SECOND assets unavailable" if not original_files else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_name("synthetic_rcd_second_mapping.jsonl").open("w", encoding="utf-8") as f:
        for row in mapping: f.write(json.dumps(row, sort_keys=True) + "\n")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["real_a_ready"] else 2


if __name__ == "__main__": raise SystemExit(main())
