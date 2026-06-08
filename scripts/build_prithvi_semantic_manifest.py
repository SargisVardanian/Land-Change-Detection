from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Prithvi-style 6-band semantic manifest from semantic task manifests.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-ms-path-key", default="before_ms_path")
    parser.add_argument("--after-ms-path-key", default="after_ms_path")
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    rows = _read_rows(args.input_manifest)
    output_rows = []
    for row in rows:
        metadata = dict(row.get("metadata", {}))
        before_ms = metadata.get(args.before_ms_path_key)
        after_ms = metadata.get(args.after_ms_path_key)
        if not before_ms or not after_ms:
            continue
        output_rows.append(
            {
                **row,
                "before_ms_path": str(before_ms),
                "after_ms_path": str(after_ms),
                "input_contract": "prithvi_6band_from_13band",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows), encoding="utf-8")
    print(f"Wrote {len(output_rows)} Prithvi semantic rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
