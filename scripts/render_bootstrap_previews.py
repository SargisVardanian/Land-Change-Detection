from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render first preview images from bootstrap preview_samples.json.")
    parser.add_argument("--project-root", type=Path, required=True)
    return parser.parse_args()


def _render(index_path: Path, sample_id: str, output_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    subprocess.run(
        [
            sys.executable,
            "scripts/render_change_retrieval_sample.py",
            "--index",
            str(index_path),
            "--sample-id",
            sample_id,
            "--output",
            str(output_path),
        ],
        check=True,
        env=env,
    )


def main() -> int:
    args = parse_args()
    preview_manifest_path = args.project_root / "indexes" / "preview_samples.json"
    payload = json.loads(preview_manifest_path.read_text(encoding="utf-8"))
    runs_root = args.project_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    levir = payload.get("LEVIR-MCI")
    if levir:
        output = runs_root / "levir_mci_preview.png"
        _render(args.project_root / "indexes" / "levir_mci_samples.jsonl", levir["sample_id"], output)
        print(f"Rendered {output}")

    second = payload.get("SECOND-CC")
    if second:
        output = runs_root / "second_cc_preview.png"
        _render(args.project_root / "indexes" / "second_cc_samples.jsonl", second["sample_id"], output)
        print(f"Rendered {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
