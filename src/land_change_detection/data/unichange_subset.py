from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from land_change_detection.levir_mci import discover_levir_mci_samples


REQUIRED_MCI_PATHS = (
    "LevirCCcaptions.json",
    "images/train/A",
    "images/train/B",
    "images/train/label",
)


@dataclass(frozen=True)
class MciRootResolution:
    root: Path
    source: str


def _has_official_mci_layout(root: Path) -> bool:
    return all((root / relative).exists() for relative in REQUIRED_MCI_PATHS)


def resolve_levir_mci_root(base: str | Path | None = None, override: str | Path | None = None) -> MciRootResolution:
    override = override or os.environ.get("MCI_ROOT")
    candidates: list[tuple[Path, str]] = []
    if override:
        candidates.append((Path(override), "MCI_ROOT"))
    if base is not None:
        base_path = Path(base)
        candidates.extend(
            [
                (base_path, "provided"),
                (base_path / "LEVIR-MCI-dataset", "provided_nested"),
            ]
        )
    for candidate, source in candidates:
        if _has_official_mci_layout(candidate):
            return MciRootResolution(candidate, source)
    checked = "\n".join(str(candidate) for candidate, _ in candidates) or "<none>"
    required = "\n".join(f"  - {path}" for path in REQUIRED_MCI_PATHS)
    raise FileNotFoundError(
        "Could not resolve official LEVIR-MCI root. Checked:\n"
        f"{checked}\nRequired structure under the selected root:\n{required}\n"
        "Set MCI_ROOT to the directory containing LevirCCcaptions.json and images/train/{A,B,label}."
    )


def current_git_commit(code_root: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_root, text=True).strip()
    except Exception:
        return "unknown"


def build_deterministic_mci_subset(root: str | Path, output_path: str | Path, count: int = 100, seed: int = 20260629, code_root: str | Path | None = None) -> dict[str, Any]:
    samples = discover_levir_mci_samples(root)
    if len(samples) < count:
        raise ValueError(f"LEVIR-MCI subset requires {count} samples, found {len(samples)} under {root}")
    ordered = sorted(samples, key=lambda sample: (sample.split, sample.sample_id))
    selected = ordered[:count]
    payload = {
        "dataset": "LEVIR-MCI",
        "root": str(root),
        "count": count,
        "seed": seed,
        "code_commit": current_git_commit(code_root),
        "items": [
            {
                "pair_id": sample.sample_id,
                "split": sample.split,
                "t1": sample.image_before,
                "t2": sample.image_after,
                "mask": sample.binary_change_mask,
                "caption_count": len(sample.captions),
            }
            for sample in selected
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_subset_pair_ids(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [str(item["pair_id"]) for item in payload.get("items", [])]
