from __future__ import annotations

import json
import os
import random
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
    candidates: list[tuple[Path, str]] = []
    if override is not None:
        candidates.append((Path(override), "override"))
    if base is not None:
        base_path = Path(base)
        candidates.extend(
            [
                (base_path, "provided"),
                (base_path / "LEVIR-MCI-dataset", "provided_nested"),
            ]
        )
    environment_root = os.environ.get("MCI_ROOT")
    if environment_root:
        candidates.append((Path(environment_root), "MCI_ROOT"))
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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def validate_mci_subset(path: str | Path, expected_count: int = 100, expected_split: str = "train") -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("items", [])
    ids = [str(item.get("pair_id")) for item in items]
    if len(items) != expected_count:
        raise ValueError(f"Subset {path} has {len(items)} items; expected {expected_count}.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Subset {path} contains duplicate pair IDs.")
    for item in items:
        if item.get("split") != expected_split:
            raise ValueError(f"Subset {path} contains split {item.get('split')!r}; expected {expected_split!r}.")
        if int(item.get("caption_count", 0)) <= 0:
            raise ValueError(f"Subset item {item.get('pair_id')} has no captions.")
        for key in ("t1", "t2", "mask"):
            if not Path(str(item.get(key, ""))).exists():
                raise FileNotFoundError(f"Subset item {item.get('pair_id')} references missing {key}: {item.get(key)}")
    return payload


def build_deterministic_mci_subset(
    root: str | Path,
    output_path: str | Path,
    count: int = 100,
    seed: int = 20260629,
    split: str = "train",
    code_root: str | Path | None = None,
) -> dict[str, Any]:
    samples = [sample for sample in discover_levir_mci_samples(root) if sample.split == split]
    if len(samples) < count:
        raise ValueError(f"LEVIR-MCI subset requires {count} {split!r} samples, found {len(samples)} under {root}")
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    random.Random(seed).shuffle(ordered)
    selected = ordered[:count]
    payload = {
        "dataset": "LEVIR-MCI",
        "root": str(root),
        "count": count,
        "split": split,
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
