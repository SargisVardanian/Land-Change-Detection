from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


OFFICIAL_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class LevirMciSample:
    sample_id: str
    split: str
    image_before: str
    image_after: str
    binary_change_mask: str
    caption: str
    captions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset_name": "LEVIR-MCI",
            "before_path": self.image_before,
            "after_path": self.image_after,
            "mask_path": self.binary_change_mask,
            "caption": self.caption,
            "split": self.split,
            "metadata": {
                **self.metadata,
                "captions": list(self.captions),
                "image_before": self.image_before,
                "image_after": self.image_after,
                "binary_change_mask": self.binary_change_mask,
            },
        }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_captions(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        for key in ("caption", "captions", "text", "texts", "sentences", "description", "descriptions"):
            if key in value:
                return _extract_captions(value[key])
        collected: list[str] = []
        for nested in value.values():
            collected.extend(_extract_captions(nested))
        return collected
    if isinstance(value, list):
        collected = []
        for item in value:
            collected.extend(_extract_captions(item))
        return collected
    return []


def load_caption_map(dataset_root: str | Path) -> dict[str, list[str]]:
    root = Path(dataset_root)
    candidates = sorted(root.glob("*.json"))
    caption_map: dict[str, list[str]] = {}
    for path in candidates:
        payload = _load_json(path)
        if isinstance(payload, dict):
            if all(isinstance(key, str) for key in payload):
                for key, value in payload.items():
                    captions = _extract_captions(value)
                    if captions:
                        caption_map[str(key)] = captions
            for key in ("images", "items", "annotations", "samples", "data"):
                rows = payload.get(key)
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    sample_id = row.get("sample_id") or row.get("id") or row.get("filename") or row.get("image_id")
                    if sample_id is None:
                        continue
                    captions = _extract_captions(row)
                    if captions:
                        caption_map[str(sample_id)] = captions
        elif isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                sample_id = row.get("sample_id") or row.get("id") or row.get("filename") or row.get("image_id")
                if sample_id is None:
                    continue
                captions = _extract_captions(row)
                if captions:
                    caption_map[str(sample_id)] = captions
    return caption_map


def _official_split_roots(dataset_root: Path) -> dict[str, tuple[Path, Path, Path]]:
    images_root = dataset_root / "images"
    split_roots: dict[str, tuple[Path, Path, Path]] = {}
    for split in OFFICIAL_SPLITS:
        before_root = images_root / split / "A"
        after_root = images_root / split / "B"
        mask_root = images_root / split / "label"
        if before_root.exists() and after_root.exists() and mask_root.exists():
            split_roots[split] = (before_root, after_root, mask_root)
    return split_roots


def discover_levir_mci_samples(dataset_root: str | Path) -> list[LevirMciSample]:
    root = Path(dataset_root)
    caption_map = load_caption_map(root)
    samples: list[LevirMciSample] = []

    for split, (before_root, after_root, mask_root) in _official_split_roots(root).items():
        for before_path in sorted(before_root.iterdir()):
            if not before_path.is_file():
                continue
            after_path = after_root / before_path.name
            mask_path = mask_root / before_path.name
            if not after_path.exists() or not mask_path.exists():
                continue
            sample_id = before_path.stem
            captions = caption_map.get(sample_id) or caption_map.get(before_path.name) or [""]
            samples.append(
                LevirMciSample(
                    sample_id=sample_id,
                    split=split,
                    image_before=str(before_path),
                    image_after=str(after_path),
                    binary_change_mask=str(mask_path),
                    caption=captions[0] if captions else "",
                    captions=tuple(captions),
                    metadata={"source_root": str(root), "official_layout": True},
                )
            )

    if samples:
        return samples

    # Fallback to the repo's generic discovery logic for locally restructured copies.
    from land_change_detection.change_retrieval_datasets import discover_change_samples

    generic = discover_change_samples(root, "LEVIR-MCI")
    return [
        LevirMciSample(
            sample_id=sample.sample_id,
            split=sample.split or "unknown",
            image_before=sample.before_path,
            image_after=sample.after_path,
            binary_change_mask=sample.mask_path or "",
            caption=sample.caption or "",
            captions=tuple([sample.caption] if sample.caption else []),
            metadata=sample.metadata,
        )
        for sample in generic
        if sample.mask_path
    ]
