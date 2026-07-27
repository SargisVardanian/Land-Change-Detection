from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
BEFORE_HINTS = ("before", "pre", "img1", "image1", "a", "t1")
AFTER_HINTS = ("after", "post", "img2", "image2", "b", "t2")
MASK_HINTS = ("mask", "label", "change", "gt", "seg", "cm")
SEMANTIC_BEFORE_HINTS = ("semanticbefore", "beforelabel", "t1label", "a_label", "label1", "sem1")
SEMANTIC_AFTER_HINTS = ("semanticafter", "afterlabel", "t2label", "b_label", "label2", "sem2")


@dataclass(frozen=True)
class ChangeRetrievalSample:
    sample_id: str
    dataset_name: str
    before_path: str
    after_path: str
    mask_path: str | None = None
    semantic_before_path: str | None = None
    semantic_after_path: str | None = None
    caption: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChangeRetrievalSample":
        return cls(
            sample_id=str(payload["sample_id"]),
            dataset_name=str(payload["dataset_name"]),
            before_path=str(payload["before_path"]),
            after_path=str(payload["after_path"]),
            mask_path=str(payload["mask_path"]) if payload.get("mask_path") else None,
            semantic_before_path=str(payload["semantic_before_path"]) if payload.get("semantic_before_path") else None,
            semantic_after_path=str(payload["semantic_after_path"]) if payload.get("semantic_after_path") else None,
            caption=str(payload["caption"]) if payload.get("caption") is not None else None,
            split=str(payload["split"]) if payload.get("split") is not None else None,
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "before_path": self.before_path,
            "after_path": self.after_path,
            "mask_path": self.mask_path,
            "semantic_before_path": self.semantic_before_path,
            "semantic_after_path": self.semantic_after_path,
            "caption": self.caption,
            "split": self.split,
            "metadata": dict(self.metadata),
        }


def load_change_samples(path: str | Path) -> list[ChangeRetrievalSample]:
    rows: list[ChangeRetrievalSample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(ChangeRetrievalSample.from_dict(json.loads(line)))
    return rows


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _file_key(path: Path) -> str:
    name = _normalize_key(path.stem)
    for suffix in ("before", "after", "pre", "post", "img1", "img2", "image1", "image2", "mask", "label", "change", "gt", "seg", "cm", "a", "b", "t1", "t2"):
        if name.endswith(suffix):
            trimmed = name[: -len(suffix)]
            if trimmed:
                return trimmed
    return name


def _contains_hint(path: Path, hints: tuple[str, ...]) -> bool:
    name = _normalize_key(path.stem)
    parent = _normalize_key(path.parent.name)
    return any(hint in name or hint == name or hint in parent for hint in hints)


def _sorted_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _load_caption_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if all(isinstance(value, str) for value in payload.values()):
            return {str(key): str(value) for key, value in payload.items()}
        if "images" in payload and isinstance(payload["images"], list):
            mapping: dict[str, str] = {}
            for row in payload["images"]:
                if not isinstance(row, dict):
                    continue
                key = row.get("id") or row.get("image_id") or row.get("sample_id") or row.get("filename")
                caption = row.get("caption") or row.get("text") or row.get("change_caption")
                if key is not None and caption:
                    mapping[str(key)] = str(caption)
            return mapping
    if isinstance(payload, list):
        mapping = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            key = row.get("id") or row.get("image_id") or row.get("sample_id") or row.get("filename")
            caption = row.get("caption") or row.get("text") or row.get("change_caption")
            if key is not None and caption:
                mapping[str(key)] = str(caption)
        return mapping
    return {}


def find_caption_sources(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if "caption" in path.name.lower() or "label" in path.name.lower() or "anno" in path.name.lower())


def discover_change_samples(root: str | Path, dataset_name: str) -> list[ChangeRetrievalSample]:
    root_path = Path(root)
    images = _sorted_images(root_path)
    before_by_key: dict[str, Path] = {}
    after_by_key: dict[str, Path] = {}
    mask_by_key: dict[str, Path] = {}
    semantic_before_by_key: dict[str, Path] = {}
    semantic_after_by_key: dict[str, Path] = {}
    for path in images:
        key = _file_key(path)
        if _contains_hint(path, BEFORE_HINTS) and key not in before_by_key:
            before_by_key[key] = path
        elif _contains_hint(path, AFTER_HINTS) and key not in after_by_key:
            after_by_key[key] = path
        elif _contains_hint(path, SEMANTIC_BEFORE_HINTS) and key not in semantic_before_by_key:
            semantic_before_by_key[key] = path
        elif _contains_hint(path, SEMANTIC_AFTER_HINTS) and key not in semantic_after_by_key:
            semantic_after_by_key[key] = path
        elif _contains_hint(path, MASK_HINTS) and key not in mask_by_key:
            mask_by_key[key] = path

    caption_map: dict[str, str] = {}
    for source in find_caption_sources(root_path):
        caption_map.update(_load_caption_map(source))

    samples: list[ChangeRetrievalSample] = []
    shared_keys = sorted(set(before_by_key) & set(after_by_key))
    for key in shared_keys:
        before_path = before_by_key[key]
        after_path = after_by_key[key]
        split = _infer_split(root_path, before_path, after_path)
        caption = caption_map.get(key) or caption_map.get(before_path.name) or caption_map.get(after_path.name)
        sample_id = key or before_path.stem
        samples.append(
            ChangeRetrievalSample(
                sample_id=sample_id,
                dataset_name=dataset_name,
                before_path=str(before_path),
                after_path=str(after_path),
                mask_path=str(mask_by_key[key]) if key in mask_by_key else None,
                semantic_before_path=str(semantic_before_by_key[key]) if key in semantic_before_by_key else None,
                semantic_after_path=str(semantic_after_by_key[key]) if key in semantic_after_by_key else None,
                caption=caption,
                split=split,
                metadata={"source_root": str(root_path)},
            )
        )
    return samples


def _infer_split(root: Path, before_path: Path, after_path: Path) -> str | None:
    for part in before_path.relative_to(root).parts + after_path.relative_to(root).parts:
        lowered = part.lower()
        if lowered in {"train", "val", "valid", "validation", "test"}:
            return "val" if lowered in {"valid", "validation"} else lowered
    return None


class IndexedChangeDataset:
    def __init__(self, samples: list[ChangeRetrievalSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        from PIL import Image

        sample = self.samples[index]
        before = np.array(Image.open(sample.before_path).convert("RGB"))
        after = np.array(Image.open(sample.after_path).convert("RGB"))
        mask = None
        if sample.mask_path:
            mask = np.array(Image.open(sample.mask_path))
        return {
            "sample_id": sample.sample_id,
            "dataset_name": sample.dataset_name,
            "before": before,
            "after": after,
            "mask": mask,
            "semantic_before_path": sample.semantic_before_path,
            "semantic_after_path": sample.semantic_after_path,
            "caption": sample.caption,
            "split": sample.split,
            "metadata": sample.metadata,
        }

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class LEVIRMciDataset(IndexedChangeDataset):
    @classmethod
    def from_root(cls, root: str | Path) -> "LEVIRMciDataset":
        return cls(discover_change_samples(root, "LEVIR-MCI"))


class SecondCCDataset(IndexedChangeDataset):
    @classmethod
    def from_root(cls, root: str | Path) -> "SecondCCDataset":
        return cls(discover_change_samples(root, "SECOND-CC"))
