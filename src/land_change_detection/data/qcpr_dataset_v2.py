"""Canonical, provenance-preserving Dataset-v2 utilities for QCPR.

Dense labels are stored in a sidecar registry and are never copied into
mask-free retrieval or grounding manifests.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "2.1"

_GENERIC_NO_CHANGE_RE = re.compile(
    r"^(?:there (?:is|are) no (?:significant )?(?:difference|change)s?|"
    r"no (?:significant |visible )?change(?: has occurred| occurred|s)?|"
    r"the (?:two )?(?:images|scenes?) (?:are|look|seem) (?:the same|identical)|"
    r"nothing (?:has )?changed|almost nothing has changed)$",
    re.I,
)
_MASK_PATH_KEY_RE = re.compile(
    r"(?:^|_)(?:mask|masks|semantic|target|targets|label|labels|ground_truth|gt)(?:_path|_paths)?$",
    re.I,
)
_SPATIAL_TERMS = {
    "left", "right", "upper", "lower", "top", "bottom", "center", "central",
    "north", "south", "east", "west", "corner", "edge", "along", "between",
    "adjacent", "parallel", "diagonal", "surrounding", "near",
}
_SCENE_TERMS = {
    "road", "roads", "river", "water", "field", "fields", "forest", "trees",
    "vegetation", "residential", "industrial", "parking", "bridge", "shore",
    "coast", "building", "buildings", "greenhouse", "playground", "runway",
}
_VAGUE_CHANGE_TERMS = {
    "new", "appeared", "disappeared", "change", "changed", "changes", "building",
    "buildings", "road", "roads", "area", "areas", "some", "several",
}


def normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w ]+", " ", str(text).casefold()).split())


def dataset_key(value: Any) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value or "unknown").casefold()).strip("_")
    return key or "unknown"


def canonical_pair_id_for(row: Mapping[str, Any]) -> str:
    source_id = str(row.get("pair_id") or row.get("source_pair_id") or "").strip()
    if not source_id:
        raise ValueError("pair row has no pair_id/source_pair_id")
    prefix = dataset_key(row.get("dataset_name") or row.get("source_dataset"))
    return source_id if source_id.startswith(f"{prefix}:") else f"{prefix}:{source_id}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: str | Path) -> str | None:
    """Return a deterministic 64-bit difference hash for near-duplicate audit."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            pixels = list(image.convert("L").resize((9, 8)).getdata())
        bits = []
        for y in range(8):
            row = pixels[y * 9 : (y + 1) * 9]
            bits.extend(left > right for left, right in zip(row, row[1:]))
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return f"{value:016x}"
    except Exception:
        return None


def jsonl_read(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def jsonl_write(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False))
    with target.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return sha256_file(target)


def source_scene_id(row: Mapping[str, Any]) -> str:
    meta = row.get("source_metadata") or {}
    raw = meta.get("source_scene_group_id") or meta.get("scene_id") or row.get("source_scene_group_id")
    raw = raw or row.get("pair_id") or row.get("source_pair_id")
    return f"{dataset_key(row.get('dataset_name') or row.get('source_dataset'))}:{raw}"


def change_status(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("change_status") or "").casefold()
    if explicit in {"changed", "change", "1", "true"}:
        return "changed"
    if explicit in {"no_change", "no-change", "unchanged", "0", "false"}:
        return "no_change"
    value = (row.get("source_metadata") or {}).get("changeflag")
    if value in (0, "0", False):
        return "no_change"
    if value in (1, "1", True):
        return "changed"
    return "unknown"


def is_generic_no_change(caption: str, status: str) -> bool:
    if status != "no_change":
        return False
    return bool(_GENERIC_NO_CHANGE_RE.fullmatch(normalize_text(caption)))


def caption_identifiability_score(caption: str, status: str) -> float:
    normalized = normalize_text(caption)
    if not normalized or is_generic_no_change(caption, status):
        return 0.0
    tokens = normalized.split()
    token_set = set(tokens)
    score = min(len(tokens) / 24.0, 0.35)
    if token_set & _SPATIAL_TERMS:
        score += 0.20
    if token_set & _SCENE_TERMS:
        score += 0.15
    if any(token.isdigit() for token in tokens) or token_set & {"one", "two", "three", "several", "multiple"}:
        score += 0.10
    if token_set & {"with", "while", "beside", "surrounded", "crossing", "layout", "density"}:
        score += 0.15
    if len(tokens) <= 7 and token_set <= _VAGUE_CHANGE_TERMS:
        score = min(score, 0.30)
    return round(min(score, 1.0), 4)


def query_scope(caption: str, status: str) -> str:
    if is_generic_no_change(caption, status):
        return "generic_no_change"
    return "exact_pair" if caption_identifiability_score(caption, status) >= 0.60 else "semantic_group"


def frame(path: str | None, timestamp: str, root: Path | None = None) -> dict[str, Any]:
    candidate = Path(path) if path else None
    if candidate and root and not candidate.is_absolute():
        candidate = root / candidate
    exists = bool(candidate and candidate.is_file())
    width = height = channels = None
    mode = None
    decode_error = None
    if exists:
        try:
            from PIL import Image

            with Image.open(candidate) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                channels = len(image.getbands())
        except Exception as exc:
            decode_error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(candidate) if candidate else None,
        "timestamp": timestamp,
        "modality": "rgb",
        "sensor": None,
        "exists": exists,
        "sha256": sha256_file(candidate) if exists else None,
        "perceptual_hash": perceptual_hash(candidate) if exists else None,
        "width": width,
        "height": height,
        "channels": channels,
        "mode": mode,
        "decode_error": decode_error,
    }


def _hash_parts(parts: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def split_role(split: Any) -> str:
    value = str(split or "unknown").casefold()
    if value in {"train", "training"}:
        return "train"
    if value in {"val", "validation", "dev", "development"}:
        return "development"
    if value in {"test", "testing"}:
        return "test"
    if value in {"", "unknown", "none"}:
        return "unknown"
    raise ValueError(f"unsupported split label: {split!r}")


def legacy_pair_to_v2(row: dict[str, Any], *, dataset_version: str = "existing") -> dict[str, Any]:
    canonical_id = canonical_pair_id_for(row)
    frames = [frame(row.get("t1_path"), "t1"), frame(row.get("t2_path"), "t2")]
    frame_ids = [str(item.get("sha256") or item.get("path") or "missing") for item in frames]
    source_pair_id = str(row.get("pair_id") or row.get("source_pair_id"))
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_pair_id": canonical_id,
        "source_dataset": str(row.get("dataset_name") or row.get("source_dataset") or "unknown"),
        "source_version": dataset_version,
        "source_pair_id": source_pair_id,
        "source_scene_group_id": source_scene_id(row),
        "frames": frames,
        "timestamps": [item["timestamp"] for item in frames],
        "modalities": [item["modality"] for item in frames],
        "sensor": None,
        "spatial_resolution": None,
        "height": frames[0].get("height"),
        "width": frames[0].get("width"),
        "is_synthetic": bool(row.get("is_synthetic", False)),
        "synthetic_generator": row.get("synthetic_generator"),
        "split": split_role(row.get("split", "unknown")),
        "license": row.get("license"),
        "ordered_pair_hash": _hash_parts(frame_ids),
        "order_invariant_pair_hash": _hash_parts(sorted(frame_ids)),
        "t1_path": row.get("t1_path"),
        "t2_path": row.get("t2_path"),
    }


def captions_for_pair(row: dict[str, Any]) -> list[dict[str, Any]]:
    status = change_status(row)
    pair_id = canonical_pair_id_for(row)
    result: list[dict[str, Any]] = []
    for index, text in enumerate(row.get("captions") or []):
        text = str(text).strip()
        if not text:
            continue
        scope = query_scope(text, status)
        normalized = normalize_text(text)
        group = "generic_no_change" if scope == "generic_no_change" else f"caption:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"
        result.append({
            "schema_version": SCHEMA_VERSION,
            "caption_id": f"{pair_id}:caption:{index}",
            "canonical_pair_id": pair_id,
            "text": text,
            "normalized_text": normalized,
            "caption_source": "human",
            "task_type": "temporal_retrieval",
            "query_scope": scope,
            "semantic_group_id": group,
            "equivalent_caption_group_id": group,
            "quality_score": 1.0,
            "identifiability_score": caption_identifiability_score(text, status),
            "language": "en",
            "is_generated": False,
            "generator": None,
            "verification_status": "human",
            "change_status": status,
            "dataset_name": row.get("dataset_name"),
        })
    return result


def build_relevance(captions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    captions = list(captions)
    by_normalized: dict[str, set[str]] = defaultdict(set)
    by_group: dict[str, set[str]] = defaultdict(set)
    for caption in captions:
        pair_id = str(caption["canonical_pair_id"])
        by_normalized[str(caption.get("normalized_text") or normalize_text(caption.get("text", "")))].add(pair_id)
        by_group[str(caption.get("semantic_group_id") or "")].add(pair_id)

    output = []
    for caption in captions:
        pair_id = str(caption["canonical_pair_id"])
        scope = str(caption.get("query_scope") or "semantic_group")
        normalized = str(caption.get("normalized_text") or normalize_text(caption.get("text", "")))
        group = str(caption.get("semantic_group_id") or "")
        collision_pairs = set(by_normalized[normalized]) - {pair_id}
        group_pairs = set(by_group[group]) if group else {pair_id}

        if scope == "generic_no_change":
            positives: list[str] = []
            ignored: list[str] = []
            policy = "semantic_group_only"
        elif scope == "semantic_group":
            positives = sorted(group_pairs or {pair_id})
            ignored = sorted(collision_pairs - set(positives))
            policy = "all_other_valid"
        elif scope == "instruction_only":
            positives = []
            ignored = []
            policy = "not_for_retrieval"
        else:
            positives = [pair_id]
            ignored = sorted((collision_pairs | (group_pairs - {pair_id})) - {pair_id})
            policy = "all_other_valid"

        output.append({
            "schema_version": SCHEMA_VERSION,
            "caption_id": caption["caption_id"],
            "canonical_pair_id": pair_id,
            "query_scope": scope,
            "semantic_group_id": group or None,
            "positive_pair_ids": positives,
            "ignored_pair_ids": ignored,
            "negative_policy": policy,
            "valid_negative_policy": policy,
        })
    return output


def _forbidden_dense_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if _MASK_PATH_KEY_RE.search(str(key)) or str(key) in {
                "binary_change_mask", "semantic_t1", "semantic_t2", "query_masks",
            }:
                found.append(path)
            found.extend(_forbidden_dense_keys(child, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(_forbidden_dense_keys(child, f"{prefix}[{index}]"))
    return found


def mask_free(row: dict[str, Any]) -> dict[str, Any]:
    forbidden = _forbidden_dense_keys(row)
    if forbidden:
        raise ValueError(f"dense-label keys forbidden in mask-free manifest: {forbidden}")
    return row


def _dense_label_for_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    meta = row.get("source_metadata") or {}
    values = {
        "binary_change_mask": row.get("binary_change_mask") or row.get("mask_path") or meta.get("mask_path"),
        "semantic_t1": row.get("semantic_t1") or meta.get("semantic_t1"),
        "semantic_t2": row.get("semantic_t2") or meta.get("semantic_t2"),
        "query_masks": row.get("query_masks") or meta.get("query_masks"),
    }
    if not any(values.values()):
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_pair_id": canonical_pair_id_for(row),
        **values,
        "label_source": row.get("dataset_name") or row.get("source_dataset"),
    }


def build_legacy_v2(manifests: Iterable[str | Path], output_root: str | Path) -> dict[str, Any]:
    rows = [row for manifest in manifests for row in jsonl_read(manifest)]
    return build_rows_v2(rows, output_root)


def build_rows_v2(rows: Iterable[dict[str, Any]], output_root: str | Path) -> dict[str, Any]:
    rows = list(rows)
    pairs: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair = legacy_pair_to_v2(row)
        pair_id = pair["canonical_pair_id"]
        previous = pairs.get(pair_id)
        if previous and previous["order_invariant_pair_hash"] != pair["order_invariant_pair_hash"]:
            raise ValueError(f"canonical pair collision for {pair_id}")
        pairs[pair_id] = pair
    captions = [caption for row in rows for caption in captions_for_pair(row)]
    relevance = build_relevance(captions)
    dense = [item for row in rows if (item := _dense_label_for_row(row)) is not None]
    root = Path(output_root)
    hashes = {
        "pair_registry": jsonl_write(root / "pair_registry.jsonl", pairs.values()),
        "caption_registry": jsonl_write(root / "caption_registry.jsonl", captions),
        "relevance_registry": jsonl_write(root / "relevance_registry.jsonl", relevance),
        "dense_label_registry": jsonl_write(root / "dense_label_registry.jsonl", dense),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "pairs": len(pairs),
        "captions": len(captions),
        "dense_labels": len(dense),
        "query_scopes": dict(Counter(str(item["query_scope"]) for item in captions)),
        "hashes": hashes,
    }


def _identity_keys(pair: Mapping[str, Any]) -> set[str]:
    keys = {f"scene:{pair['source_scene_group_id']}"}
    if pair.get("order_invariant_pair_hash"):
        keys.add(f"pair:{pair['order_invariant_pair_hash']}")
    for item in pair.get("frames", []):
        if item.get("sha256"):
            keys.add(f"frame:{item['sha256']}")
    return keys


def _components(pairs: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    parent = list(range(len(pairs)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    owner: dict[str, int] = {}
    for index, pair in enumerate(pairs):
        for key in _identity_keys(pair):
            if key in owner:
                union(index, owner[key])
            else:
                owner[key] = index
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(pairs)):
        grouped[find(index)].append(index)
    return list(grouped.values())


def exclude_cross_split_image_conflicts(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(rows)
    pairs = [legacy_pair_to_v2(row) for row in rows]
    priority = {"train": 0, "development": 1, "test": 2}
    excluded_ids: dict[str, set[str]] = defaultdict(set)
    conflict_components = 0
    for component in _components(pairs):
        roles = {split_role(pairs[index]["split"]) for index in component}
        if "unknown" in roles:
            raise ValueError("unknown split encountered during leakage resolution")
        if len(roles) <= 1:
            continue
        conflict_components += 1
        retained_role = max(roles, key=priority.__getitem__)
        for index in component:
            role = split_role(pairs[index]["split"])
            if role != retained_role:
                excluded_ids[role].add(pairs[index]["canonical_pair_id"])
    excluded_all = set().union(*excluded_ids.values()) if excluded_ids else set()
    kept = [row for row in rows if canonical_pair_id_for(row) not in excluded_all]
    return kept, {
        "policy": "preserve highest-priority held-out component; exclude lower-priority records",
        "priority": ["test", "development", "train"],
        "conflict_components_before": conflict_components,
        "excluded_pair_ids_by_source_role": {key: sorted(value) for key, value in sorted(excluded_ids.items())},
        "excluded_pairs": len(excluded_all),
        "kept_pairs": len(kept),
    }


def leakage_audit(pairs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    pairs = list(pairs)
    leaks = []
    for component in _components(pairs):
        roles = sorted({split_role(pairs[index]["split"]) for index in component})
        if len(roles) > 1:
            leaks.append({
                "splits": roles,
                "pair_ids": sorted(str(pairs[index]["canonical_pair_id"]) for index in component),
            })
    return {"passed": not leaks, "leak_components": leaks, "leak_component_count": len(leaks)}


def validate_relevance(
    pairs: Iterable[dict[str, Any]],
    captions: Iterable[dict[str, Any]],
    relevance: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    pairs = list(pairs)
    captions = list(captions)
    relevance = list(relevance)
    pair_ids = {str(item["canonical_pair_id"]) for item in pairs}
    captions_by_id = {str(item["caption_id"]): item for item in captions}
    violations: list[dict[str, Any]] = []
    for row in relevance:
        caption_id = str(row.get("caption_id"))
        caption = captions_by_id.get(caption_id)
        positives = set(map(str, row.get("positive_pair_ids") or []))
        ignored = set(map(str, row.get("ignored_pair_ids") or []))
        if caption is None:
            violations.append({"caption_id": caption_id, "reason": "missing_caption"})
            continue
        true_pair = str(caption["canonical_pair_id"])
        scope = str(caption.get("query_scope"))
        if true_pair not in pair_ids:
            violations.append({"caption_id": caption_id, "reason": "missing_true_pair"})
        if scope in {"exact_pair", "localized_query"} and true_pair not in positives:
            violations.append({"caption_id": caption_id, "reason": "true_pair_not_positive"})
        if true_pair in ignored:
            violations.append({"caption_id": caption_id, "reason": "true_pair_ignored"})
        if positives & ignored:
            violations.append({"caption_id": caption_id, "reason": "positive_ignored_overlap"})
        if not positives <= pair_ids or not ignored <= pair_ids:
            violations.append({"caption_id": caption_id, "reason": "unknown_referenced_pair"})
    return {"passed": not violations, "violations": violations, "checked": len(relevance)}


def inventory(paths: dict[str, Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in paths.items():
        exists = path.exists()
        files = [item for item in path.rglob("*") if item.is_file()] if exists else []
        result[name] = {
            "path": str(path),
            "exists": exists,
            "file_count": len(files),
            "bytes": sum(item.stat().st_size for item in files),
        }
    return result


def stable_manifest_hashes(root: str | Path) -> dict[str, str]:
    root = Path(root)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*.jsonl"))
        if path.is_file()
    }
