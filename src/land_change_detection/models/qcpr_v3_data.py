from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.utils.data import Sampler

import numpy as np
from PIL import Image

from land_change_detection.models.retrieval_heads import classify_caption_semantics


def normalize_caption(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold()).strip(" .,!?:;")


def _direction(semantics: dict[str, Any]) -> str:
    if semantics["no_change"]:
        return "no_change"
    if semantics["appeared"] or semantics["constructed"] or semantics["added"]:
        return "appeared"
    if semantics["disappeared"] or semantics["demolished"] or semantics["removed"]:
        return "disappeared"
    if semantics["increased"] or semantics["expanded"]:
        return "increased"
    if semantics["decreased"] or semantics["reduced"]:
        return "decreased"
    return "changed" if semantics["changed"] else "unknown"


def _relation(text: str) -> str:
    normalized = normalize_caption(text)
    return "replacement" if any(term in normalized for term in ("replace", "in place of", "converted to", "turned into")) else "none"


@lru_cache(maxsize=None)
def _mask_stats(path: str | None) -> tuple[bool | None, float | None]:
    if not path or not Path(path).exists():
        return None, None
    with Image.open(path) as image:
        array = np.asarray(image.convert("L")) > 0
    return not bool(array.any()), float(array.mean())


@lru_cache(maxsize=None)
def _dhash(path: str | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return f"{int(''.join('1' if value else '0' for value in bits.flat), 2):016x}"


def _quality(row: dict[str, Any]) -> float:
    source = str(row.get("caption_source", "unknown"))
    return {"human": 1.0, "semantic_template": 0.7, "template": 0.65}.get(source, 0.6)


def _audit_caption(row: dict[str, Any], caption_index: int, caption: str) -> dict[str, Any]:
    semantics = classify_caption_semantics(caption)
    mask_path = row.get("query_mask_path") or row.get("mask_path")
    mask_empty, mask_area = _mask_stats(mask_path)
    objects = sorted(set(semantics["object_terms"]))
    locations = sorted(set(semantics["location_terms"]))
    counts = sorted(set(semantics["count_terms"]))
    confidence = _quality(row)
    parser_known = bool(objects or semantics["no_change"] or semantics["changed"])
    return {
        "dataset": row["dataset_name"],
        "split": row["split"],
        "pair_id": row["pair_id"],
        "caption_id": f"{row['pair_id']}:{caption_index}",
        "caption": caption,
        "normalized_caption": normalize_caption(caption),
        "caption_source": row.get("caption_source"),
        "retrieval_supervision": bool(row.get("retrieval_supervision", True)),
        "localization_supervision_type": row.get("seg_supervision_mode", "generic" if mask_path else "none"),
        "change_status": "no_change" if semantics["no_change"] else "changed" if semantics["changed"] else "unknown",
        "temporal_direction": _direction(semantics),
        "object_family_audit_tags": objects,
        "location_audit_tags": locations,
        "count_audit_tags": counts,
        "relation_audit_tags": [_relation(caption)],
        "mask_empty": mask_empty,
        "foreground_area": mask_area,
        "label_confidence": confidence,
        "parser_confidence": 0.8 if parser_known else 0.35,
    }


def _cell(audit: dict[str, Any]) -> tuple[Any, ...]:
    return (
        audit["dataset"], audit["change_status"], audit["temporal_direction"],
        tuple(audit["object_family_audit_tags"]) or ("unknown",),
        tuple(audit["location_audit_tags"]) or ("unknown",),
        tuple(audit["count_audit_tags"]) or ("unknown",),
        tuple(audit["relation_audit_tags"]),
        audit["localization_supervision_type"],
    )


@dataclass(frozen=True)
class WeightingConfig:
    alpha: float = 0.4
    tau: float = 5.0
    min_weight: float = 0.2
    max_weight: float = 5.0
    natural_fraction: float = 0.5


class CappedCompositionalBatchSampler(Sampler[list[int]]):
    """Deterministic pair-level sampler with a strict no-change batch cap.

    Inputs are manifest rows, so a pair occurs at most once in a batch and its
    five caption paraphrases cannot be emitted as independent examples.
    """
    def __init__(self, rows: list[dict[str, Any]], batch_size: int, *, seed: int = 0, no_change_fraction_cap: float = 0.25):
        if batch_size <= 0 or not 0.0 <= no_change_fraction_cap <= 1.0:
            raise ValueError("positive batch_size and no_change_fraction_cap in [0,1] required")
        self.rows, self.batch_size, self.seed = rows, int(batch_size), int(seed)
        self.no_change_cap = (
            max(1, int(math.floor(self.batch_size * no_change_fraction_cap)))
            if no_change_fraction_cap > 0
            else 0
        )
        self.no_change = [index for index, row in enumerate(rows) if any(_direction(classify_caption_semantics(caption)) == "no_change" for caption in row.get("captions", []))]
        self.changed = [index for index in range(len(rows)) if index not in set(self.no_change)]

    def __len__(self) -> int:
        total_batches = math.ceil(len(self.rows) / self.batch_size)
        if self.no_change:
            if self.no_change_cap == 0:
                raise RuntimeError("no_change_fraction_cap=0 is incompatible with no-change training rows")
            total_batches = max(total_batches, math.ceil(len(self.no_change) / self.no_change_cap))
        return total_batches

    def _weighted_order(self, indices: list[int], generator: torch.Generator) -> list[int]:
        if not indices:
            return []
        weights = torch.tensor([max(float(self.rows[index].get("sampling_weight", 1.0)), 1e-8) for index in indices])
        # Gumbel-top-k is deterministic for a generator and samples without replacement.
        keys = torch.log(weights) - torch.log(-torch.log(torch.rand(len(indices), generator=generator).clamp_min(1e-8)))
        return [indices[index] for index in keys.argsort(descending=True).tolist()]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        changed, no_change = self._weighted_order(self.changed, generator), self._weighted_order(self.no_change, generator)
        ci = ni = 0
        while ci < len(changed) or ni < len(no_change):
            batch: list[int] = []
            while ni < len(no_change) and len(batch) < self.batch_size and len(batch) < self.no_change_cap:
                batch.append(no_change[ni]); ni += 1
            while ci < len(changed) and len(batch) < self.batch_size:
                batch.append(changed[ci]); ci += 1
            if ni < len(no_change) and len(batch) < self.batch_size:
                if not changed:
                    raise RuntimeError("cannot enforce a no-change cap when every training pair is no-change")
                # Reuse changed pairs across batches only to fill a capped tail; never
                # duplicate a pair within one batch.
                refill = 0
                while len(batch) < self.batch_size:
                    candidate = changed[refill % len(changed)]
                    refill += 1
                    if candidate not in batch:
                        batch.append(candidate)
            if ci >= len(changed) and ni < len(no_change) and self.no_change_cap == 0:
                raise RuntimeError("no-change examples remain but no_change_fraction_cap permits none")
            if batch:
                yield batch


class DirectionalCurriculumBatchSampler(Sampler[list[tuple[int, str]]]):
    """Deterministic 60/20/20 pair-unique mask curriculum.

    Each yielded tuple is ``(row_index, crop_mode)``.  Stress rows are never
    sampled.  The dataset interprets ``hard_background`` as a same-scene
    least-foreground native crop and ``hard_directional`` as a positive crop
    from the audited hard tier.
    """

    def __init__(self, rows: list[dict[str, Any]], batch_size: int, *, seed: int = 0):
        if batch_size < 5:
            raise ValueError("directional curriculum requires batch_size >= 5")
        self.rows, self.batch_size, self.seed = rows, int(batch_size), int(seed)
        self.core = [i for i, row in enumerate(rows) if row.get("quality_tier") == "core"]
        self.hard = [i for i, row in enumerate(rows) if row.get("quality_tier") == "hard"]
        if not self.core or not self.hard:
            raise ValueError("directional curriculum requires non-empty core and hard pools")
        self.positive_count = max(1, round(self.batch_size * 0.60))
        self.background_count = max(1, round(self.batch_size * 0.20))
        self.directional_count = self.batch_size - self.positive_count - self.background_count
        if self.directional_count <= 0:
            raise ValueError("batch size cannot realize a positive directional-hard quota")

    def __len__(self) -> int:
        return math.ceil((len(self.core) + len(self.hard)) / self.batch_size)

    @staticmethod
    def _draw(pool: list[int], count: int, cursor: int, used: set[int]) -> tuple[list[int], int]:
        selected: list[int] = []
        attempts = 0
        while len(selected) < count and attempts < max(len(pool) * 2, count * 4):
            candidate = pool[cursor % len(pool)]; cursor += 1; attempts += 1
            if candidate not in used:
                selected.append(candidate); used.add(candidate)
        if len(selected) != count:
            raise RuntimeError("insufficient unique pairs to satisfy directional curriculum batch")
        return selected, cursor

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        core = [self.core[i] for i in torch.randperm(len(self.core), generator=generator).tolist()]
        hard = [self.hard[i] for i in torch.randperm(len(self.hard), generator=generator).tolist()]
        core_cursor = hard_cursor = background_cursor = 0
        background_pool = core + hard
        for _ in range(len(self)):
            used: set[int] = set()
            positive, core_cursor = self._draw(core, self.positive_count, core_cursor, used)
            background, background_cursor = self._draw(
                background_pool, self.background_count, background_cursor, used
            )
            directional, hard_cursor = self._draw(hard, self.directional_count, hard_cursor, used)
            yield (
                [(index, "positive") for index in positive]
                + [(index, "hard_background") for index in background]
                + [(index, "hard_directional") for index in directional]
            )


class DirectionalM0BatchSampler(Sampler[list[tuple[int, str, int]]]):
    """Deterministic 70/30 M0 crop sampler over immutable pair-level records."""

    def __init__(self, rows: list[dict[str, Any]], batch_size: int, *, seed: int = 0, start_step: int = 0):
        if batch_size < 2:
            raise ValueError("M0 directional sampler requires batch_size >= 2")
        eligible = [
            index for index, row in enumerate(rows)
            if row.get("quality_tier") in {"core", "hard"}
            and row.get("dataset_name") == "s2looking"
            and bool(row.get("directional_targets"))
        ]
        if len(eligible) < batch_size:
            raise ValueError("M0 sampler has insufficient non-stress pair-level rows")
        self.rows, self.eligible = rows, eligible
        self.batch_size, self.seed, self.start_step = int(batch_size), int(seed), int(start_step)
        self.positive_count = max(1, int(round(self.batch_size * 0.70)))
        self.background_count = self.batch_size - self.positive_count
        if self.background_count <= 0:
            raise ValueError("M0 sampler must allocate background crops")

    def __len__(self) -> int:
        return math.ceil(len(self.eligible) / self.batch_size)

    def __iter__(self):
        for local_batch_index in range(len(self)):
            absolute_step = self.start_step + local_batch_index
            generator = torch.Generator().manual_seed(self.seed + absolute_step)
            order = [self.eligible[index] for index in torch.randperm(len(self.eligible), generator=generator).tolist()]
            selected = order[:self.batch_size]
            modes = ["positive"] * self.positive_count + ["hard_background"] * self.background_count
            yield [
                (index, mode, self.seed + absolute_step * self.batch_size + offset)
                for offset, (index, mode) in enumerate(zip(selected, modes, strict=True))
            ]


class DirectionalSanitySampler(Sampler[list[tuple[int, str, int]]]):
    """One-pair schedule: 60% jittered positive, 20% background, 20% reversal."""

    def __init__(self, *, row_index: int = 0, steps: int, seed: int):
        if row_index < 0:
            raise ValueError("directional sanity row index must be non-negative")
        if steps <= 0:
            raise ValueError("directional sanity steps must be positive")
        self.row_index, self.steps, self.seed = int(row_index), int(steps), int(seed)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        modes = ("positive", "positive", "positive", "hard_background", "temporal_reversal")
        generator = torch.Generator().manual_seed(self.seed)
        offset = int(torch.randint(0, len(modes), (), generator=generator))
        for step in range(self.steps):
            mode = modes[(step + offset) % len(modes)]
            yield [(self.row_index, mode, self.seed + step)]


def derive_qcpr_v3_manifests(
    manifest_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    config: WeightingConfig | None = None,
    balanced_validation_per_cell: int = 20,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    config = config or WeightingConfig()
    _mask_stats.cache_clear()
    _dhash.cache_clear()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for raw_path in manifest_paths:
        path = Path(raw_path)
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    pair_ids = Counter(str(row["pair_id"]) for row in rows)
    pair_splits = {str(row["pair_id"]): str(row["split"]) for row in rows}
    caption_audits: list[dict[str, Any]] = []
    normalized_clusters: defaultdict[str, list[str]] = defaultdict(list)
    pair_hash_clusters: defaultdict[str, list[str]] = defaultdict(list)
    for row_index, row in enumerate(rows, start=1):
        t1_hash, t2_hash = _dhash(row.get("t1_path")), _dhash(row.get("t2_path"))
        pair_hash = f"{t1_hash}:{t2_hash}" if t1_hash and t2_hash else "unavailable"
        pair_hash_clusters[pair_hash].append(str(row["pair_id"]))
        for index, caption in enumerate(row.get("captions", [])):
            audit = _audit_caption(row, index, caption)
            audit["near_duplicate_pair_cluster"] = pair_hash
            normalized_clusters[audit["normalized_caption"]].append(audit["caption_id"])
            caption_audits.append(audit)
        if progress_callback is not None:
            progress_callback(row_index, len(rows))
    for audit in caption_audits:
        audit["duplicate_caption_cluster"] = hashlib.sha1(audit["normalized_caption"].encode()).hexdigest()[:16]

    train_audits = [audit for audit in caption_audits if audit["split"] == "train"]
    cell_counts = Counter(_cell(audit) for audit in train_audits)
    raw_weights: dict[str, float] = {}
    for audit in train_audits:
        raw = audit["label_confidence"] * (cell_counts[_cell(audit)] + config.tau) ** (-config.alpha)
        raw_weights[audit["caption_id"]] = min(config.max_weight, max(config.min_weight, raw))
    normalizer = sum(raw_weights.values()) / max(len(raw_weights), 1)
    normalized_weights = {key: value / max(normalizer, 1e-8) for key, value in raw_weights.items()}
    caption_count_by_pair = Counter(audit["pair_id"] for audit in train_audits)

    by_pair: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit in caption_audits:
        by_pair[audit["pair_id"]].append(audit)
    natural_train: list[dict[str, Any]] = []
    balanced_train: list[dict[str, Any]] = []
    natural_validation: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        audits = by_pair[str(row["pair_id"])]
        if row["split"] == "train":
            pair_weight = sum(normalized_weights[audit["caption_id"]] for audit in audits) / max(len(audits), 1)
            pair_weight /= max(caption_count_by_pair[str(row["pair_id"])], 1)
            natural = item | {"sampling_weight": 1.0 / max(caption_count_by_pair[str(row["pair_id"])], 1), "sampling_policy": "natural_duplicate_aware"}
            balanced = item | {"sampling_weight": pair_weight, "sampling_policy": "smoothed_capped_compositional"}
            natural_train.append(natural)
            balanced_train.append(balanced)
        elif row["split"] == "val":
            natural_validation.append(item)

    validation_by_cell: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in natural_validation:
        audits = by_pair[str(row["pair_id"])]
        key = min((_cell(audit) for audit in audits), key=repr)
        validation_by_cell[key].append(row)
    balanced_validation: list[dict[str, Any]] = []
    for key in sorted(validation_by_cell, key=repr):
        selected = sorted(validation_by_cell[key], key=lambda row: str(row["pair_id"]))[:balanced_validation_per_cell]
        balanced_validation.extend(selected)

    duplicate_audit = []
    for normalized, members in sorted(normalized_clusters.items()):
        if len(members) > 1:
            duplicate_audit.append({"kind": "normalized_caption", "cluster": hashlib.sha1(normalized.encode()).hexdigest()[:16], "normalized_caption": normalized, "members": members})
    for fingerprint, members in sorted(pair_hash_clusters.items()):
        if fingerprint != "unavailable" and len(members) > 1:
            duplicate_audit.append({"kind": "perceptual_pair_candidate", "cluster": fingerprint, "members": sorted(members), "requires_review": True})
    for pair_id, count in sorted(pair_ids.items()):
        if count > 1:
            duplicate_audit.append({"kind": "duplicate_pair_id", "pair_id": pair_id, "count": count, "leakage_blocker": True})

    cross_split_pair_clusters = []
    for fingerprint, members in pair_hash_clusters.items():
        unique_members = sorted(set(members))
        splits = {pair_splits[member] for member in unique_members}
        if "train" in splits and ("val" in splits or "test" in splits):
            cross_split_pair_clusters.append({"cluster": fingerprint, "members": unique_members, "splits": sorted(splits)})
    if cross_split_pair_clusters:
        raise RuntimeError(
            f"Train/validation perceptual pair leakage detected in {len(cross_split_pair_clusters)} clusters; "
            f"first={cross_split_pair_clusters[0]}"
        )

    def write_jsonl(name: str, values: list[dict[str, Any]]) -> None:
        (output / name).write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")
    write_jsonl("duplicate_audit.jsonl", duplicate_audit)
    write_jsonl("caption_quality_audit.jsonl", caption_audits)
    write_jsonl("natural_train_manifest.jsonl", natural_train)
    write_jsonl("balanced_train_manifest.jsonl", balanced_train)
    write_jsonl("natural_validation_manifest.jsonl", natural_validation)
    write_jsonl("balanced_validation_manifest.jsonl", balanced_validation)
    natural_train_retrieval = [row for row in natural_train if bool(row.get("retrieval_supervision", True))]
    natural_validation_retrieval = [row for row in natural_validation if bool(row.get("retrieval_supervision", True))]
    natural_train_localization = [row for row in natural_train if bool(row.get("query_mask_path") or row.get("mask_path"))]
    natural_validation_localization = [row for row in natural_validation if bool(row.get("query_mask_path") or row.get("mask_path"))]
    write_jsonl("natural_train_retrieval_manifest.jsonl", natural_train_retrieval)
    write_jsonl("natural_validation_retrieval_manifest.jsonl", natural_validation_retrieval)
    write_jsonl("natural_train_localization_manifest.jsonl", natural_train_localization)
    write_jsonl("natural_validation_localization_manifest.jsonl", natural_validation_localization)

    physical_pair_ids = set()
    directional_query_target_count = 0
    for row in rows:
        pair_id = str(row["pair_id"])
        dataset_name = str(row["dataset_name"])
        if dataset_name == "s2looking" and pair_id.rsplit(":", 1)[-1] in {
            "appeared",
            "disappeared",
        }:
            pair_id = pair_id.rsplit(":", 1)[0]
            directional_query_target_count += 1
        physical_pair_ids.add((dataset_name, pair_id))
    coverage = {
        "schema_version": "qcpr-v3-coverage-v1",
        "source_manifest_sha256": source_hashes,
        "pair_rows": len(rows),
        "physical_unique_pair_count": len(physical_pair_ids),
        "directional_query_target_count": directional_query_target_count,
        "caption_rows": len(caption_audits),
        "dataset_pairs": dict(Counter(row["dataset_name"] for row in rows)),
        "split_pairs": dict(Counter(row["split"] for row in rows)),
        "retrieval_supervised_pairs": sum(bool(row.get("retrieval_supervision", True)) for row in rows),
        "localization_supervised_pairs": sum(bool(row.get("query_mask_path") or row.get("mask_path")) for row in rows),
        "joint_cell_count": len(cell_counts),
        "singleton_cells": sum(value == 1 for value in cell_counts.values()),
        "under_five_cells": sum(value < 5 for value in cell_counts.values()),
        "duplicate_caption_clusters": sum(len(value) > 1 for value in normalized_clusters.values()),
        "no_change_pair_fraction": sum(any(_direction(classify_caption_semantics(caption)) == "no_change" for caption in row.get("captions", [])) for row in rows) / max(len(rows), 1),
        "no_change_caption_fraction": sum(audit["change_status"] == "no_change" for audit in caption_audits) / max(len(caption_audits), 1),
        "captions_per_pair_min": min((len(row.get("captions", [])) for row in rows), default=0),
        "captions_per_pair_max": max((len(row.get("captions", [])) for row in rows), default=0),
        "caption_cluster_size_max": max((len(value) for value in normalized_clusters.values()), default=0),
        "perceptual_pair_candidate_clusters": sum(key != "unavailable" and len(value) > 1 for key, value in pair_hash_clusters.items()),
        "duplicate_pair_ids": sum(value > 1 for value in pair_ids.values()),
        "train_validation_perceptual_leakage_clusters": 0,
        "derived_counts": {
            "natural_train": len(natural_train), "balanced_train": len(balanced_train),
            "natural_validation": len(natural_validation), "balanced_validation": len(balanced_validation),
            "natural_train_retrieval": len(natural_train_retrieval),
            "natural_validation_retrieval": len(natural_validation_retrieval),
            "natural_train_localization": len(natural_train_localization),
            "natural_validation_localization": len(natural_validation_localization),
        },
    }
    (output / "dataset_coverage_report.json").write_text(json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8")
    csv_header = "dataset,split,pair_id,caption_id,change_status,temporal_direction,objects,locations,counts,relations,mask_empty,foreground_area,label_confidence,parser_confidence\n"
    csv_rows = []
    for audit in caption_audits:
        values = [
            audit["dataset"], audit["split"], audit["pair_id"], audit["caption_id"], audit["change_status"], audit["temporal_direction"],
            "|".join(audit["object_family_audit_tags"]), "|".join(audit["location_audit_tags"]), "|".join(audit["count_audit_tags"]), "|".join(audit["relation_audit_tags"]),
            audit["mask_empty"], audit["foreground_area"], audit["label_confidence"], audit["parser_confidence"],
        ]
        csv_rows.append(",".join(json.dumps(value) if isinstance(value, str) else str(value) for value in values))
    (output / "dataset_coverage_report.csv").write_text(csv_header + "\n".join(csv_rows) + "\n", encoding="utf-8")
    weighting = {
        "formula": "quality * (cell_count + tau)^(-alpha), clipped then mean-normalized and duplicate-pair adjusted",
        "config": config.__dict__,
        "weight_min": min(normalized_weights.values(), default=0.0),
        "weight_max": max(normalized_weights.values(), default=0.0),
        "weight_mean": sum(normalized_weights.values()) / max(len(normalized_weights), 1),
        "cell_count_min": min(cell_counts.values(), default=0),
        "cell_count_max": max(cell_counts.values(), default=0),
        "no_change_batch_fraction_cap": 0.25,
        "large_duplicate_clusters_downweighted": True,
        "sample_weights_capped": True,
    }
    (output / "dataset_weighting_report.json").write_text(json.dumps(weighting, indent=2, sort_keys=True), encoding="utf-8")
    return {"coverage": coverage, "weighting": weighting, "output_dir": str(output)}
