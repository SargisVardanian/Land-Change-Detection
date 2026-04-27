from __future__ import annotations

from collections import Counter

import numpy as np

from ..contracts import CellPack, SegmentationArtifact

HIGH_SIGNAL_LABELS = {"building", "road", "pavement", "bareland", "cropland", "grass", "tree", "water"}
BACKGROUND_LABELS = {"background", "other", "unknown"}


def _class_ratios(class_map: np.ndarray, legend: dict[int, str]) -> dict[str, float]:
    total = max(int(class_map.size), 1)
    counts = Counter(int(value) for value in class_map.reshape(-1))
    ratios: dict[str, float] = {}
    for class_id, count in counts.items():
        label = legend.get(class_id, str(class_id))
        ratios[label] = count / total
    return ratios


def _top_labels(ratios: dict[str, float], limit: int = 3) -> list[tuple[str, float]]:
    filtered = [(label, value) for label, value in ratios.items() if label not in BACKGROUND_LABELS and value >= 0.04]
    filtered.sort(key=lambda item: item[1], reverse=True)
    return filtered[:limit]


def _object_phrase(top_labels: list[tuple[str, float]]) -> str:
    if not top_labels:
        return "mixed low-detail surface"
    parts: list[str] = []
    for label, ratio in top_labels:
        if label == "building":
            parts.append("built fragments")
        elif label == "road":
            parts.append("road-like linear surface")
        elif label == "pavement":
            parts.append("compact paved surface")
        elif label == "bareland":
            parts.append("bare or prepared ground")
        elif label == "cropland":
            parts.append("cultivated ground")
        elif label == "grass":
            parts.append("low vegetation")
        elif label == "tree":
            parts.append("tree or shrub cover")
        elif label == "water":
            parts.append("water surface")
        else:
            parts.append(label.replace("_", " "))
        if ratio >= 0.35:
            break
    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return "; ".join(deduped[:3]) if deduped else "mixed low-detail surface"


def _dominant_transition(before: dict[str, float], after: dict[str, float]) -> tuple[str, float]:
    labels = sorted(set(before) | set(after))
    best_label = "stable"
    best_delta = 0.0
    for label in labels:
        delta = after.get(label, 0.0) - before.get(label, 0.0)
        magnitude = abs(delta)
        if magnitude > best_delta:
            best_delta = magnitude
            best_label = label
    return best_label, best_delta


def _support_from_deltas(before: dict[str, float], after: dict[str, float]) -> str:
    signals: list[str] = []
    labels = sorted(set(before) | set(after))
    deltas = [(label, after.get(label, 0.0) - before.get(label, 0.0)) for label in labels]
    deltas.sort(key=lambda item: abs(item[1]), reverse=True)
    for label, delta in deltas[:3]:
        if abs(delta) < 0.06:
            continue
        direction = "more" if delta > 0 else "less"
        nice_label = label.replace("_", " ")
        signals.append(f"{direction} {nice_label} ({abs(delta) * 100:.0f}%)")
    return ", ".join(signals) if signals else "weak semantic transition signal"


def _likely_change(before: dict[str, float], after: dict[str, float], changed_ratio: float) -> str:
    building_delta = after.get("building", 0.0) - before.get("building", 0.0)
    pavement_delta = after.get("pavement", 0.0) - before.get("pavement", 0.0)
    road_delta = after.get("road", 0.0) - before.get("road", 0.0)
    bare_delta = after.get("bareland", 0.0) - before.get("bareland", 0.0)
    crop_delta = after.get("cropland", 0.0) - before.get("cropland", 0.0)
    grass_delta = after.get("grass", 0.0) - before.get("grass", 0.0)
    tree_delta = after.get("tree", 0.0) - before.get("tree", 0.0)
    water_delta = after.get("water", 0.0) - before.get("water", 0.0)

    if changed_ratio < 0.08:
        return "no clear structural change"
    if building_delta >= 0.16:
        return "possible new built fragment or rectilinear surface addition"
    if pavement_delta >= 0.14:
        return "possible compact hard-surface expansion"
    if road_delta >= 0.14:
        return "possible road-edge or access-line change"
    if bare_delta >= 0.16:
        return "possible prepared ground or grading expansion"
    if crop_delta >= 0.16 or grass_delta >= 0.16:
        return "possible vegetation or cultivated-surface expansion"
    if tree_delta >= 0.14:
        return "possible tree or shrub cover increase"
    if water_delta >= 0.12:
        return "possible water-surface expansion"
    if bare_delta <= -0.16 and (building_delta > 0.08 or pavement_delta > 0.08):
        return "possible conversion from bare ground to more structured surface"
    if (crop_delta <= -0.14 or grass_delta <= -0.14) and bare_delta > 0.10:
        return "possible vegetation loss with exposed ground increase"
    return "localized semantic surface transition"


def _technical_interpretation(before: dict[str, float], after: dict[str, float], changed_ratio: float) -> str:
    likely = _likely_change(before, after, changed_ratio)
    if likely == "no clear structural change":
        return "Semantic composition stays broadly similar in this cell, so any visible difference is weak or ambiguous."
    if "built fragment" in likely:
        return "The segmentation shift is more consistent with a new compact built surface or small structured addition than with a purely photometric change."
    if "hard-surface" in likely:
        return "This cell shifts toward compact paved or service-surface classes, which is consistent with yard formation, paving, or another hardened surface."
    if "road-edge" in likely:
        return "The class mix shifts toward road-like linear surface, suggesting access-line rework, edge widening, or another transport-surface modification."
    if "prepared ground" in likely:
        return "The cell gains bare/prepared ground, which is more consistent with grading, scraping, clearing, or surface preparation than with stable land cover."
    if "vegetation" in likely:
        return "The cell gains vegetation-related classes, suggesting cultivated or low-vegetation expansion rather than compact-surface growth."
    if "tree" in likely:
        return "The semantic balance shifts toward woody cover, suggesting canopy or shrub increase."
    if "water-surface" in likely:
        return "The cell gains water-class area, suggesting a local water-surface increase."
    if "conversion from bare ground" in likely:
        return "This looks like a cautious transition from exposed ground toward a more structured compact surface, but the exact object type remains uncertain."
    if "vegetation loss" in likely:
        return "The class balance suggests vegetation retreat with more exposed ground, consistent with clearing, drying, or disturbance."
    return "The cell shows a mixed semantic transition, but the exact object-level interpretation remains cautious."


def _confidence(score: float, changed_ratio: float) -> str:
    if score >= 75 or changed_ratio >= 0.35:
        return "high"
    if score >= 40 or changed_ratio >= 0.16:
        return "medium"
    return "low"


def build_cell_report_rows_from_segmentation(packs: list[CellPack]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pack in packs:
        before_ratios = _class_ratios(pack.before_semantic_map, pack.legend)
        after_ratios = _class_ratios(pack.after_semantic_map, pack.legend)
        changed_ratio = float(np.mean(pack.before_semantic_map != pack.after_semantic_map))
        dominant_label, dominant_delta = _dominant_transition(before_ratios, after_ratios)
        score = round(min(100.0, changed_ratio * 100.0 + abs(dominant_delta) * 140.0), 1)
        rows.append(
            {
                "cell": pack.cell.cell_id,
                "objects_before": _object_phrase(_top_labels(before_ratios)),
                "objects_after": _object_phrase(_top_labels(after_ratios)),
                "likely_change": _likely_change(before_ratios, after_ratios, changed_ratio),
                "technical_interpretation": _technical_interpretation(before_ratios, after_ratios, changed_ratio),
                "support": _support_from_deltas(before_ratios, after_ratios),
                "score": score,
                "confidence": _confidence(score, changed_ratio),
                "changed_ratio": round(changed_ratio, 4),
                "dominant_transition_label": dominant_label,
            }
        )
    rows.sort(key=lambda row: row["cell"])
    return rows


def build_scene_overview_from_rows(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No cell-level interpretation is available."
    changed_rows = [row for row in rows if str(row.get("likely_change", "")).strip() != "no clear structural change"]
    top_rows = sorted(rows, key=lambda row: float(row.get("score", 0.0)), reverse=True)[:4]
    top_cells = ", ".join(str(row["cell"]) for row in top_rows if float(row.get("score", 0.0)) >= 20) or "no strong cells"

    category_counts = Counter()
    for row in changed_rows:
        likely = str(row.get("likely_change", ""))
        if "prepared ground" in likely or "grading" in likely:
            category_counts["prepared ground / grading"] += 1
        elif "built fragment" in likely or "hard-surface" in likely:
            category_counts["compact or rectilinear surface additions"] += 1
        elif "road-edge" in likely or "access-line" in likely:
            category_counts["linear access or boundary changes"] += 1
        elif "vegetation" in likely or "tree" in likely or "water" in likely:
            category_counts["land-cover class shifts"] += 1
        else:
            category_counts["mixed localized transitions"] += 1
    dominant = ", ".join(label for label, _ in category_counts.most_common(2)) or "weak localized transitions"
    locality = "localized in a limited set of cells" if len(changed_rows) <= 6 else "distributed across much of the crop"
    return (
        f"Change is {locality}, with the strongest cells around {top_cells}. "
        f"The dominant semantic transition patterns are {dominant}. "
        "These labels are conservative class-based interpretations from before/after segmentation and should not be read as confirmed object identity where the imagery remains ambiguous."
    )


def build_pairwise_visual_context(
    packs: list[CellPack],
    before_segmentation: SegmentationArtifact,
    after_segmentation: SegmentationArtifact,
    rows: list[dict[str, object]],
) -> str:
    before_summary = ", ".join(
        f"{item['label']} {item['percent']:.1f}%"
        for item in sorted(before_segmentation.label_summary, key=lambda item: item["percent"], reverse=True)
        if item["label"] not in BACKGROUND_LABELS and item["percent"] >= 4
    )
    after_summary = ", ".join(
        f"{item['label']} {item['percent']:.1f}%"
        for item in sorted(after_segmentation.label_summary, key=lambda item: item["percent"], reverse=True)
        if item["label"] not in BACKGROUND_LABELS and item["percent"] >= 4
    )
    lines = [
        "Pairwise cell context from per-timestamp semantic segmentation and aligned before/after crops.",
        "Use before and after RGB images as primary evidence. Use overlays only as semantic support and say so when the semantic classes look uncertain.",
        f"Before semantic summary: {before_summary or 'no dominant non-background class'}",
        f"After semantic summary: {after_summary or 'no dominant non-background class'}",
        "Cell-by-cell semantic transition summary:",
    ]
    row_by_cell = {str(row["cell"]): row for row in rows}
    for pack in packs:
        row = row_by_cell.get(pack.cell.cell_id, {})
        lines.append(
            "- "
            + f"{pack.cell.cell_id}: before={row.get('objects_before', 'unknown')}; "
            + f"after={row.get('objects_after', 'unknown')}; "
            + f"candidate transition={row.get('likely_change', 'unknown')}; "
            + f"support={row.get('support', 'none')}; "
            + f"score={row.get('score', 0.0):.1f}"
        )
    return "\n".join(lines)
