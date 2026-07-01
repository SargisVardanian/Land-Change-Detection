from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from land_change_detection.data.unichange_subset import resolve_levir_mci_root
from land_change_detection.levir_mci import discover_levir_mci_samples
from land_change_detection.models.retrieval_heads import normalize_caption_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic, stratified LEVIR-MCI evaluation manifest.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--per-stratum", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def mask_fraction(path: str | Path) -> float:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    return float(mask.mean())


def stratum_for_fraction(fraction: float) -> str:
    if fraction == 0.0:
        return "no_change"
    if fraction <= 0.01:
        return "small_change"
    if fraction <= 0.05:
        return "medium_change"
    return "large_change"


def main() -> int:
    args = parse_args()
    root = resolve_levir_mci_root(args.data_root).root
    samples = [sample for sample in discover_levir_mci_samples(root) if sample.split == args.split]
    if not samples:
        raise RuntimeError(f"No {args.split!r} samples found under {root}")

    normalized_counts = Counter(
        normalize_caption_text(caption)
        for sample in samples
        for caption in sample.captions
        if caption.strip()
    )

    rows = []
    for sample in samples:
        fraction = mask_fraction(sample.binary_change_mask)
        captions = [caption.strip() for caption in sample.captions if caption.strip()]
        if not captions:
            continue
        query_caption = min(
            captions,
            key=lambda text: (
                normalized_counts[normalize_caption_text(text)],
                len(normalize_caption_text(text)),
                normalize_caption_text(text),
            ),
        )
        rows.append(
            {
                "pair_id": sample.sample_id,
                "split": sample.split,
                "stratum": stratum_for_fraction(fraction),
                "mask_fraction": fraction,
                "query_caption": query_caption,
                "query_caption_frequency": normalized_counts[normalize_caption_text(query_caption)],
                "captions": captions,
                "t1": sample.image_before,
                "t2": sample.image_after,
                "mask": sample.binary_change_mask,
            }
        )

    rng = random.Random(args.seed)
    selected = []
    strata = ("no_change", "small_change", "medium_change", "large_change")
    for stratum in strata:
        candidates = [row for row in rows if row["stratum"] == stratum]
        rng.shuffle(candidates)
        selected.extend(candidates[: args.per_stratum])

    selected.sort(key=lambda row: (strata.index(row["stratum"]), row["pair_id"]))
    payload = {
        "dataset": "LEVIR-MCI",
        "root": str(root),
        "split": args.split,
        "seed": args.seed,
        "per_stratum": args.per_stratum,
        "count": len(selected),
        "strata_counts": dict(Counter(row["stratum"] for row in selected)),
        "selection_rule": "deterministic stratification by binary-mask area; least-frequent caption per pair",
        "items": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": len(selected), "strata_counts": payload["strata_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
