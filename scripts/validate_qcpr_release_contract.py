#!/usr/bin/env python3
"""Run CPU loader/relevance/mask-free contracts on an immutable release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from PIL import Image


FORBIDDEN = {"mask", "masks", "semantic", "target", "targets", "label", "labels", "ground_truth", "gt", "query_masks"}


def walk(value, prefix=""):
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if str(key).casefold() in FORBIDDEN or str(key).casefold().split("_")[-1] in FORBIDDEN:
                yield path
            yield from walk(nested, path)
    elif isinstance(value, list):
        for i, nested in enumerate(value): yield from walk(nested, f"{prefix}[{i}]")


def read(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--release", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); args=ap.parse_args()
    root=args.release; retrieval=read(root/"manifests/retrieval_exact_train_v2.jsonl"); grounding=read(root/"manifests/grounding_mask_free_train_v2.jsonl")
    sample=retrieval[:256]; physical={row["canonical_pair_id"] for row in sample}; pairs={row["canonical_pair_id"] for row in retrieval}
    decode_failures=[]
    for row in sample:
        for key in ("t1_path", "t2_path"):
            path=Path(row.get(key,""))
            try:
                with Image.open(path) as image: image.verify()
            except Exception as exc: decode_failures.append({"path":str(path),"error":type(exc).__name__})
    forbidden=[]
    for row in retrieval[:256]+grounding[:256]:
        forbidden.extend(walk(row))
    matrix={"physical_microbatch":16,"logical_physical_batch":128,"logical_text_queries":256,"score_matrix":"256x128","sampled_retrieval_rows":len(sample),"unique_physical_ids":len(physical)}
    result={"retrieval_manifest_rows":len(retrieval),"grounding_manifest_rows":len(grounding),"contract":matrix,"decode_failures":decode_failures[:20],"mask_free_forbidden_keys":sorted(set(forbidden)),"passed":not decode_failures and not forbidden and len(pairs)>0}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2)); return 0 if result["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
