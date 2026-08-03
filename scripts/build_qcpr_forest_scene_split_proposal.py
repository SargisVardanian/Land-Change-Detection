#!/usr/bin/env python3
"""Derive a leakage-free Forest-Change split proposal by image components.

This does not promote the source. It records a deterministic alternative to
Forest-Change's pair-level split, which shares temporal images across splits.
"""
from __future__ import annotations
import argparse, collections, hashlib, json
from pathlib import Path
from typing import Any

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+"\n", encoding="utf-8")
def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    rows=read_jsonl(args.input)
    groups: dict[str,list[dict[str,Any]]]=collections.defaultdict(list)
    for row in rows:
        groups[str(row["source_metadata"]["source_scene_group_id"])].append(row)
    total=len(rows)
    targets={"train":round(total*0.70),"development":round(total*0.15),"test":total-round(total*0.70)-round(total*0.15)}
    assigned=collections.Counter()
    group_split={}
    # Largest components first; deterministic hash breaks ties. This is a
    # structural leakage repair, not a claim that the resulting balance is
    # scientifically optimal.
    ordered=sorted(groups.items(), key=lambda item:(-len(item[1]),sha(item[0])))
    for group_id, group_rows in ordered:
        candidates=sorted(targets, key=lambda name:(-(targets[name]-assigned[name]),name))
        split=candidates[0]
        group_split[group_id]=split
        assigned[split]+=len(group_rows)
    proposal=[]
    for row in rows:
        copied=dict(row)
        copied["split"]=group_split[str(row["source_metadata"]["source_scene_group_id"])]
        copied["source_metadata"]={**row["source_metadata"],"split_policy":"derived_image_component_hash_balanced_proposal","official_split":row["split"],"training_enabled":False}
        proposal.append(copied)
    output=args.output; output.mkdir(parents=True,exist_ok=True)
    write_jsonl(output/"forest_change_pair_manifest_scene_disjoint_proposal.jsonl",proposal)
    groups_by_split=collections.defaultdict(set)
    for group_id,split in group_split.items(): groups_by_split[split].add(group_id)
    overlap=sum(len(groups_by_split[a]&groups_by_split[b]) for a in groups_by_split for b in groups_by_split if a<b)
    audit={"status":"SCENE_DISJOINT_SPLIT_PROPOSAL_NOT_RELEASED","input":str(args.input),"pair_count":total,"component_count":len(groups),"target_pair_counts":targets,"proposed_pair_counts":dict(sorted(assigned.items())),"proposed_component_counts":{k:len(v) for k,v in sorted(groups_by_split.items())},"cross_split_component_overlap":overlap,"official_split_counts":dict(collections.Counter(str(row["split"]) for row in rows)),"training_enabled":False,"reason":"Forest-Change official pair split has shared images; proposal is derived only from exact image-hash components and still requires source-level review/licence/split acceptance"}
    write_json(output/"forest_change_scene_disjoint_split_proposal.json",audit)
    print(json.dumps(audit,sort_keys=True))
    return 0
if __name__=="__main__": raise SystemExit(main())
