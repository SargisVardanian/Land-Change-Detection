#!/usr/bin/env python3
"""Build a small, unverified TAMMs long-series pilot."""
from __future__ import annotations
import argparse, collections, hashlib, json, tarfile
from datetime import datetime
from pathlib import Path
from typing import Any
from PIL import Image

REVISION = "2fbb79418e121514fc9f382f503e92f07aaf2481"
EXPECTED_INPUTS = 3

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

def date_from_path(path: str) -> datetime:
    return datetime.strptime(path.rsplit("/", 1)[-1].rsplit(".", 1)[0], "%Y-%m-%d")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    raw = args.project_root / "datasets" / "raw" / "TAMMs"
    metadata_path = raw / "tamms_data.json"
    shard = raw / "waste_disposal.tar"
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = [row for row in rows if str(row.get("output_image", "")).startswith("waste_disposal/")]
    rows = sorted(rows, key=lambda row: str(row["output_image"]))[:args.limit]
    if len(rows) != args.limit:
        raise SystemExit(f"requested {args.limit} TAMMs rows, found {len(rows)}")
    args.output.mkdir(parents=True, exist_ok=True)
    extracted = args.output / "assets"
    extracted.mkdir(exist_ok=True)
    sequences=[]; texts=[]; image_hashes=[]; decode_failures=[]
    with tarfile.open(shard) as archive:
        members={m.name:m for m in archive.getmembers() if m.isfile()}
        for row in rows:
            image_paths=list(row.get("input_images", []))+[str(row["output_image"])]
            if len(row.get("input_images", [])) != EXPECTED_INPUTS or len(set(image_paths)) != 4:
                raise SystemExit("TAMMs sequence must contain three distinct inputs and one output")
            sequence_source_id=str(row["output_image"]).rsplit("/",1)[0]
            sequence_id=f"tamms:{REVISION}:{sequence_source_id}"
            dates=[date_from_path(path) for path in image_paths]
            if dates != sorted(dates) or len(set(dates)) != len(dates):
                raise SystemExit(f"non-monotonic or duplicate timestamps for {sequence_id}")
            frames=[]
            seq_dir=extracted / sequence_source_id.replace("/", "__")
            seq_dir.mkdir(parents=True, exist_ok=True)
            for index, (source_path, timestamp) in enumerate(zip(image_paths, dates, strict=True)):
                member=members.get(source_path)
                if member is None:
                    raise SystemExit(f"missing TAMMs archive member {source_path}")
                data=archive.extractfile(member).read()
                target=seq_dir / Path(source_path).name
                target.write_bytes(data)
                try:
                    with Image.open(target) as image:
                        image.verify()
                    with Image.open(target) as image:
                        width,height=int(image.width),int(image.height)
                except Exception as exc:
                    decode_failures.append(f"{sequence_id}:{source_path}:{exc}")
                    continue
                image_hashes.append(sha256_bytes(data))
                frames.append({"frame_id":f"{sequence_id}:frame:{index}","path":str(target),"source_path":source_path,"timestamp":timestamp.date().isoformat(),"role":"input" if index < 3 else "output","width":width,"height":height,"sha256":sha256_bytes(data)})
            if len(frames)!=4:
                continue
            intervals=[(dates[i+1]-dates[i]).days for i in range(3)]
            sequences.append({"schema_version":"qcpr-long-series-v1","sequence_id":sequence_id,"source_dataset":"TAMMs","source_version":REVISION,"source_sequence_id":sequence_source_id,"frames":frames,"frame_count":4,"time_intervals_days":intervals,"query_temporal_extent":{"start":dates[0].date().isoformat(),"end":dates[-1].date().isoformat()},"relevant_frame_range":None,"change_onset":None,"change_duration":None,"spatial_evidence_per_time":None,"input_prompt":row.get("input_prompt"),"output_description":row.get("output_description"),"text_provenance":"generated_by_Qwen2.5-VL_per_official_dataset_card","verification_status":"generated_unverified","training_enabled":False,"reason_training_disabled":"generated text has no independent factual review; source split contract is not provided"})
            texts.extend([{"schema_version":"qcpr-long-series-text-v1","query_id":f"{sequence_id}:scene_query","sequence_id":sequence_id,"text":str(row.get("input_prompt", "")),"query_scope":"stable_scene_candidate","text_provenance":"generated_by_Qwen2.5-VL","verification_status":"generated_unverified","training_enabled":False},{"schema_version":"qcpr-long-series-text-v1","query_id":f"{sequence_id}:change_query","sequence_id":sequence_id,"text":str(row.get("output_description", "")),"query_scope":"long_series_change_candidate","text_provenance":"generated_by_Qwen2.5-VL","verification_status":"generated_unverified","training_enabled":False}])
    if decode_failures:
        raise SystemExit(f"TAMMs decode failures: {decode_failures[:3]}")
    if len(sequences)!=args.limit:
        raise SystemExit(f"only {len(sequences)} valid sequences were built")
    write_jsonl(args.output/"tamms_long_series_manifest_unverified.jsonl",sequences)
    write_jsonl(args.output/"tamms_long_series_text_registry_unverified.jsonl",texts)
    split_values=collections.Counter(row.get("split") for row in sequences)
    audit={"status":"LONG_SERIES_PILOT_TEXT_VERIFICATION_REQUIRED","source":"TAMMs","source_revision":REVISION,"metadata_path":str(metadata_path),"shard_path":str(shard),"pilot_sequence_count":len(sequences),"pilot_text_count":len(texts),"frame_count_distribution":dict(collections.Counter(row["frame_count"] for row in sequences)),"time_interval_days_summary":{"min":min(x for row in sequences for x in row["time_intervals_days"]),"max":max(x for row in sequences for x in row["time_intervals_days"])},"decode_failure_count":len(decode_failures),"duplicate_image_hash_count":len(image_hashes)-len(set(image_hashes)),"official_split_present":False,"training_enabled":False,"mask_free":True,"required_followup":["define event/scene-disjoint train/development/test split","verify generated input_prompt and output_description claims independently","annotate change onset/duration/relevant frame range before sequence training"]}
    write_json(args.output/"tamms_long_series_pilot_audit.json",audit)
    print(json.dumps({"status":audit["status"],"sequences":len(sequences),"texts":len(texts),"frames":sum(row["frame_count"] for row in sequences),"duplicate_image_hash_count":audit["duplicate_image_hash_count"],"output":str(args.output)},sort_keys=True))
    return 0
if __name__ == "__main__": raise SystemExit(main())
