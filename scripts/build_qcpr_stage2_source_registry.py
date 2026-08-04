#!/usr/bin/env python3
"""Conservative, hashable Stage-2 source registry."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
from typing import Any

STATES = ("NOT_REQUESTED","ACCESS_REQUIRED","DOWNLOAD_PARTIAL","DOWNLOADED","EXTRACTED","CANONICALIZED","MANIFESTED","LOADER_VALIDATED","TRAINING_ENABLED")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def inventory(path: Path) -> dict[str, Any]:
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.exists() else []
    return {"path":str(path),"exists":path.exists(),"file_count":len(files),"bytes":sum(p.stat().st_size for p in files),"files":[{"path":str(p),"bytes":p.stat().st_size} for p in files[:256]],"file_sample_truncated":len(files)>256}

def source(name: str, official: list[str], raw: Path, roles: list[str], license_name: str, state: str, *, blocker: str|None=None, version: str|None=None, pair_count: int|None=None, caption_count: int|None=None, loader_validation: dict[str,Any]|None=None, notes: list[str]|None=None) -> dict[str,Any]:
    if state not in STATES: raise ValueError(state)
    return {"source_dataset":name,"official_locations":official,"version_or_revision":version,"license":license_name,"raw_inventory":inventory(raw),"state":state,"roles":roles,"physical_pair_count":pair_count,"text_count":caption_count,"dense_label_count":None,"official_split":None,"sensor":None,"native_dimensions":None,"gsd":None,"temporal_metadata":None,"accessibility":"blocked" if blocker else "available","blocker":blocker,"notes":notes or [],"loader_validation":loader_validation or {"passed":False,"artifact":None}}

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True); p.add_argument("--project-root",type=Path,required=True); p.add_argument("--current-release",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    raw=a.project_root/"datasets/raw"; ebd=raw/"RSCC/EBD"; rcd_base=raw/"synthetic_rcd_second"; rcd_candidates=sorted(rcd_base.glob("*/train/train_A.txt")); rcd=rcd_candidates[0].parents[1] if rcd_candidates else rcd_base; sysu=raw/"SYSU-CD"; hiucd=raw/"Hi-UCD-S"
    old=a.current_release/"registries/source_registry.json"; old_rows=json.loads(old.read_text()) if old.exists() else []; by_name={str(r.get("source_dataset")):r for r in old_rows}
    required=[rcd/"train"/n for n in ("train_A.txt","train_B.txt","train_gt.txt","train_synthetic_A.txt",*[f"B-{i:05d}.tar" for i in range(10)],*[f"gt-{i:05d}.tar" for i in range(10)])]
    rcd_ok=all(x.is_file() and x.stat().st_size>0 for x in required); ebd_ok=all((ebd/f"EBD.tar.gz-part-{i}").is_file() and (ebd/f"EBD.tar.gz-part-{i}").stat().st_size>0 for i in range(5)); pilot_path=a.project_root/"manifests/qcpr_dataset_v2_stage2_audit/rscc_ebd/rscc_ebd_pair_audit.json"; pilot=json.loads(pilot_path.read_text()) if pilot_path.exists() else {}; loader_path=a.project_root/"manifests/qcpr_dataset_v2_stage2_audit/rscc_ebd/rscc_mask_free_loader_contract.json"; loader=json.loads(loader_path.read_text()) if loader_path.exists() else {}; qvq_path=a.project_root/"manifests/qcpr_dataset_v2_stage2_audit/rscc_ebd/rscc_ebd_qvq_caption_audit.json"; qvq=json.loads(qvq_path.read_text()) if qvq_path.exists() else {}; ebd_loader_ok=bool(pilot.get("identity_proven") and loader.get("passed") and len(pilot.get("split_counts",{}))==3); ebd_pairs=pilot.get("pair_count"); ebd_caption_count=qvq.get("caption_count")
    rows=[
      source("LEVIR-MCI",["https://github.com/SaiyangHuang/LEVIR"],raw/"LEVIR-MCI",["temporal_retrieval","dense_evaluation"],"source terms; official repository","TRAINING_ENABLED",version="current audited release",pair_count=8143),
      source("SECOND-CC",["https://github.com/Sun-Yuting/SECOND-CC"],raw/"SECOND-CC-extracted",["temporal_retrieval","dense_evaluation"],"source terms; official repository","TRAINING_ENABLED",version="current audited release",pair_count=4701),
      source("RSCC-EBD",["https://huggingface.co/datasets/BiliSakura/RSCC"],ebd,["temporal_physical","dense_evaluation"],"CC-BY-4.0 for RSCC metadata/EBD; xBD restrictions separate","LOADER_VALIDATED" if ebd_loader_ok else ("DOWNLOADED" if ebd_ok else ("DOWNLOAD_PARTIAL" if ebd.exists() else "ACCESS_REQUIRED")),blocker=None if ebd_ok else "official EBD multipart archive incomplete",version="HF 791a00849c3e684f54df38291cf35a7d828d7004",pair_count=ebd_pairs,caption_count=ebd_caption_count,loader_validation={"passed":ebd_loader_ok,"artifact":str(loader_path) if ebd_loader_ok else None}),
      source("Synthetic-RCD-SECOND",["https://huggingface.co/datasets/yilmazkorkmaz/Synthetic_RCD_1","https://captain-whu.github.io/SCD/"],rcd,["synthetic_auxiliary","dense_evaluation"],"see official source terms","DOWNLOADED" if rcd_ok else "DOWNLOAD_PARTIAL",blocker="original SECOND-A mapping absent; synthetic-A mode is diagnostic only",version="local shards; HF 0203878cabeac82c14b7dc9f98e7be68e6eb07be"),
      source("SYSU-CD",["https://github.com/liumency/SYSU-CD","https://pan.baidu.com/s/15lQPG_hXZbLp91VywwcT7Q","https://mail2sysueducn-my.sharepoint.com/"],sysu,["temporal_generated_caption_candidate","dense_evaluation"],"see official release","ACCESS_REQUIRED",blocker="repository clone contains README/illustrations but no official image archive",version=str(by_name.get("SYSU-CD",{}).get("version_or_revision") or "repository metadata only")),
      source("Hi-UCD",["https://github.com/Daisy-7/Hi-UCD-S","official request form in README"],hiucd,["temporal_generated_caption_candidate","dense_evaluation"],"academic-only per official README","ACCESS_REQUIRED",blocker="corrected 2025-11-01 archive is request-gated and absent",version="repository metadata only"),
      source("S2Looking",["current audited release"],raw/"S2Looking",["grounding","dense_evaluation"],"see current release","MANIFESTED",blocker="derived queries are not independently reviewed semantic multi-positives",version="current audited release",pair_count=5000,caption_count=10000),
    ]
    rows.extend([
      source(
        "Forest-Change",
        ["https://huggingface.co/datasets/JimmyBrocko/Forest-Change"],
        raw / "Forest-Change",
        ["non_disaster_temporal", "forest_vegetation", "dense_evaluation"],
        "source terms require redistribution audit",
        "EXTRACTED",
        blocker="official split has 22 cross-split image components; captions are mixed-provenance and unverified",
        version="pilot revision e8b25bf09c85ec85633d1b1b554f7bb23e47724d",
        pair_count=334,
        caption_count=1670,
        notes=["scene-disjoint proposal exists but is not released", "training_enabled=false"],
      ),
      source(
        "DUBAI-CC",
        ["https://service.tib.eu/ldmservice/dataset/dubai-cc%E2%80%93a-dataset-for-remote-sensing-change-captioning"],
        raw / "DUBAI-CC",
        ["external_exact_benchmark", "change_captioning"],
        "paper/source terms; verify before acquisition",
        "ACCESS_REQUIRED",
        blocker="official TIB endpoint was unavailable during the cluster audit; no archive acquired",
        version="official release not locally pinned",
        pair_count=500,
        caption_count=2500,
        notes=["reported counts are not locally verified"],
      ),
      source(
        "TAMMs",
        ["https://huggingface.co/datasets/IceInPot/TAMMs"],
        raw / "TAMMs",
        ["long_series_retrieval_candidate", "temporal_query_generation"],
        "Apache-2.0 metadata; base fMoW terms require audit",
        "EXTRACTED",
        blocker="pilot has no official/event-disjoint split and generated text is unverified",
        version="pilot revision 2fbb79418e121514fc9f382f503e92f07aaf2481",
        pair_count=100,
        caption_count=200,
        notes=["pilot only; 4 frames per sequence", "training_enabled=false"],
      ),
      source(
        "DynamicEarthNet",
        ["https://mediatum.ub.tum.de/1650201"],
        raw / "DynamicEarthNet",
        ["long_series_physical", "temporal_query_generation"],
        "official terms required",
        "ACCESS_REQUIRED",
        blocker="metadata-only; official physical archive is not locally acquired",
        version="metadata audit only",
        notes=["do not create T1=T2 pairs", "query generation requires independent verification"],
      ),
      source(
        "SpaceNet-7",
        ["https://spacenet.ai/datasets/"],
        raw / "SpaceNet-7",
        ["long_series_physical", "urban_temporal_query_generation"],
        "official challenge terms required",
        "ACCESS_REQUIRED",
        blocker="metadata-only; official sequence archive is not locally acquired",
        version="metadata audit only",
        notes=["monthly sequence source; event/geography split required"],
      ),
      source(
        "TERRA-CD",
        ["https://github.com/omkarsoak/TERRA-CD"],
        raw / "TERRA-CD",
        ["non_disaster_temporal", "land_cover_transition", "dense_evaluation"],
        "CC-BY-4.0 as declared by repository; verify archive",
        "ACCESS_REQUIRED",
        blocker="official physical archive is not locally acquired; no verified text",
        version="repository metadata only",
        pair_count=5221,
        notes=["candidate for structured transition caption generation"],
      ),
      source(
        "RSRCC",
        ["https://huggingface.co/datasets/google/RSRCC"],
        raw / "RSRCC",
        ["semantic_retrieval", "localized_queries", "qa_evaluation"],
        "official dataset terms required",
        "ACCESS_REQUIRED",
        blocker="official source/parent-image overlap and mask-derived provenance audit incomplete",
        version="metadata audit only",
        caption_count=126131,
        notes=["do not route QA automatically into exact retrieval", "localized use requires parent-image audit"],
      ),
    ])
    for name in ("RSICD","NWPU-Captions","RSITMD"):
        rows.append(source(name,["official source required before acquisition"],raw/name,["static_scene_language"],"unverified","NOT_REQUESTED",blocker="static auxiliary acquisition deferred until temporal pilot is complete"))
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=a.repo,check=True,text=True,capture_output=True).stdout.strip()
    payload={"schema_version":"qcpr-stage2-source-registry-v1","code_sha":sha,"current_release":str(a.current_release),"sources":rows,"source_count":len(rows),"new_real_physical_source_present":ebd_loader_ok,"stage2_ready":False,"status":"DATA_QUALITY_HOLD","notes":["RSCC-EBD physical pair pilot is loader-validated but generated QvQ captions remain disabled pending independent verification.","Synthetic-RCD real-A mode remains blocked without original SECOND-A assets.","Semantic multi-positive supervision is audited separately and is not inferred from same-pair captions."]}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps({"output":str(a.output),"status":payload["status"],"rscc_ebd_complete":ebd_ok,"rcd_complete":rcd_ok},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
