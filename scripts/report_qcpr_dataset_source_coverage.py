#!/usr/bin/env python3
"""Write an evidence-based Dataset-v2 source-coverage report."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from collections import Counter

SOURCES = {
 "LEVIR-MCI": ("https://github.com/justchenhao/LEVIR", "physically confirmed", "TRAINING_ENABLED"),
 "SECOND-CC": ("https://github.com/Chen-Yang-Liu/SECOND-CC", "physically confirmed", "TRAINING_ENABLED"),
 "S2Looking": ("https://github.com/S2Looking/Dataset", "physically confirmed", "LOADER_VALIDATED"),
 "Synthetic RCD SECOND": ("https://huggingface.co/datasets/yilmazkorkmaz/Synthetic_RCD_1", "partial", "DOWNLOAD_PARTIAL"),
 "ChangeChat-87k": ("https://github.com/hanlinwu/ChangeChat", "adapter-only", "ACCESS_REQUIRED"),
 "Hi-UCD": ("https://github.com/Daisy-7/Hi-UCD-S", "adapter-only", "ACCESS_REQUIRED"),
 "SYSU-CD": ("https://github.com/liumency/SYSU-CD", "adapter-only", "NOT_REQUESTED"),
 "RSCC": ("official source pending registry confirmation", "adapter-plan", "NOT_REQUESTED"),
 "RSICD": ("https://github.com/201528014227051/RSICD_optimal", "adapter-only", "NOT_REQUESTED"),
 "NWPU-Captions": ("https://github.com/YonghaoXu/RSICD_optimal", "adapter-only", "NOT_REQUESTED"),
 "RSITMD": ("https://github.com/xiaoyuan1996/RSITMD", "adapter-only", "NOT_REQUESTED"),
 "CA-CDD": ("official access required", "adapter-only", "ACCESS_REQUIRED"),
 "DisasterM3": ("official access required", "adapter-only", "ACCESS_REQUIRED"),
 "SkyScript": ("official access required", "adapter-only", "ACCESS_REQUIRED"),
}

def digest(path: Path) -> str | None:
 if not path.exists(): return None
 h=hashlib.sha256()
 for p in sorted(x for x in path.rglob('*') if x.is_file()):
  h.update(str(p.relative_to(path)).encode()); h.update(str(p.stat().st_size).encode())
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--dataset-v2-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
 summary=json.loads((a.dataset_v2_root/'build_summary.json').read_text())
 pair_counts=Counter(); caption_counts=Counter()
 for line in (a.dataset_v2_root/'registries/pair_registry.jsonl').open(): pair_counts[json.loads(line)['source_dataset'].casefold()]+=1
 for line in (a.dataset_v2_root/'registries/caption_registry.jsonl').open(): caption_counts[str(json.loads(line).get('dataset_name','')).casefold()]+=1
 aliases={'LEVIR-MCI':'LEVIR-MCI','SECOND-CC':'SECOND-CC-extracted','S2Looking':'S2Looking','Synthetic RCD SECOND':'synthetic_rcd_second'}
 rows=[]
 for name,(official,kind,state) in SOURCES.items():
  raw=a.raw_root/aliases.get(name,'__missing__'); exists=raw.exists(); size=sum(x.stat().st_size for x in raw.rglob('*') if x.is_file()) if exists else 0
  source_key={'LEVIR-MCI':'levir_mci','SECOND-CC':'second_cc'}.get(name,'')
  integrated=bool(pair_counts[source_key])
  rows.append({'source':name,'official_source':official,'classification':kind,'state':state,'raw_path':str(raw) if exists else None,'raw_exists':exists,'disk_bytes':size,'raw_tree_sha256':digest(raw),'adapter_status':'implemented' if (a.output_dir.parent.parent/'code/project-qcpr-dataset-v2/scripts').exists() else 'unknown','canonical_pair_status':'manifested' if integrated else 'not_proven','task_manifest_status':'manifested' if integrated else 'not_proven','loader_batch_validation':'not_run' if not integrated else 'pending_per_source','training_enabled':integrated,'pair_count':pair_counts[source_key],'caption_count':caption_counts[source_key]})
 report={'dataset_v2_root':str(a.dataset_v2_root),'dataset_v2_build_summary_sha256':hashlib.sha256((a.dataset_v2_root/'build_summary.json').read_bytes()).hexdigest(),'sources':rows}
 (a.output_dir/'dataset_source_coverage.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 lines=['# Dataset-v2 Source Coverage','', 'This report records evidence, not planned integration.']
 for x in rows: lines += ['',f"## {x['source']}",f"- State: `{x['state']}`; classification: `{x['classification']}`",f"- Raw: `{x['raw_path']}`; bytes: {x['disk_bytes']}",f"- Training enabled: {x['training_enabled']}; canonical/task manifest: {x['canonical_pair_status']}/{x['task_manifest_status']}"]
 (a.output_dir/'dataset_source_coverage.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
