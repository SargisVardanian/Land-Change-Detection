#!/usr/bin/env python3
"""Controlled P0/P1 entrypoint with a shared no-hard-mining contract."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--arm',choices=['p0','p1','c0','c1'],required=True); ap.add_argument('--train-manifest',type=Path,required=True); ap.add_argument('--development-manifest',type=Path,required=True); ap.add_argument('--relevance-manifest',type=Path,required=True); ap.add_argument('--initial-checkpoint',type=Path,required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--steps',type=int,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--manifest-root',type=Path,required=True); ap.add_argument('--baseline-reproduction',type=Path,required=True); ap.add_argument('--identifiability-audit',type=Path,required=True); ap.add_argument('--workers',type=int,default=8); a=ap.parse_args()
    if a.steps <= 0 or a.steps % 1: raise ValueError('steps must be a positive fixed integer')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    runtime=a.output_dir/'runtime_manifest'; runtime.mkdir()
    for name,src in [('natural_train_retrieval_manifest.jsonl',a.train_manifest),('natural_validation_retrieval_manifest.jsonl',a.development_manifest),('caption_quality_audit.jsonl',a.manifest_root/f'{a.arm}_collision_audit.jsonl')]:
        os.symlink(src.resolve(),runtime/name)
    (a.output_dir/'control_contract.json').write_text(json.dumps({'schema_version':'qcpr-p0-p1-control-v1','arm':a.arm,'seed':a.seed,'fixed_steps':a.steps,'physical_microbatch':16,'logical_physical_batch':128,'captions_per_pair':2,'logical_score_matrix':'256x128','gradcache':True,'hard_negative_mining':False,'initial_checkpoint':str(a.initial_checkpoint),'initial_checkpoint_sha256':sha(a.initial_checkpoint),'train_manifest_sha256':sha(a.train_manifest),'development_manifest_sha256':sha(a.development_manifest),'relevance_manifest_sha256':sha(a.relevance_manifest)},indent=2,sort_keys=True)+'\n')
    trainer=Path(__file__).with_name('train_qcpr_retrieval_repair_r1.py')
    cmd=[sys.executable,str(trainer),'--baseline-checkpoint',str(a.initial_checkpoint),'--reproduction',str(a.baseline_reproduction),'--identifiability-audit',str(a.identifiability_audit),'--output-dir',str(a.output_dir/'r1_runtime'),'--manifest-dir',str(runtime),'--physical-micro-batch','16','--logical-physical-batch','128','--captions-per-pair','2','--hard-warmup-epochs','999999','--hard-refresh-epochs','999999','--max-steps',str(a.steps),'--disable-hard-mining','--seed',str(a.seed),'--workers',str(a.workers)]
    subprocess.run(cmd,check=True)
    summary={'arm':a.arm,'status':'COMPLETED','control_contract':json.loads((a.output_dir/'control_contract.json').read_text()),'runtime_output':str(a.output_dir/'r1_runtime')}
    if (a.output_dir/'r1_runtime/r1_acceptance.json').exists(): summary['acceptance']=json.loads((a.output_dir/'r1_runtime/r1_acceptance.json').read_text())
    if (a.output_dir/'r1_runtime/exposure_accounting.json').exists(): summary['exposure']=json.loads((a.output_dir/'r1_runtime/exposure_accounting.json').read_text())
    (a.output_dir/'control_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
