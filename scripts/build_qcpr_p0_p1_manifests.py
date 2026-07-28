#!/usr/bin/env python3
"""Build immutable, mask-free P0/P1 controlled retrieval manifests."""
from __future__ import annotations
import argparse, hashlib, json
from collections import OrderedDict
from pathlib import Path

def read(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def grouped(rows, split, accepted_splits=None):
    accepted_splits = set(accepted_splits or [split])
    out=OrderedDict()
    for r in rows:
        if r.get('split') not in accepted_splits: continue
        pid=str(r.get('pair_id') or r.get('canonical_pair_id'))
        if pid not in out:
            out[pid]={'pair_id':pid,'dataset_name':r.get('dataset_name','unknown'),
                      'split':split,'t1_path':r['t1_path'],'t2_path':r['t2_path'],
                      'captions':[],'source_metadata':dict(r.get('source_metadata',{}))}
        text=r.get('caption')
        texts=[text] if text else list(r.get('captions',[]))
        for text in texts:
            if text and text not in out[pid]['captions']: out[pid]['captions'].append(text)
        if 'query_scope' in r: out[pid].setdefault('_scopes',[]).append(r['query_scope'])
    result=[]
    for row in out.values():
        row['source_metadata'].pop('mask_path',None)
        row['source_metadata']['changeflag']=int(row['source_metadata'].get('changeflag',0 if any(x=='generic_no_change' for x in row.pop('_scopes',[])) else 1))
        result.append(row)
    return result

def write_jsonl(path, rows):
    with Path(path).open('w') as f:
        for row in rows: f.write(json.dumps(row,sort_keys=True)+'\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--old-root',type=Path,required=True); ap.add_argument('--v2-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    p0_train=grouped([r for r in read(a.old_root/'natural_train_manifest.jsonl') if r.get('dataset_name') in {'levir_mci','second_cc'}],'train')
    p0_dev=grouped([r for r in read(a.old_root/'natural_validation_manifest.jsonl') if r.get('dataset_name') in {'levir_mci','second_cc'}],'development',{'val','development'})
    exact_train=[r for r in read(a.v2_root/'manifests/retrieval_exact_train_v2.jsonl') if r.get('query_scope')=='exact_pair']
    exact_dev=[r for r in read(a.v2_root/'manifests/retrieval_exact_development_v2.jsonl') if r.get('query_scope')=='exact_pair']
    p1_train=grouped(exact_train,'train'); p1_dev=grouped(exact_dev,'development')
    for name, rows in [('p0_train.jsonl',p0_train),('p0_development.jsonl',p0_dev),('p1_train.jsonl',p1_train),('p1_development.jsonl',p1_dev)]: write_jsonl(a.output/name,rows)
    for arm, rows in [('p0',p0_train+p0_dev),('p1',p1_train+p1_dev)]:
        rel=[]
        for r in rows:
            for i,text in enumerate(r['captions']):
                rel.append({'query_id':f"{r['pair_id']}:{i}",'pair_id':r['pair_id'],'positive_pair_ids':[r['pair_id']], 'ignored_pair_ids':[], 'query_scope':'exact_pair','text':text})
        write_jsonl(a.output/f'{arm}_relevance.jsonl',rel)
    # Collision audit is consumed by the existing ambiguity-aware loss path.
    # P0 preserves the legacy empty collision contract; P1 carries ignored IDs
    # from the corrected exact-pair rows without exposing masks.
    collisions=[]
    for raw in exact_train + exact_dev:
        ignored=raw.get('ignored_pair_ids',[])
        if ignored:
            collisions.append({'pair_id':raw['canonical_pair_id'],'normalized_caption':' '.join(raw['caption'].casefold().strip(' .').split()),'duplicate_caption_cluster':raw['caption_id'],'ignored_pair_ids':ignored})
    write_jsonl(a.output/'p0_collision_audit.jsonl',[])
    write_jsonl(a.output/'p1_collision_audit.jsonl',collisions)
    summary={'schema_version':'qcpr-p0-p1-control-v1','status':'READY_FOR_GATES','p0':{},'p1':{},'common_contract':{'physical_microbatch':16,'logical_physical_batch':128,'captions_per_pair':2,'logical_score_matrix':'256x128','gradcache':True,'hard_negative_mining':False}}
    for arm, files in [('p0',['p0_train.jsonl','p0_development.jsonl','p0_relevance.jsonl']),('p1',['p1_train.jsonl','p1_development.jsonl','p1_relevance.jsonl'])]:
        chosen = p0_train+p0_dev if arm=='p0' else p1_train+p1_dev
        summary[arm]={'files':{x:{'rows':sum(1 for _ in (a.output/x).open()),'sha256':sha(a.output/x)} for x in files},'physical_pairs':len(chosen),'sources':sorted(set(x['dataset_name'] for x in chosen))}
    (a.output/'manifest_contract.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
