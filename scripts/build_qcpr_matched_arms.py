#!/usr/bin/env python3
"""Build matched causal C0/C1 and N0/N1 mask-free evaluation arms."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

def read(path):
    with Path(path).open() as f: return [json.loads(x) for x in f]

def norm(s): return ' '.join(str(s).casefold().strip(' .').split())

def write(path, rows):
    with Path(path).open('w') as f:
        for x in rows: f.write(json.dumps(x,sort_keys=True)+'\n')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def index_rows(rows):
    out=defaultdict(list)
    for r in rows:
        pid=str(r.get('pair_id') or r.get('canonical_pair_id'))
        texts=r.get('captions') or [r.get('caption','')]
        for text in texts:
            if str(text).strip(): out[(pid,norm(text))].append((r,str(text)))
    return out

def build_shared(p0, p1, scope, outdir, prefix):
    i0,i1=index_rows(p0),index_rows(p1)
    keys=sorted(set(i0)&set(i1))
    pair_keys=defaultdict(list)
    for pid,text in keys: pair_keys[pid].append(text)
    p0map={}; p1map={}
    for pid,texts in pair_keys.items():
        a=i0[(pid,texts[0])][0][0]; b=i1[(pid,texts[0])][0][0]
        row={'pair_id':pid,'dataset_name':a.get('dataset_name',b.get('dataset_name','unknown')),'split':a.get('split',b.get('split')),
             't1_path':a['t1_path'],'t2_path':a['t2_path'],'captions':sorted(set(texts)),
             'source_metadata':dict(a.get('source_metadata',{}))}
        row['source_metadata'].pop('mask_path',None); row['source_metadata'].setdefault('changeflag',1)
        p0map[pid]=row; p1map[pid]=json.loads(json.dumps(row))
    for arm, rows in [('c0',p0map),('c1',p1map)]:
        train=[x for x in rows.values() if x['split']=='train']; dev=[x for x in rows.values() if x['split'] in {'development','val'}]
        for x in train+dev: x['split']='development' if x['split']=='val' else x['split']
        write(outdir/f'{prefix}_{arm}_train.jsonl',train); write(outdir/f'{prefix}_{arm}_development.jsonl',dev)
    by_text=defaultdict(set)
    for pid,text in keys:
        by_text[text].add(pid)
    rel0=[]; rel1=[]
    for n,(pid,text) in enumerate(keys):
        rel0.append({'query_id':f'{prefix}:{n}','pair_id':pid,'text':text,'positive_pair_ids':[pid],'ignored_pair_ids':[],'query_scope':scope})
        other=sorted(by_text[text] - {pid})
        rel1.append({'query_id':f'{prefix}:{n}','pair_id':pid,'text':text,'positive_pair_ids':[pid],'ignored_pair_ids':other,'query_scope':scope})
    write(outdir/f'{prefix}_c0_relevance.jsonl',rel0); write(outdir/f'{prefix}_c1_relevance.jsonl',rel1)
    return {'query_count':len(keys),'pair_count':len(pair_keys),'source_counts':dict(Counter(str(x['dataset_name']) for x in p0map.values())), 'c0_train_pairs':sum(x['split']=='train' for x in p0map.values()), 'c0_development_pairs':sum(x['split']=='development' for x in p0map.values())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--control-root',type=Path,required=True); ap.add_argument('--old-root',type=Path,required=True); ap.add_argument('--v2-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    p0t=read(a.control_root/'p0_train.jsonl'); p0d=read(a.control_root/'p0_development.jsonl'); p1t=read(a.control_root/'p1_train.jsonl'); p1d=read(a.control_root/'p1_development.jsonl')
    exact=build_shared(p0t+p0d,p1t+p1d,'exact_pair',a.output,'common_exact')
    # Semantic and generic rows are drawn from Dataset-v2 task views and matched
    # against the old physical source universe, never against S2Looking.
    old=index_rows(p0t+p0d)
    sem=read(a.v2_root/'manifests/retrieval_semantic_train_v2.jsonl')+read(a.v2_root/'manifests/retrieval_semantic_development_v2.jsonl')
    generic=read(a.v2_root/'manifests/retrieval_exact_train_v2.jsonl')+read(a.v2_root/'manifests/retrieval_exact_development_v2.jsonl')
    def matched_v2(rows, scope):
        out=[]
        for r in rows:
            if r.get('query_scope') != scope: continue
            key=(str(r['canonical_pair_id']),norm(r['caption']))
            if key in old: out.append({'pair_id':key[0],'caption':r['caption'],'t1_path':r['t1_path'],'t2_path':r['t2_path'],'dataset_name':r['dataset_name'],'split':'development' if r['split']!='train' and r['split']!='val' else ('val' if r['split']=='val' else 'train'),'source_metadata':{'changeflag':0 if scope=='generic_no_change' else 1},'positive_pair_ids':r.get('positive_pair_ids',[key[0]]),'ignored_pair_ids':r.get('ignored_pair_ids',[])})
        return out
    def build_query_view(rows,prefix,scope):
        bypair=defaultdict(dict)
        for r in rows: bypair[r['pair_id']][norm(r['caption'])]=r
        physical={}
        for pid, texts in bypair.items():
            r=next(iter(texts.values())); physical[pid]={'pair_id':pid,'dataset_name':r['dataset_name'],'t1_path':r['t1_path'],'t2_path':r['t2_path'],'captions':[x['caption'] for x in texts.values()],'split':r['split'],'source_metadata':r['source_metadata']}
        for arm in ('n0','n1'):
            for split in ('train','development'):
                write(a.output/f'{prefix}_{arm}_{split}.jsonl',[dict(x,split=split) for x in physical.values() if x['split']==split])
        rel=[]
        for n,r in enumerate(rows):
            same=sorted({x['pair_id'] for x in rows if norm(x['caption'])==norm(r['caption'])})
            rel.append({'query_id':f'{prefix}:{n}','pair_id':r['pair_id'],'text':r['caption'],'positive_pair_ids':[r['pair_id']], 'ignored_pair_ids':[] if prefix=='common_semantic' else [x for x in same if x!=r['pair_id']], 'query_scope':scope})
        write(a.output/f'{prefix}_n0_relevance.jsonl',rel); write(a.output/f'{prefix}_n1_relevance.jsonl',rel)
        return {'query_count':len(rows),'pair_count':len(physical),'source_counts':dict((s,sum(x['dataset_name']==s for x in physical.values())) for s in sorted({x['dataset_name'] for x in physical.values()}))}
    sem_result=build_query_view(matched_v2(sem,'semantic_group'),'common_semantic','semantic_group')
    generic_result=build_query_view(matched_v2(generic,'generic_no_change'),'common_generic','generic_no_change')
    summary={'schema_version':'qcpr-matched-causal-v1','common_exact':exact,'common_semantic':sem_result,'common_generic':generic_result,'notes':['C0/C1 share pair IDs, paths and query text for each generated view.','N0/N1 generic no-change is diagnostic/semantic only, never exact-pair primary supervision.','Exposure schedule is fixed by the matched pair/query ID list and must be shared by the trainer.']}
    (a.output/'matched_contract.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
