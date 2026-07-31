#!/usr/bin/env python3
"""Executable preflight proving C0/C1 differ only in relevance masks."""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import torch
from land_change_detection.models.qcpr_single_pass import build_pair_masks, multi_positive_sigmoid_loss

def read(p):
    with Path(p).open() as f: return [json.loads(x) for x in f if x.strip()]
def norm(s): return ' '.join(str(s).casefold().strip(' .').split())
def digest(items): return hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); a=ap.parse_args(); r=a.root
    c0t=read(r/'c0_train.jsonl'); c1t=read(r/'c1_train.jsonl'); c0d=read(r/'c0_development.jsonl'); c1d=read(r/'c1_development.jsonl')
    assert [(x['pair_id'],x['t1_path'],x['t2_path']) for x in c0t+c0d] == [(x['pair_id'],x['t1_path'],x['t2_path']) for x in c1t+c1d]
    c0=read(r/'c0_relevance.jsonl'); c1=read(r/'c1_relevance.jsonl')
    assert [(x['query_id'],x['text']) for x in c0] == [(x['query_id'],x['text']) for x in c1]
    assert all(len(x['positive_pair_ids'])==1 and not x.get('ignored_pair_ids') for x in c0)
    assert all(x['pair_id'] not in x['ignored_pair_ids'] for x in c1)
    collision=next((x for x in c1 if x.get('ignored_pair_ids')),None)
    if collision is None: raise AssertionError('no effective relevance difference: no audited collision')
    pairs=sorted({x['pair_id'] for x in c0t+c0d})
    collision_pair=collision['pair_id']; ignored_pair=collision['ignored_pair_ids'][0]
    ordered=[collision_pair,ignored_pair]+[x for x in pairs if x not in {collision_pair,ignored_pair}]
    gallery=ordered[:128]
    bypair=defaultdict(list)
    for x in c0: bypair[x['pair_id']].append(x)
    c1_by_pair=defaultdict(list)
    for x in c1: c1_by_pair[x['pair_id']].append(x)
    queries=[]
    for p in gallery:
        candidates=sorted(bypair[p],key=lambda x:norm(x['text']))
        if p == collision_pair:
            collision_row=next(x for x in c1_by_pair[p] if x.get('ignored_pair_ids'))
            candidates=[next(x for x in candidates if norm(x['text'])==norm(collision_row['text']))] + [x for x in candidates if norm(x['text'])!=norm(collision_row['text'])]
        queries.extend(candidates[:2])
    assert len(queries)==256
    qids=[x['pair_id'] for x in queries]
    collisions0=[set() for _ in queries]
    c1_by_query={x['query_id']:set(x.get('ignored_pair_ids',[])) for x in c1}
    collisions1=[c1_by_query[x['query_id']] for x in queries]
    pos0,ex0=build_pair_masks(qids,gallery,collisions0)
    pos1,ex1=build_pair_masks(qids,gallery,collisions1)
    assert torch.equal(pos0,pos1) and torch.equal(ex0,torch.zeros_like(ex0))
    assert torch.equal(pos1, pos0) and int(ex1.sum())>0
    logits=torch.arange(256*128,dtype=torch.float32).reshape(256,128)/100.0
    loss0,_=multi_positive_sigmoid_loss(logits,pos0,ex0)
    loss1,_=multi_positive_sigmoid_loss(logits,pos1,ex1)
    if torch.equal(ex0,ex1) or torch.equal(loss0,loss1): raise AssertionError('C0/C1 loss did not differ')
    schedule=[{'pair_id':p,'query_ids':[x['query_id'] for x in bypair[p][:2]]} for p in gallery]
    result={'passed':True,'gallery_size':128,'logical_query_count':256,'pair_ids_identical':True,'query_ids_identical':True,'query_texts_identical':True,'image_paths_identical':True,'sampling_schedule_sha256':digest(schedule),'c0_ignored_entries':int(ex0.sum()),'c1_ignored_entries':int(ex1.sum()),'collision_query_id':collision['query_id'],'collision_pair_id':collision_pair,'ignored_pair_id':ignored_pair,'c0_loss':float(loss0),'c1_loss':float(loss1),'loss_difference':float(abs(loss0-loss1)),'semantic_matched_query_count':0}
    print(json.dumps(result,indent=2,sort_keys=True))
    (r/'matched_mask_contract_test.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
