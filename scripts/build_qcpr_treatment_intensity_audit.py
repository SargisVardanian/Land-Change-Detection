#!/usr/bin/env python3
"""Pre-result treatment-intensity and deterministic schedule audit."""
import argparse, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path
from land_change_detection.training.qcpr_retrieval_repair import hard_aware_epoch_order

def read(p):
    with Path(p).open() as f: return [json.loads(x) for x in f if x.strip()]
def norm(s): return ' '.join(str(s).casefold().strip(' .').split())
def h(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--control-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seed',type=int,default=20260728); ap.add_argument('--steps',type=int,default=348); ap.add_argument('--logical-batch',type=int,default=128); ap.add_argument('--captions-per-pair',type=int,default=2); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    c0=read(a.control_root/'c0_relevance.jsonl'); c1=read(a.control_root/'c1_relevance.jsonl'); assert [(x['query_id'],x['text']) for x in c0]==[(x['query_id'],x['text']) for x in c1]
    affected=[x for x in c1 if x.get('ignored_pair_ids')]; unaffected=[x for x in c1 if not x.get('ignored_pair_ids')]
    def write_ids(name,items):
        with (a.output/name).open('w') as f:
            for x in items: f.write(json.dumps({'query_id':x['query_id'],'pair_id':x['pair_id'],'text':x['text'],'query_scope':x.get('query_scope'),'ignored_pair_count':len(x.get('ignored_pair_ids',[]))},sort_keys=True)+'\n')
    write_ids('affected_query_ids.jsonl',affected); write_ids('unaffected_query_ids.jsonl',unaffected)
    cluster_sizes=Counter(len(x.get('ignored_pair_ids',[]))+1 for x in affected)
    audit={'treatment_definition':'affected = C1 query has at least one ignored ambiguity/duplicate candidate; unaffected = no ignored candidates','total_queries':len(c1),'affected_queries':len(affected),'unaffected_queries':len(unaffected),'affected_rate':len(affected)/max(len(c1),1),'affected_pairs':len({x['pair_id'] for x in affected}),'unaffected_pairs':len({x['pair_id'] for x in unaffected}),'ignored_entries':sum(len(x.get('ignored_pair_ids',[])) for x in affected),'cluster_size_distribution':dict(sorted(cluster_sizes.items())),'affected_by_source':dict(Counter(x['pair_id'].split(':')[0] for x in affected)),'unaffected_by_source':dict(Counter(x['pair_id'].split(':')[0] for x in unaffected)),'semantic_matched_query_count':0,'notes':['Affected/unaffected IDs are frozen before model results.','Generic no-change is not promoted to exact treatment; it remains a separate diagnostic view.']}
    (a.output/'treatment_intensity_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    pairs=read(a.control_root/'c0_train.jsonl'); pairs.sort(key=lambda x:str(x['pair_id'])); pair_ids=[x['pair_id'] for x in pairs]; captions={x['pair_id']:x['captions'] for x in pairs}; index={p:i for i,p in enumerate(pair_ids)}
    # Add development pairs only if this is accidentally run on a combined root.
    if len(pair_ids)<a.logical_batch: raise ValueError('train root smaller than logical batch')
    step_pairs=[]; step_queries=[]; step_hashes=[]; global_step=0; epoch=0; batches_per_epoch=(len(pair_ids)+a.logical_batch-1)//a.logical_batch
    while global_step<a.steps:
        order,_=hard_aware_epoch_order(pair_ids,logical_batch=a.logical_batch,seed=a.seed+epoch,hard_by_pair=None)
        for start in range(0,len(order),a.logical_batch):
            if global_step>=a.steps: break
            ids=[pair_ids[i] for i in order[start:start+a.logical_batch]]
            qs=[]
            for pid in ids:
                order_idx=list(range(len(captions[pid]))); random.Random(f'{a.seed}:{pid}').shuffle(order_idx); begin=(epoch*a.captions_per_pair)%len(order_idx); chosen=[order_idx[(begin+j)%len(order_idx)] for j in range(a.captions_per_pair)]
                qs.extend([f'{pid}:{norm(captions[pid][i])}' for i in chosen])
            step_pairs.append(ids); step_queries.append(qs); step_hashes.append({'step':global_step+1,'epoch':epoch+1,'pair_sequence_sha256':h(ids),'query_sequence_sha256':h(qs),'schedule_sha256':h({'epoch':epoch+1,'step':global_step+1,'pairs':ids,'queries':qs})}); global_step+=1
        epoch+=1
    expected={'steps':global_step,'logical_batch':a.logical_batch,'captions_per_pair':a.captions_per_pair,'pair_sequence_sha256':h(step_pairs),'query_sequence_sha256':h(step_queries),'schedule_sequence_sha256':h(step_hashes),'per_step':step_hashes}
    (a.output/'expected_schedule_hashes.json').write_text(json.dumps(expected,indent=2,sort_keys=True)+'\n')
    md=['# Treatment-intensity audit','',f"Affected queries: **{len(affected)} / {len(c1)} ({audit['affected_rate']:.4%})**",f"Unaffected queries: **{len(unaffected)}**",f"Ignored ambiguity entries: **{audit['ignored_entries']}**",'', 'Affected and unaffected query IDs were frozen before training results.', '', 'Semantic matched query count: **0**.', '', f"Expected schedule hashes were generated for {global_step} logical steps with a 256×128 score matrix contract."]
    (a.output/'treatment_intensity_audit.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'audit':audit,'expected_schedule':{k:v for k,v in expected.items() if k!='per_step'}},indent=2,sort_keys=True))
if __name__=='__main__': main()
