import argparse, hashlib, json, shutil
from pathlib import Path
def sha(s): return hashlib.sha256(s.encode()).hexdigest()[:16]
def copy(src,dst): shutil.copy2(src,dst)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--matched',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
 for arm in ['c0','c1']:
  for split in ['train','development']:
   copy(a.matched/f'common_exact_{arm}_{split}.jsonl',a.output/f'{arm}_{split}.jsonl')
  copy(a.matched/f'common_exact_{arm}_relevance.jsonl',a.output/f'{arm}_relevance.jsonl')
  rel=json.loads('[]') if False else None
  # CaptionCollisionIndex consumes one compact row per pair/text cluster; it
  # does not need the expanded ignored_pair_ids list.
  with (a.matched/f'common_exact_{arm}_relevance.jsonl').open() as src, (a.output/f'{arm}_collision_audit.jsonl').open('w') as dst:
   seen=set()
   for line in src:
    r=json.loads(line); ignored=r.get('ignored_pair_ids',[])
    if arm=='c1' and ignored:
     key=(r['pair_id'],' '.join(r['text'].casefold().strip(' .').split()))
     if key not in seen:
      seen.add(key); dst.write(json.dumps({'pair_id':r['pair_id'],'normalized_caption':key[1],'duplicate_caption_cluster':'exact:'+sha(key[1])},sort_keys=True)+'\n')
   if arm=='c0': pass
 (a.output/'matched_control_contract.json').write_text(json.dumps({'schema_version':'qcpr-matched-control-v1','arms':['c0','c1'],'same_physical_and_query_manifests':True,'c0_relevance':'single_positive','c1_relevance':'multi_positive_with_ignored_ambiguities','logical_batch':128,'physical_microbatch':16,'score_matrix':'256x128','captions_per_pair':2},indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
