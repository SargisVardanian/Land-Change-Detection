#!/usr/bin/env python3
"""Build an expanded *draft* release without claiming unavailable sources."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from land_change_detection.data.qcpr_dataset_v2 import frame, jsonl_read, jsonl_write, legacy_pair_to_v2, build_relevance, stable_manifest_hashes

def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--core-root',type=Path,required=True); p.add_argument('--s2-manifest',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); a=p.parse_args()
 core_pairs=jsonl_read(a.core_root/'registries/pair_registry.jsonl'); core_caps=jsonl_read(a.core_root/'registries/caption_registry.jsonl'); core_rel=jsonl_read(a.core_root/'registries/relevance_registry.jsonl'); core_dense=jsonl_read(a.core_root/'registries/dense_label_registry.jsonl')
 s2_pairs=[]; s2_caps=[]; s2_dense=[]
 for row in jsonl_read(a.s2_manifest):
  pair_id=str(row['pair_id']); pair=legacy_pair_to_v2({'pair_id':pair_id,'dataset_name':'s2looking','split':row.get('split','unknown'),'t1_path':row.get('t1_path'),'t2_path':row.get('t2_path'),'source_metadata':row.get('source_metadata') or {'scene_id':row.get('original_id',pair_id)}}); pair.update({'source_version':'s2looking_manifest','license':'dataset_terms','is_synthetic':False,'ordered_pair_hash':pair_id,'order_invariant_pair_hash':pair_id}); s2_pairs.append(pair)
  for i,d in enumerate(row.get('directional_targets') or []):
   text=str(d.get('caption','')).strip()
   if not text: continue
   cid=f'{pair_id}:s2_caption:{i}'; s2_caps.append({'caption_id':cid,'canonical_pair_id':pair_id,'text':text,'normalized_text':' '.join(text.casefold().split()),'caption_source':row.get('caption_source','derived_unverified'),'task_type':'grounding','query_scope':'semantic_group','semantic_group_id':f'{pair_id}:direction:{d.get("direction",i)}','equivalent_caption_group_id':None,'quality_score':float(d.get('caption_confidence',0.0)),'identifiability_score':0.0,'verification_status':'derived_not_human_reviewed','is_generated':True,'generator':'S2Looking mask attributes','change_status':'changed','dataset_name':'s2looking','retrieval_supervision':False})
   if d.get('mask_path'): s2_dense.append({'canonical_pair_id':pair_id,'label_type':'query_specific_change_mask','mask_path':d['mask_path'],'caption_id':cid,'source_dataset':'s2looking'})
 out=a.output_root; (out/'registries').mkdir(parents=True,exist_ok=True); (out/'reports').mkdir(parents=True,exist_ok=True)
 pairs=core_pairs+s2_pairs; caps=core_caps+s2_caps; dense=core_dense+s2_dense; rel=core_rel+build_relevance(s2_caps)
 jsonl_write(out/'registries/pair_registry.jsonl',pairs); jsonl_write(out/'registries/caption_registry.jsonl',caps); jsonl_write(out/'registries/relevance_registry.jsonl',rel); jsonl_write(out/'registries/dense_label_registry.jsonl',dense)
 by={p['canonical_pair_id']:p for p in pairs}; cap_by={}
 for c in caps: cap_by.setdefault(c['canonical_pair_id'],[]).append(c)
 for split in ('train','development','test'):
  rows=[]
  for p in pairs:
   role='development' if p['split'] in {'val','validation','dev','development'} else p['split']
   if role!=split: continue
   for c in cap_by.get(p['canonical_pair_id'],[]):
    if c.get('retrieval_supervision',True) is False: continue
    rows.append({'canonical_pair_id':p['canonical_pair_id'],'caption_id':c['caption_id'],'caption':c['text'],'query_scope':c['query_scope'],'positive_pair_ids':[p['canonical_pair_id']],'ignored_pair_ids':[],'t1_path':p['t1_path'],'t2_path':p['t2_path'],'dataset_name':p['source_dataset'],'split':split})
  jsonl_write(out/f'retrieval_{split}_v2_expanded.jsonl',rows)
  grounding=list(rows)
  for p in pairs:
   role='development' if p['split'] in {'val','validation','dev','development'} else p['split']
   if role!=split or p['source_dataset']!='s2looking': continue
   for c in cap_by.get(p['canonical_pair_id'],[]): grounding.append({'canonical_pair_id':p['canonical_pair_id'],'caption_id':c['caption_id'],'caption':c['text'],'query_scope':c['query_scope'],'positive_pair_ids':[],'ignored_pair_ids':[],'t1_path':p['t1_path'],'t2_path':p['t2_path'],'dataset_name':'s2looking','split':split})
  jsonl_write(out/f'grounding_{split}_mask_free_v2_expanded.jsonl',grounding)
 jsonl_write(out/'dense_evaluation_v2_expanded.jsonl',s2_dense); jsonl_write(out/'scene_language_pretrain_v2_expanded.jsonl',[])
 summary={'status':'EXPANDED_DRAFT_NOT_READY','sources_included':['LEVIR-MCI','SECOND-CC','S2Looking_grounding_dense'],'sources_blocked':['ChangeChat-87k','Synthetic_RCD_SECOND_original_A_mapping','SYSU-CD','Hi-UCD','RSCC','auxiliary_scene_language'],'pairs':len(pairs),'captions':len(caps),'dense_labels':len(dense),'manifest_hashes':stable_manifest_hashes(out),'registry_hashes':stable_manifest_hashes(out/'registries')}
 (out/'reports/dataset_v2_expanded_build_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
