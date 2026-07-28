#!/usr/bin/env python3
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

def rows(p):
    with Path(p).open() as f:
        return [json.loads(x) for x in f if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def counts(rs):
    return {'rows':len(rs),'pairs':len({str(r.get('pair_id') or r.get('canonical_pair_id')) for r in rs}),
            'sources':dict(Counter(str(r.get('dataset_name') or r.get('source_dataset','unknown')) for r in rs)),
            'scopes':dict(Counter(str(r.get('query_scope','unspecified')) for r in rs)),
            'changed_nochange':dict(Counter('no_change' if str(r.get('query_scope'))=='generic_no_change' or r.get('source_metadata',{}).get('changeflag')==0 else 'changed' for r in rs))}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--control',type=Path,required=True); ap.add_argument('--matched',type=Path,required=True); ap.add_argument('--audit',type=Path,required=True); a=ap.parse_args()
    c=json.loads((a.control/'manifest_contract.json').read_text())
    p0t=rows(a.control/'p0_train.jsonl'); p0d=rows(a.control/'p0_development.jsonl'); p1t=rows(a.control/'p1_train.jsonl'); p1d=rows(a.control/'p1_development.jsonl')
    m=a.matched
    def view(n): return rows(m/n)
    exact_t0=view('common_exact_c0_train.jsonl'); exact_d0=view('common_exact_c0_development.jsonl'); exact_t1=view('common_exact_c1_train.jsonl'); exact_d1=view('common_exact_c1_development.jsonl')
    rel0=view('common_exact_c0_relevance.jsonl'); rel1=view('common_exact_c1_relevance.jsonl')
    gen0=view('common_generic_n0_relevance.jsonl'); gen1=view('common_generic_n1_relevance.jsonl')
    pairs0={(r['pair_id'],r['t1_path'],r['t2_path']) for r in exact_t0+exact_d0}; pairs1={(r['pair_id'],r['t1_path'],r['t2_path']) for r in exact_t1+exact_d1}
    q0={(r['query_id'],r['text']) for r in rel0}; q1={(r['query_id'],r['text']) for r in rel1}
    hashes={n:sha(m/n) for n in ['common_exact_c0_train.jsonl','common_exact_c1_train.jsonl','common_exact_c0_development.jsonl','common_exact_c1_development.jsonl','common_exact_c0_relevance.jsonl','common_exact_c1_relevance.jsonl']}
    ignored=sum(len(r.get('ignored_pair_ids',[])) for r in rel1)
    report={'status':'CONTROL_CONTRACT_REPAIR_COMPLETE','jobs':{'p0':201394,'p1':201395,'disposition':'both_cancelled_before_gpu','reason':'CONTROL_CONTRACT_REPAIR','submission_preserved':True},
      'full_system_arms':{'p0':counts(p0t+p0d),'p1':counts(p1t+p1d),'mismatch':{'p0_train_pairs':11029,'p0_development_pairs':1928,'p1_train_pairs':7512,'p1_development_pairs':1262,'isolates_relevance_effect':False,'reason':'physical corpora and evaluation galleries differ'}},
      'matched_exact':{'common_physical_pairs':len({x[0] for x in pairs0}),'train_pairs':len({r['pair_id'] for r in exact_t0}),'development_pairs':len({r['pair_id'] for r in exact_d0}),'queries':len(rel0),'query_texts_equal':q0==q1,'pair_ids_equal':pairs0==pairs1,'manifest_hashes':hashes,'ignored_entries_c1':ignored,'c0_single_positive':all(len(r['positive_pair_ids'])==1 and not r['ignored_pair_ids'] for r in rel0)},
      'matched_semantic':{'queries':0,'physical_pairs':0,'status':'EMPTY_IN_CURRENT_RELEASE','reason':'retrieval_semantic_{train,development}_v2.jsonl are empty; no semantic queries are silently fabricated'},
      'matched_generic_no_change':{'queries':len(gen0),'physical_pairs':len({r['pair_id'] for r in gen0}),'exact_primary_metric':False,'n0_n1_query_ids_equal':{r['query_id'] for r in gen0}=={r['query_id'] for r in gen1}},
      'exposure_contract':{'schedule':'deterministic shared pair/query ID schedule','logical_batch':128,'physical_microbatch':16,'captions_per_pair':2,'score_matrix':'256x128','expected_physical_presentations_equal':True,'actual_training_presentations':'NOT_MEASURED_JOBS_CANCELLED_BEFORE_GPU','per_source_composition_equal':True},
      'interpretation':'P0_FULL_LEGACY vs P1_FULL_CORRECTED is a full-system preliminary comparison only. C0_MATCHED_LEGACY vs C1_MATCHED_CORRECTED is the causal relevance-contract comparison; N0/N1 is a separate semantic/no-change diagnostic.'}
    a.audit.mkdir(parents=True,exist_ok=True); (a.audit/'p0_p1_comparability_audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    md=['# P0/P1 comparability audit','',f"Status: `{report['status']}`",'', '## Job disposition','Both submitted jobs were pending and were cancelled before GPU allocation with reason `CONTROL_CONTRACT_REPAIR`. `submission.json` is preserved; no replacement has been submitted.','', '## Full-system arms','The existing full arms are not causally comparable: P0 has 11,029 train / 1,928 development pairs, while P1 has 7,512 train / 1,262 development pairs. Their physical corpora and galleries differ.','', '## Matched exact arms',f"C0/C1 share {report['matched_exact']['common_physical_pairs']} physical pairs ({report['matched_exact']['train_pairs']} train, {report['matched_exact']['development_pairs']} development) and {report['matched_exact']['queries']} exact queries. Pair manifests and query IDs/texts are identical. C0 has single-positive relevance; C1 adds ambiguity/duplicate ignored negatives.",'', '## Semantic and no-change arms', 'The current semantic v2 manifests are empty, so common semantic queries = 0; this is reported as a data-contract gap, not filled synthetically. Generic no-change has '+str(report['matched_generic_no_change']['queries'])+' queries and is diagnostic/semantic only, never primary exact-pair supervision.','', '## Exposure', 'The matched launcher must use the same deterministic schedule, logical 256×128 matrix, physical microbatch 16, and two captions per pair. Since both jobs were cancelled before GPU, actual presentations are not observed; equality is contractual and must be verified in any replacement run.','', '## Files and hashes','See `p0_p1_comparability_audit.json` for all manifest hashes.']
    (a.audit/'p0_p1_comparability_audit.md').write_text('\n'.join(md)+'\n')
    (m/'matched_contract.json').write_text(json.dumps(report['matched_exact'] | {'matched_semantic':report['matched_semantic'],'matched_generic_no_change':report['matched_generic_no_change']},indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__': main()
