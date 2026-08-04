#!/usr/bin/env python3
"""Record frozen architecture screening and explicit availability gates."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); sha=subprocess.run(["git","rev-parse","HEAD"],cwd=a.repo,check=True,text=True,capture_output=True).stdout.strip()
    rows=[{"id":"B0","name":"current UniverSat joint temporal + Jina","state":"AVAILABLE_LOCAL","selection_role":"anchor_candidate"},{"id":"B1","name":"UniverSat framewise + gated difference fusion + Jina","state":"IMPLEMENTATION_REQUIRED","selection_role":"alternative_candidate"},{"id":"B2","name":"UniverSat framewise + text-conditioned temporal fusion","state":"IMPLEMENTATION_REQUIRED","selection_role":"alternative_candidate"},{"id":"B3","name":"RemoteCLIP framewise + gated temporal fusion","state":"WEIGHTS_OR_PREPROCESSING_AUDIT_REQUIRED","selection_role":"alternative_candidate"},{"id":"B4","name":"LRSCLIP/DGTRS framewise + gated temporal fusion","state":"WEIGHTS_OR_PREPROCESSING_AUDIT_REQUIRED","selection_role":"alternative_candidate"}]
    payload={"schema_version":"qcpr-stage2-architecture-screen-v1","code_sha":sha,"common_benchmark":"current audited Dataset-v2 development; replace with Stage-2 expanded real-only development when ready","frozen_only":True,"full_finetuning":False,"metrics":["exact_mrr","exact_recall_at_1","exact_recall_at_5","exact_recall_at_10","semantic_map","semantic_ndcg","per_source","per_gsd","latency","peak_vram"],"architectures":rows,"selection_rule":"select anchor only after actual frozen evaluation; do not use external headline results","status":"SCREENING_PENDING"}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps({"output":str(a.output),"architectures":len(rows),"status":payload["status"]},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
