#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from land_change_detection.training.qcpr_v3_resolution_contract import apply_spatial_contract, foreground_preserving_target, target_aware_crop_box, target_bbox


def stats(mask: torch.Tensor) -> dict:
    binary = mask > 0
    components = int(ndimage.label(binary.numpy())[1])
    bbox = target_bbox(binary)
    edge = torch.zeros_like(binary); edge[[0, -1]] = True; edge[:, [0, -1]] = True
    count = int(binary.sum())
    return {"foreground_pixels": count, "target_area": count / binary.numel(), "cell_equivalents_32": count / binary.numel() * 1024, "connected_components": components, "bbox": bbox, "edge_fraction": float((binary & edge).sum()) / max(count, 1)}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--pair-id",action="append",required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    if len(a.pair_id)!=16: raise ValueError("resolution audit requires exactly 16 fixed pair IDs")
    rows={r["pair_id"]:r for r in map(json.loads,a.manifest.read_text().splitlines())}
    missing=[x for x in a.pair_id if x not in rows]
    if missing: raise ValueError(f"missing pair IDs: {missing}")
    reports=[]; panels=[]
    for pair_id in a.pair_id:
        row=rows[pair_id]
        t1=torch.from_numpy(np.asarray(Image.open(row["t1_path"]).convert("RGB")).copy()).permute(2,0,1).float()/255
        t2=torch.from_numpy(np.asarray(Image.open(row["t2_path"]).convert("RGB")).copy()).permute(2,0,1).float()/255
        raw=torch.from_numpy((np.asarray(Image.open(row["query_mask_path"]).convert("L"))>0).copy()).float()
        if raw.any():
            box=target_aware_crop_box(raw,output_size=256,context=2); c1,c2,resized=apply_spatial_contract(t1,t2,raw,box)
        else:
            box=(0,1024,0,1024); c1,c2,resized=apply_spatial_contract(t1,t2,raw,box)
        canonical=foreground_preserving_target(resized,32); restored=F.interpolate(canonical[None,None],(256,256),mode="nearest")[0,0]
        stages={"raw":stats(raw),"crop_resize_256":stats(resized),"canonical_32":stats(canonical),"restored_256":stats(restored)}
        vanished=max(stages["raw"]["connected_components"]-stages["crop_resize_256"]["connected_components"],0)
        split="core" if stages["raw"]["cell_equivalents_32"]>=4 else "stress"
        reports.append({"pair_id":pair_id,"audit_split":split,"direction":"disappeared" if "disappear" in row["captions"][0] or "demolished" in row["captions"][0] else "appeared","crop_box":box,"stages":stages,"components_disappeared":vanished,"nonempty_survived":bool((not bool(raw.any())) or bool(resized.any()))})
        panels.append((c1,c2,resized,canonical,restored))
    ratios=[r["stages"]["crop_resize_256"]["target_area"] for r in reports]
    status="PASS" if all(r["nonempty_survived"] for r in reports) else "FAIL"
    core=[r for r in reports if r["audit_split"]=="core"]
    output={"schema_version":"qcpr-v3-resolution-audit-v1","preprocessing_geometry":status,"mask_survival_at_256":status,"canonical_grid_resolution":"PASS" if core and all(r["stages"]["canonical_32"]["foreground_pixels"]>=4 for r in core) else "FAIL","stress_rows_excluded_from_core_gate":len(reports)-len(core),"tiny_full_scene_support":"NOT_IMPLEMENTED","target_area_quantiles":{"min":min(ratios),"median":float(np.median(ratios)),"max":max(ratios)},"rows":reports}
    a.output_dir.mkdir(parents=True,exist_ok=False); (a.output_dir/"resolution_audit_report.json").write_text(json.dumps(output,indent=2)+"\n")
    fig,axes=plt.subplots(16,5,figsize=(14,42),squeeze=False)
    for i,(c1,c2,resized,canonical,restored) in enumerate(panels):
        for j,image in enumerate((c1.permute(1,2,0),c2.permute(1,2,0),resized,canonical,restored)): axes[i,j].imshow(image,cmap=None if j<2 else "gray"); axes[i,j].axis("off")
        axes[i,0].set_ylabel(a.pair_id[i],fontsize=7)
    for ax,title in zip(axes[0],("T1 crop","T2 crop","mask 256","canonical 32","restored"),strict=True): ax.set_title(title)
    fig.savefig(a.output_dir/"resolution_audit_panel.png",dpi=140,bbox_inches="tight"); plt.close(fig); print(json.dumps(output,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
