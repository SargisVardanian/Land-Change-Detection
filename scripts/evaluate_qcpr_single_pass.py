from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qcpr_single_pass_runtime import evaluate_retrieval
from land_change_detection.models.qcpr_single_pass import QueryConditionedLocalizer
from land_change_detection.models.qcpr_single_pass_factory import build_single_pass_retriever
from land_change_detection.training.qcpr_single_pass_data import MaskFreePairDataset

def parse():
 p=argparse.ArgumentParser(); p.add_argument("--localization-checkpoint",type=Path,required=True)
 p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--manifest-dir",type=Path,required=True)
 p.add_argument("--universat-source",default="/mnt/weka/svardanyan/rs_change_project/external/UniverSat")
 p.add_argument("--universat-checkpoint",default="/mnt/weka/svardanyan/rs_change_project/models/universat-base")
 p.add_argument("--jina-model",default="/mnt/weka/svardanyan/rs_change_project/models/jina-v5-text-small-retrieval")
 p.add_argument("--output-grid",type=int,default=32); p.add_argument("--workers",type=int,default=8)
 return p.parse_args()

def load_mask(path,size=256):
 with Image.open(path) as im:
  array=np.asarray(im.convert("L").resize((size,size),Image.Resampling.NEAREST),dtype=np.float32)
 return torch.from_numpy((array>0).astype(np.float32))

def main():
 a=parse(); device=torch.device("cuda"); a.output_dir.mkdir(parents=True,exist_ok=True)
 val_path=a.manifest_dir/"natural_validation_retrieval_manifest.jsonl"; dataset=MaskFreePairDataset(val_path,"val")
 model=build_single_pass_retriever(universat_source=a.universat_source,universat_checkpoint=a.universat_checkpoint,
  jina_model=a.jina_model,device=device,output_grid=a.output_grid)
 payload=torch.load(a.localization_checkpoint,map_location=device,weights_only=False)
 if payload.get("role")!="localization": raise RuntimeError("evaluation requires localization checkpoint")
 model.load_state_dict(payload["model"]); model.eval()
 localizer=QueryConditionedLocalizer(grid_size=a.output_grid).to(device); localizer.load_state_dict(payload["localizer"]); localizer.eval()
 metrics,top10=evaluate_retrieval(model,dataset,8,a.workers,device)
 with (a.output_dir/"retrieval_top10.jsonl").open("w") as f:
  for row in top10: f.write(json.dumps(row)+"\n")
 sums={"soft_dice":0.,"soft_iou":0.,"pointing_accuracy":0.,"mass_inside":0.,"map_area_ratio":0.,
       "query_map_change":0.,"pair_map_change":0.}; count=0; examples=[]
 with torch.no_grad():
  for index,item in enumerate(dataset):
   row=dataset.samples[index]; mask_path=row.get("mask_path")
   if not mask_path or not Path(mask_path).exists(): continue
   pair=model.encode_pairs(item.images.unsqueeze(0).to(device))
   captions=item.captions[:2] if len(item.captions)>1 else item.captions
   text=model.encode_texts(captions).text_search_vector.to(device)
   patches=pair.contextual_patch_tokens.expand(len(captions),-1,-1)
   out=localizer(text,patches,output_size=(256,256)); maps=out.upsampled_soft_map
   maps=maps/maps.amax(dim=(1,2),keepdim=True).clamp_min(1e-8); pred=maps[0].cpu(); target=load_mask(mask_path)
   intersection=(pred*target).sum(); denom=pred.sum()+target.sum()
   sums["soft_dice"]+=float((2*intersection+1e-6)/(denom+1e-6))
   sums["soft_iou"]+=float((intersection+1e-6)/(pred.sum()+target.sum()-intersection+1e-6))
   flat=int(pred.argmax()); sums["pointing_accuracy"]+=float(target.flatten()[flat]>0)
   sums["mass_inside"]+=float((pred*target).sum()/pred.sum().clamp_min(1e-8))
   sums["map_area_ratio"]+=float((pred>=.5).float().mean())
   if len(captions)>1: sums["query_map_change"]+=float((maps[0]-maps[1]).abs().mean())
   if index+1<len(dataset):
    other=model.encode_pairs(dataset[index+1].images.unsqueeze(0).to(device))
    other_map=localizer(text[:1],other.contextual_patch_tokens,output_size=(256,256)).upsampled_soft_map
    other_map=other_map/other_map.amax().clamp_min(1e-8)
    sums["pair_map_change"]+=float((maps[0]-other_map[0]).abs().mean())
   if len(examples)<50:
    path=a.output_dir/f"map_{index:05d}.pt"; torch.save({"pair_id":item.pair_id,"query":captions[0],
      "soft_map":out.soft_map[0].cpu(),"upsampled_map":pred,"retrieval_score":None,
      "grounded_score":float(out.grounded_score[0]),"cross_attention_weights":out.cross_attention_weights[0].cpu()},path)
    examples.append(str(path))
   count+=1
 localization={k:v/max(count,1) for k,v in sums.items()}|{"evaluated_masks":count,
  "collapse_probability_std_mean":None,"mask_use":"evaluation_only"}
 report={"retrieval":metrics,"localization":localization,"examples":examples,
  "checkpoint_role":payload["role"],"masks_used_for_gradients":False,"masks_used_for_selection":False}
 (a.output_dir/"evaluation.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
