from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qcpr_single_pass_runtime import evaluate_retrieval
from land_change_detection.models.qcpr_single_pass import QueryConditionedLocalizer
from land_change_detection.models.qcpr_single_pass_factory import build_single_pass_retriever
from land_change_detection.training.qcpr_single_pass_data import MaskFreePairDataset

def arguments():
 p=argparse.ArgumentParser(); p.add_argument("--grounding-checkpoint",type=Path,required=True)
 p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--manifest-dir",type=Path,required=True)
 p.add_argument("--universat-source",default="/mnt/weka/svardanyan/rs_change_project/external/UniverSat")
 p.add_argument("--universat-checkpoint",default="/mnt/weka/svardanyan/rs_change_project/models/universat-base")
 p.add_argument("--jina-model",default="/mnt/weka/svardanyan/rs_change_project/models/jina-v5-text-small-retrieval")
 p.add_argument("--output-grid",type=int,default=32); p.add_argument("--workers",type=int,default=8); return p.parse_args()

def load_mask(path,size=256):
 with Image.open(path) as image:
  array=np.asarray(image.convert("L").resize((size,size),Image.Resampling.NEAREST),dtype=np.float32)
 return torch.from_numpy((array>0).astype(np.float32))

@torch.no_grad()
def localize(model,grounder,captions,images,device):
 pair=model.encode_pairs(images.to(device)); text=model.encode_texts(captions); count=len(captions)
 patches=pair.adapted_dense_tokens
 if patches.shape[0]==1 and count>1: patches=patches.expand(count,-1,-1)
 return grounder(text.text_search_vector.to(device),text.contextual_text_tokens.to(device),
   text.attention_mask.to(device),text.content_mask.to(device),patches,output_size=(256,256))

def main():
 a=arguments(); device=torch.device("cuda"); a.output_dir.mkdir(parents=True,exist_ok=True)
 val_path=a.manifest_dir/"natural_validation_retrieval_manifest.jsonl"; dataset=MaskFreePairDataset(val_path,"val")
 model=build_single_pass_retriever(universat_source=a.universat_source,universat_checkpoint=a.universat_checkpoint,
  jina_model=a.jina_model,device=device,output_grid=a.output_grid)
 payload=torch.load(a.grounding_checkpoint,map_location=device,weights_only=False)
 if payload.get("role")!="grounding": raise RuntimeError("evaluation requires frozen grounding checkpoint")
 model.load_state_dict(payload["model"]); model.eval()
 grounder=QueryConditionedLocalizer(grid_size=a.output_grid).to(device)
 grounder.load_state_dict(payload["localizer"]); grounder.eval()
 metrics,top10=evaluate_retrieval(model,dataset,16,a.workers,device)
 with (a.output_dir/"retrieval_top10.jsonl").open("w") as handle:
  for row in top10: handle.write(json.dumps(row)+"\n")
 sums={key:0. for key in ("soft_dice","soft_iou","pointing_accuracy","mass_inside","map_area_ratio",
                           "query_map_change","pair_map_change","normalized_entropy","concentration",
                           "spatial_variance","fixed_location_count")}
 count=0; examples=[]; argmax_counts=torch.zeros(a.output_grid**2)
 with torch.no_grad():
  for index,item in enumerate(dataset):
   row=dataset.samples[index]; mask_path=row.get("mask_path")
   if not mask_path or not Path(mask_path).exists(): continue
   captions=item.captions[:2] if len(item.captions)>1 else item.captions
   output=localize(model,grounder,captions,item.images.unsqueeze(0),device)
   maps=output.upsampled_soft_map; maps=maps/maps.amax(dim=(1,2),keepdim=True).clamp_min(1e-8)
   prediction=maps[0].cpu(); target=load_mask(mask_path); intersection=(prediction*target).sum()
   sums["soft_dice"]+=float((2*intersection+1e-6)/(prediction.sum()+target.sum()+1e-6))
   sums["soft_iou"]+=float((intersection+1e-6)/(prediction.sum()+target.sum()-intersection+1e-6))
   sums["pointing_accuracy"]+=float(target.flatten()[int(prediction.argmax())]>0)
   sums["mass_inside"]+=float((prediction*target).sum()/prediction.sum().clamp_min(1e-8))
   sums["map_area_ratio"]+=float((prediction>=.5).float().mean())
   sums["normalized_entropy"]+=float(output.diagnostics["entropy"][0]/np.log(a.output_grid**2))
   sums["concentration"]+=float(output.diagnostics["concentration"][0])
   sums["spatial_variance"]+=float(output.diagnostics["spatial_variance"][0])
   argmax_counts[int(output.relevance_probabilities[0].argmax().cpu())]+=1
   if len(captions)>1: sums["query_map_change"]+=float((maps[0]-maps[1]).abs().mean())
   if index+1<len(dataset):
    other=localize(model,grounder,captions[:1],dataset[index+1].images.unsqueeze(0),device)
    other_map=other.upsampled_soft_map; other_map=other_map/other_map.amax().clamp_min(1e-8)
    sums["pair_map_change"]+=float((maps[0]-other_map[0]).abs().mean())
   if len(examples)<50:
    path=a.output_dir/f"map_{index:05d}.pt"
    torch.save({"pair_id":item.pair_id,"query":captions[0],"native_soft_map":output.native_soft_map[0].cpu(),
      "upsampled_map":prediction,"grounded_score":float(output.grounded_score[0]),
      "cross_attention_weights":output.cross_attention_weights[0].cpu()},path); examples.append(str(path))
   count+=1
 localization={key:value/max(count,1) for key,value in sums.items()}
 localization.update({"evaluated_masks":count,"fixed_location_ratio":float(argmax_counts.max()/max(count,1)),
  "mask_use":"evaluation_only","global_pair_bypass":False})
 report={"retrieval":metrics,"localization":localization,"examples":examples,
  "checkpoint_role":payload["role"],"masks_used_for_gradients":False,"masks_used_for_selection":False}
 (a.output_dir/"evaluation.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
