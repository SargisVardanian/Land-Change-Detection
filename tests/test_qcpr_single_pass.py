from __future__ import annotations
import json
from pathlib import Path
import torch
from PIL import Image
from torch import nn
from land_change_detection.backbones.jina_v5_text import TextFeatures
from land_change_detection.models.qcpr_single_pass import (
 BiTemporalPairTransformer,QCPRSinglePassRetriever,QueryConditionedLocalizer,
 SinglePassConfig,build_pair_masks,multi_positive_sigmoid_loss)
from land_change_detection.training.qcpr_single_pass_data import MaskFreePairDataset,PairCaptionCollator

class FakeVisual(nn.Module):
 def __init__(self,grid=4): super().__init__(); self.grid=grid; self.anchor=nn.Parameter(torch.zeros(1))
 def forward(self,images):
  b=images.shape[0]; base=images.mean((2,3,4))[:,:,None,None]
  return type("F",(),{"features":base.expand(b,2,self.grid*self.grid,8)+torch.arange(self.grid*self.grid,device=images.device)[None,None,:,None]/100,
   "metadata":{"output_grid":self.grid}})()

class FakeText(nn.Module):
 def __init__(self): super().__init__(); self.anchor=nn.Parameter(torch.zeros(1))
 def forward(self,texts,role="query"):
  values=torch.tensor([[len(x)%7+i for i in range(8)] for x in texts],dtype=torch.float32,device=self.anchor.device)
  tokens=values[:,None,:].expand(-1,3,-1); mask=torch.ones(len(texts),3,dtype=torch.bool,device=values.device)
  return TextFeatures(torch.nn.functional.normalize(values,dim=-1),tokens,mask,mask,role,{})

def cfg(grid=4):
 return SinglePassConfig(visual_dim=8,text_dim=8,d_model=12,retrieval_dim=8,grid_size=grid,layers=2,heads=3,dropout=0)

def test_grid_contract_is_native_and_dynamic():
 model=BiTemporalPairTransformer(cfg(4)).eval(); out=model(torch.randn(2,2,16,8))
 assert out.contextual_patch_tokens.shape==(2,16,12)
 assert out.metadata["sequence_length"]==17
 assert out.pair_search_vector.shape==(2,8)
 assert torch.allclose(out.pair_search_vector.norm(dim=-1),torch.ones(2),atol=1e-5)

def test_moving_content_changes_cls():
 torch.manual_seed(1); model=BiTemporalPairTransformer(cfg(4)).eval(); x=torch.zeros(1,2,16,8)
 x[:,:,0]=1; first=model(x).pair_cls; y=torch.zeros_like(x); y[:,:,15]=1
 second=model(y).pair_cls; assert not torch.allclose(first,second)

def test_text_pair_vectors_and_backbones_frozen():
 model=QCPRSinglePassRetriever(FakeVisual(),FakeText(),cfg()).eval()
 out=model(torch.randn(2,2,3,8,8),["alpha","beta"])
 assert out.score_matrix.shape==(2,2)
 assert len(model.text_projection.adapter.layers)==2
 assert out.text.contextual_text_tokens.shape==(2,3,8)
 assert torch.allclose(out.text.text_search_vector.norm(dim=-1),torch.ones(2),atol=1e-5)
 assert not any(p.requires_grad for p in model.visual_encoder.parameters())
 assert not any(p.requires_grad for p in model.text_encoder.parameters())

def test_multi_positive_and_collision_exclusion():
 pos,exc=build_pair_masks(["a","a","b"],["a","b"],[set(),{"b"},set()])
 assert pos.tolist()==[[True,False],[True,False],[False,True]]
 assert exc[1,1]
 scores=torch.randn(3,2,requires_grad=True); loss,_=multi_positive_sigmoid_loss(scores,pos,exc)
 loss.backward(); assert torch.isfinite(loss) and scores.grad is not None

def test_localization_changes_with_query_and_pair():
 torch.manual_seed(4); loc=QueryConditionedLocalizer(text_dim=8,patch_dim=12,retrieval_dim=8,heads=3,layers=1,dropout=0,grid_size=4).eval()
 patches=torch.randn(2,16,12); q=torch.randn(2,8)
 out=loc(q,patches,output_size=(32,32))
 assert out.soft_map.shape==(2,4,4) and out.upsampled_soft_map.shape==(2,32,32)
 assert not torch.allclose(out.soft_map[0],out.soft_map[1])
 same=loc(q[:1].expand(2,-1),patches).soft_map
 assert not torch.allclose(same[0],same[1])

def test_mask_free_dataset_and_collator(tmp_path:Path):
 image=tmp_path/"x.png"; Image.new("RGB",(8,8),"white").save(image)
 mask=tmp_path/"must_not_open.png"
 row={"pair_id":"p","dataset_name":"d","split":"train","t1_path":str(image),"t2_path":str(image),
  "mask_path":str(mask),"captions":["one","two"],"source_metadata":{"changeflag":1}}
 manifest=tmp_path/"m.jsonl"; manifest.write_text(json.dumps(row)+"\n")
 dataset=MaskFreePairDataset(manifest,"train",8); batch=PairCaptionCollator(2)([dataset[0]])
 assert batch["images"].shape==(1,2,3,8,8)
 assert not ({"mask","masks","query_masks","segmentation_targets"}&batch.keys())

def test_retrieval_and_localization_optimizer_roles_are_separate():
 retrieval=QCPRSinglePassRetriever(FakeVisual(),FakeText(),cfg())
 loc=QueryConditionedLocalizer(text_dim=8,patch_dim=12,retrieval_dim=8,heads=3,layers=1,grid_size=4)
 rids={id(p) for p in retrieval.parameters() if p.requires_grad}; lids={id(p) for p in loc.parameters()}
 assert rids.isdisjoint(lids)

def test_in_job_oom_fallback_contract_present():
 source=(Path(__file__).parents[1]/"scripts/train_qcpr_single_pass.py").read_text()
 local=(Path(__file__).parents[1]/"scripts/train_qcpr_query_localization.py").read_text()
 assert "actual_batch,actual_accum=16,3" in source
 assert "actual=4" in local
