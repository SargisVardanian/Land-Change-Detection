from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image
from torch import nn
from land_change_detection.backbones.jina_v5_text import TextFeatures
from land_change_detection.models.qcpr_single_pass import (
 DeepResidualPairAdapter,QCPRSinglePassRetriever,QueryConditionedLocalizer,
 ResidualBottleneckTokenAdapter,SinglePassConfig,balanced_siglip_loss,build_pair_masks,
 multi_positive_sigmoid_loss)
from land_change_detection.models.qcpr_single_pass_factory import JointUniverSatSeriesEncoder
from land_change_detection.training.qcpr_single_pass_data import MaskFreePairDataset,PairCaptionCollator

class FakeVisual(nn.Module):
 def __init__(self,grid=4,dim=12):
  super().__init__(); self.grid=grid; self.dim=dim; self.anchor=nn.Parameter(torch.zeros(1)); self.last_t=None
 def forward(self,images,dates=None):
  self.last_t=images.shape[1]; batch=images.shape[0]
  base=images.mean((1,2,3,4))[:,None,None]
  tokens=base.expand(batch,self.grid*self.grid,self.dim)+torch.arange(self.grid*self.grid,device=images.device)[None,:,None]/100
  return SimpleNamespace(features=tokens,metadata={"grid_height":self.grid,"grid_width":self.grid,
    "temporal_series_length":images.shape[1]})

class FakeText(nn.Module):
 def __init__(self,dim=8): super().__init__(); self.anchor=nn.Parameter(torch.zeros(1)); self.dim=dim
 def forward(self,texts,role="query"):
  values=torch.tensor([[len(x)%7+i for i in range(self.dim)] for x in texts],dtype=torch.float32,device=self.anchor.device)
  tokens=values[:,None,:].expand(-1,3,-1); mask=torch.ones(len(texts),3,dtype=torch.bool,device=values.device)
  return TextFeatures(torch.nn.functional.normalize(values,dim=-1),tokens,mask,mask,role,{})

def config(grid=4):
 return SinglePassConfig(visual_dim=12,text_dim=8,retrieval_dim=8,grid_size=grid,
  token_adapter_layers=2,token_bottleneck_ratio=3,pair_layers=3,pair_heads=3,
  ffn_ratio=2,dropout=0,text_adapter_layers=2,text_heads=2)

def test_zero_initialized_token_adapters_are_exact_identity():
 layer=ResidualBottleneckTokenAdapter(12,3,0).eval(); x=torch.randn(2,16,12)
 assert torch.equal(layer(x),x)
 assert torch.count_nonzero(layer.up.weight)==0 and torch.count_nonzero(layer.up.bias)==0

def test_pair_adapter_baseline_parity_and_depth():
 torch.manual_seed(3); adapter=DeepResidualPairAdapter(config()).eval(); native=torch.randn(2,16,12)
 output=adapter(native); baseline=torch.nn.functional.normalize(adapter.baseline_vector(native),dim=-1)
 assert torch.allclose(output.adapted_dense_tokens,native,atol=0,rtol=0)
 assert torch.allclose(output.pair_search_vector,baseline,atol=1e-6)
 assert len(adapter.token_adapters)==2 and len(adapter.pair_blocks)==3
 assert torch.count_nonzero(adapter.delta_projection.weight)==0
 for block in adapter.pair_blocks:
  assert torch.allclose(block.attention_layer_scale,torch.full((12,),1e-3))
  assert torch.allclose(block.ffn_layer_scale,torch.full((12,),1e-3))
 assert not any(isinstance(module,nn.TransformerEncoder) for module in adapter.modules())

def test_retriever_supports_arbitrary_temporal_length_and_normalized_vectors():
 visual=FakeVisual(); model=QCPRSinglePassRetriever(visual,FakeText(),config()).eval()
 out=model(torch.randn(2,5,3,8,8),["alpha","beta"])
 assert visual.last_t==5 and out.pair.adapted_dense_tokens.shape==(2,16,12)
 assert out.score_matrix.shape==(2,2) and model.logit_bias.shape==()
 assert torch.allclose(out.pair.pair_search_vector.norm(dim=-1),torch.ones(2),atol=1e-5)
 assert torch.allclose(out.text.text_search_vector.norm(dim=-1),torch.ones(2),atol=1e-5)

def test_joint_universat_wrapper_preserves_arbitrary_t():
 class Enc(nn.Module):
  def __init__(self): super().__init__(); self.weight=nn.Parameter(torch.zeros(1)); self.last=None
  def encode(self,payload,**kwargs):
   series=payload["spot"]; self.last=series.shape[1]; return torch.ones(series.shape[0],16,12),{}
 encoder=Enc(); spec=SimpleNamespace(modality_name="spot",date_key="spot_dates",encode_kwargs=lambda grid:{"output_grid":grid})
 backend=SimpleNamespace(model=encoder,config=SimpleNamespace(adapter_spec=spec))
 wrapper=JointUniverSatSeriesEncoder(backend,4,12); output=wrapper(torch.randn(2,7,3,8,8))
 assert encoder.last==7 and output.features.shape==(2,16,12) and not output.features.requires_grad

def test_balanced_siglip_does_not_let_negative_count_dominate():
 scores=torch.tensor([[2.,-1.,-1.,-1.]],requires_grad=True)
 positive=torch.tensor([[1,0,0,0]],dtype=torch.bool); negative=~positive
 loss,stats=balanced_siglip_loss(scores,positive,negative)
 expected=.5*torch.nn.functional.softplus(torch.tensor(-2.))+.5*torch.nn.functional.softplus(torch.tensor(-1.))
 assert torch.allclose(loss,expected)
 loss.backward(); assert scores.grad is not None
 pos,exc=build_pair_masks(["a","a"],["a","b"],[set(),{"b"}])
 assert pos.tolist()==[[True,False],[True,False]] and exc[1,1]

def test_grounding_uses_contextual_text_and_only_weighted_dense_tokens():
 torch.manual_seed(4); grounder=QueryConditionedLocalizer(text_dim=8,patch_dim=12,retrieval_dim=8,
  heads=3,layers=2,ffn_ratio=2,dropout=0,grid_size=4).eval()
 text=torch.randn(2,8); tokens=torch.randn(2,3,8); mask=torch.ones(2,3,dtype=torch.bool); patches=torch.randn(2,16,12)
 out=grounder(text,tokens,mask,mask,patches,output_size=(32,32))
 assert out.native_soft_map.shape==(2,4,4) and out.upsampled_soft_map.shape==(2,32,32)
 projected=grounder.dense_value_projection(patches)
 expected=torch.nn.functional.normalize(torch.einsum("bn,bnd->bd",out.relevance_probabilities,projected),dim=-1)
 assert torch.allclose(out.grounded_visual_embedding,expected,atol=1e-6)
 changed_tokens=tokens.clone(); changed_tokens[:,0]+=3
 changed=grounder(text,changed_tokens,mask,mask,patches).native_soft_map
 assert not torch.allclose(out.native_soft_map,changed)

def test_map_changes_with_query_and_pair():
 torch.manual_seed(5); grounder=QueryConditionedLocalizer(text_dim=8,patch_dim=12,retrieval_dim=8,
  heads=3,layers=1,ffn_ratio=2,dropout=0,grid_size=4).eval()
 patches=torch.randn(2,16,12); mask=torch.ones(2,3,dtype=torch.bool)
 first=grounder(torch.randn(2,8),torch.randn(2,3,8),mask,mask,patches).native_soft_map
 same_query=torch.randn(1,8).expand(2,-1); same_tokens=torch.randn(1,3,8).expand(2,-1,-1)
 second=grounder(same_query,same_tokens,mask,mask,patches).native_soft_map
 assert not torch.allclose(first[0],first[1]) and not torch.allclose(second[0],second[1])

def test_mask_free_dataset_and_collator(tmp_path:Path):
 image=tmp_path/"x.png"; Image.new("RGB",(8,8),"white").save(image); missing_mask=tmp_path/"must_not_open.png"
 row={"pair_id":"p","dataset_name":"d","split":"train","t1_path":str(image),"t2_path":str(image),
  "mask_path":str(missing_mask),"captions":["one","two"],"source_metadata":{"changeflag":1}}
 manifest=tmp_path/"m.jsonl"; manifest.write_text(json.dumps(row)+"\n")
 batch=PairCaptionCollator(2)([MaskFreePairDataset(manifest,"train",8)[0]])
 assert batch["images"].shape==(1,2,3,8,8)
 assert not ({"mask","masks","query_masks","segmentation_targets"}&batch.keys())

def test_in_job_oom_fallback_and_no_joint_training():
 root=Path(__file__).parents[1]; retrieval=(root/"scripts/train_qcpr_single_pass.py").read_text()
 grounding=(root/"scripts/train_qcpr_query_localization.py").read_text()
 assert "actual_batch,actual_accum=16,4" in retrieval
 assert "actual_batch,actual_accum=4,4" in grounding
 assert "for parameter in model.parameters(): parameter.requires_grad_(False)" in grounding
 assert "optimizer.load_state_dict(payload" not in grounding
