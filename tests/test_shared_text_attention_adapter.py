import torch
from land_change_detection.models.shared_adapters import SharedTextAttentionAdapter
from land_change_detection.models.qcpr_v3 import QCPRV3Config, QCPRV3GenericGrounding

def _input():
    return (torch.nn.functional.normalize(torch.randn(2,512),dim=-1),
            torch.randn(2,5,512),
            torch.ones(2,5,dtype=torch.bool),
            torch.tensor([[True,False,True,False,False],[True,True,False,False,False]]))

def test_shared_adapter_exact_identity_and_empty_masks():
    model=SharedTextAttentionAdapter(); base,tokens,attention,content=_input()
    global_out,adapted,safe=model(base,tokens,attention,content)
    assert torch.allclose(global_out,base,atol=1e-6,rtol=1e-6)
    assert torch.allclose(adapted,tokens)
    assert safe.shape==attention.shape
    empty=torch.zeros(1,4,dtype=torch.bool)
    out,local,safe=model(base[:1],tokens[:1,:4],empty,empty)
    assert safe.tolist()==[[True,False,False,False]]
    assert torch.isfinite(out).all() and torch.isfinite(local).all()

def test_global_pooled_branch_wakes_after_first_step():
    model=SharedTextAttentionAdapter(); optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
    base,tokens,attention,content=_input()
    out,_,_=model(base,tokens,attention,content); loss=out[:,0].sum(); loss.backward()
    assert model.beta_text.grad is not None and model.beta_text.grad.abs().item()>0
    optimizer.step(); optimizer.zero_grad()
    out,_,_=model(base,tokens,attention,content); out[:,0].sum().backward()
    assert model.adapted_projection.weight.grad is not None
    assert model.attention_query.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.endswith(("alpha_attn","alpha_ffn")))

def test_b_token_patch_score_responds_to_adapted_tokens():
    torch.manual_seed(3)
    grounder=QCPRV3GenericGrounding(QCPRV3Config(input_dim=512,visual_source_dim=512,hidden_dim=256,output_size=(32,32)))
    pair=torch.nn.functional.normalize(torch.randn(1,512),dim=-1)
    times=torch.randn(1,2,1024,512)
    tokens=torch.randn(1,4,512); mask=torch.ones(1,4,dtype=torch.bool)
    first=grounder.score_query_pair_chunks(pair,tokens,mask,pair,times,decode_mask=False).token_patch_score
    second=grounder.score_query_pair_chunks(pair,tokens+0.5,mask,pair,times,decode_mask=False).token_patch_score
    assert not torch.allclose(first,second)
