import torch
from land_change_detection.models.shared_adapters import SharedTextAttentionAdapter

def test_shared_adapter_exact_zero_gate_and_shapes():
    m=SharedTextAttentionAdapter()
    assert all(torch.equal(p,torch.zeros_like(p)) for n,p in m.named_parameters() if n.endswith(("alpha_attn","alpha_ffn")))
    x=torch.randn(2,5,512); base=torch.nn.functional.normalize(torch.randn(2,512),dim=-1)
    attn=torch.ones(2,5,dtype=torch.bool); content=torch.zeros(2,5,dtype=torch.bool)
    g,t=m(base,x,attn,content)
    assert g.shape==(2,512) and t.shape==(2,5,512)
    assert torch.isfinite(g).all() and torch.isfinite(t).all()
    assert torch.allclose(t,x)
    assert torch.allclose(g,base,atol=1e-6,rtol=1e-6)

def test_shared_adapter_uses_content_tokens_when_available():
    m=SharedTextAttentionAdapter()
    x=torch.randn(1,4,512); base=torch.nn.functional.normalize(torch.randn(1,512),dim=-1)
    attn=torch.ones(1,4,dtype=torch.bool); content=torch.tensor([[True,False,False,False]])
    g,t=m(base,x,attn,content)
    assert g.shape==(1,512)
