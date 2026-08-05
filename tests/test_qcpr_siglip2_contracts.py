import pytest
import torch
from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.evaluation.retrieval import candidate_hit_at_k,multi_positive_recall_at_k,mrr_full,mrr_at_k
from qcpr_siglip2.models.evidence import EvidenceBottleneck
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel
from qcpr_siglip2.training.objective import multi_positive_listwise_loss
from run_qcpr_siglip2_real_smoke import build_relevance

def _features(q=4,p=3,t=2,n=256,l=7,d=768):
    torch.manual_seed(7); return (torch.randn(p,t,n,d),torch.randn(p,t,d),torch.randn(q,l,d),torch.randn(q,d),torch.ones(q,l,dtype=torch.bool))
def test_config_is_minimal_two_layer_contract():
    cfg=Siglip2TemporalConfig(); assert cfg.validate().temporal_layers==2; assert cfg.mlp_size==4*cfg.hidden_size
def test_model_outputs_causal_evidence_and_map():
    model=Siglip2TemporalRetrievalModel(None); out=model.forward_from_features(*_features(q=3,p=2)); assert out.score_matrix.shape==(3,2); assert out.evidence.evidence_map.shape==(3,2,2,16,16); assert out.evidence.evidence_vector.requires_grad; out.score_matrix.sum().backward(); assert model.evidence_bottleneck.raw_gate.grad is not None; assert model.temporal_adapter.blocks[0].attn_scale.grad is not None
def test_query_swap_changes_evidence_and_score():
    model=Siglip2TemporalRetrievalModel(None); out=model.forward_from_features(*_features(q=2,p=1)); assert not torch.allclose(out.evidence.evidence_map[0],out.evidence.evidence_map[1]); assert not torch.allclose(out.score_matrix[0],out.score_matrix[1])
def test_evidence_zeroing_changes_final_score():
    model=Siglip2TemporalRetrievalModel(None); frame_tokens,frame_embeddings,text_tokens,text_embeddings,text_mask=_features(q=2,p=2); normal=model.forward_from_features(frame_tokens,frame_embeddings,text_tokens,text_embeddings,text_mask); pair=torch.nn.functional.normalize(normal.pair_cls.unsqueeze(0),dim=-1); zero_score=torch.einsum("qd,qpd->qp",normal.text_embedding,pair)/model.retrieval_temperature; assert not torch.allclose(normal.score_matrix,zero_score)
def test_one_primary_scalar_loss_and_metrics():
    scores=torch.tensor([[4.,3.,1.],[2.,1.,0.]]); positives=torch.tensor([[True,True,False],[False,True,False]]); loss=multi_positive_listwise_loss(scores,positives); assert loss.ndim==0 and torch.isfinite(loss); assert candidate_hit_at_k(scores,positives,1)==.5; assert multi_positive_recall_at_k(scores,positives,1)==.25; assert mrr_full(scores,positives)==.75; assert mrr_at_k(scores,positives,1)==.5
def test_hit_and_recall_differ_for_multiple_positives():
    scores=torch.tensor([[3.,2.,1.]]); positives=torch.tensor([[True,True,False]]); assert candidate_hit_at_k(scores,positives,1)==1.; assert multi_positive_recall_at_k(scores,positives,1)==.5
def test_evidence_bottleneck_rejects_no_valid_text():
    with pytest.raises(ValueError,match="at least one"): EvidenceBottleneck(Siglip2TemporalConfig())(torch.randn(1,2,768),torch.zeros(1,2,dtype=torch.bool),torch.randn(1,512,768),frame_count=2,patch_count=256)

def test_smoke_relevance_does_not_infer_positives_from_text_collision():
    rows = [
        {"caption_id": "q0", "canonical_pair_id": "p0", "caption": "the same text", "positive_pair_ids": ["p0"], "ignored_pair_ids": []},
        {"caption_id": "q1", "canonical_pair_id": "p1", "caption": "the same text", "positive_pair_ids": ["p1"], "ignored_pair_ids": []},
    ]
    pairs = [{"canonical_pair_id": "p0"}, {"canonical_pair_id": "p1"}]
    positive, ignored, meta = build_relevance(rows, pairs, 1, torch.device("cpu"))
    assert positive.tolist() == [[True, False], [False, True]]
    assert not ignored.any()
    assert meta["multi_positive_queries"] == 0
    assert meta["text_collision_not_used_as_positive"] is True
