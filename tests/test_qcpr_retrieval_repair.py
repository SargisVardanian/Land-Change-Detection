from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

from land_change_detection.models.qcpr_single_pass import balanced_siglip_loss
from land_change_detection.training.qcpr_retrieval_repair import (
    DeterministicRotatingCaptionCollator,
    LogicalBatchContract,
    aggregate_hard_pairs,
    exact_grad_cache_backward,
    final_attention_block_prefixes,
    filip_token_patch_scores,
    hard_aware_epoch_order,
    inject_lora_attention_projections,
    mine_hard_negative_rows,
)


class TinyDualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = nn.Linear(5, 4)
        self.pair = nn.Linear(6, 4)
        self.logit_scale = nn.Parameter(torch.tensor(1.3))
        self.logit_bias = nn.Parameter(torch.tensor(-0.2))

    def encode(self, batch):
        text = F.normalize(self.text(batch["text"]), dim=-1)
        pair = F.normalize(self.pair(batch["pair"]), dim=-1)
        return text, pair

    def loss(self, text, pair):
        logits = self.logit_scale.exp() * text @ pair.T + self.logit_bias
        positive = torch.tensor([
            [query // 2 == candidate for candidate in range(pair.shape[0])]
            for query in range(text.shape[0])
        ], dtype=torch.bool)
        return balanced_siglip_loss(logits, positive, ~positive)


def test_logical_batch_contract_is_one_256_by_128_matrix():
    contract = LogicalBatchContract(16, 128, 2)
    assert contract.micro_batches_per_logical_batch == 8
    assert contract.logical_query_count == 256
    assert contract.score_matrix_shape == (256, 128)
    assert contract.score_matrix_shape != (32, 16)


def test_grad_cache_matches_direct_full_batch_gradients():
    torch.manual_seed(11)
    direct = TinyDualEncoder()
    cached = copy.deepcopy(direct)
    text = torch.randn(16, 5)
    pair = torch.randn(8, 6)
    full = {"text": text, "pair": pair}
    direct_text, direct_pair = direct.encode(full)
    direct_loss, _ = direct.loss(direct_text, direct_pair)
    direct_loss.backward()
    micro = [
        {"text": text[:8], "pair": pair[:4]},
        {"text": text[8:], "pair": pair[4:]},
    ]
    cached_loss, _, shape = exact_grad_cache_backward(
        micro_batches=micro,
        encode=cached.encode,
        loss_function=cached.loss,
        device=torch.device("cpu"),
    )
    assert shape == (16, 8)
    assert torch.allclose(cached_loss, direct_loss, atol=1e-7)
    for first, second in zip(direct.parameters(), cached.parameters(), strict=True):
        assert first.grad is not None and second.grad is not None
        assert torch.allclose(first.grad, second.grad, atol=2e-6, rtol=2e-5)


def test_hard_aware_order_is_deterministic_and_equal_weight_before_padding():
    pair_ids = [f"p{i}" for i in range(259)]
    hard = {"p0": ["p17", "p18"], "p17": ["p0"]}
    first, stats = hard_aware_epoch_order(pair_ids, logical_batch=128, seed=7, hard_by_pair=hard)
    second, second_stats = hard_aware_epoch_order(pair_ids, logical_batch=128, seed=7, hard_by_pair=hard)
    assert first == second and stats == second_stats
    assert len(first) % 128 == 0
    assert set(first[: stats["unique_physical_items"]]) == set(range(259))
    assert len(first[: stats["unique_physical_items"]]) == len(set(first[:259]))
    assert stats["logical_padding_items"] == 125


def test_hard_negative_mining_excludes_positive_and_ambiguous_pairs():
    pairs = F.normalize(torch.tensor([[1., 0.], [.99, .01], [.8, .2], [0., 1.]]), dim=-1)
    queries = pairs[[0, 3]]
    kwargs = dict(
        query_vectors=queries,
        pair_vectors=pairs,
        query_pair_ids=["p0", "p3"],
        gallery_pair_ids=["p0", "ambiguous", "p2", "p3"],
        query_captions=["same", "other"],
        excluded_pair_ids=[{"ambiguous"}, set()],
        top_k=2,
    )
    rows = mine_hard_negative_rows(**kwargs, chunk_size=1)
    assert "p0" not in rows[0]["hard_pair_ids"]
    assert "ambiguous" not in rows[0]["hard_pair_ids"]
    assert "p3" not in rows[1]["hard_pair_ids"]
    assert aggregate_hard_pairs(rows)["p0"] == rows[0]["hard_pair_ids"]
    assert rows == mine_hard_negative_rows(**kwargs, chunk_size=2)


def test_filip_local_score_is_finite_and_uses_only_content_tokens():
    torch.manual_seed(2)
    text = torch.randn(2, 3, 4)
    patches = torch.randn(2, 6, 4)
    mask = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.bool)
    score = filip_token_patch_scores(text, patches, mask, top_k=4)
    changed = text.clone()
    changed[:, 2] += 10_000
    ignored_score = filip_token_patch_scores(changed, patches, mask, top_k=4)
    assert score.shape == (2, 2)
    assert torch.isfinite(score).all()
    assert torch.allclose(score, ignored_score)


def test_two_captions_per_item_have_equal_physical_weight():
    pair_ids = ["p0", "p1", "p2"]
    query_ids = ["p0", "p0", "p1", "p1", "p2", "p2"]
    positive = torch.tensor([[q == p for p in pair_ids] for q in query_ids])
    assert positive.sum(1).tolist() == [1] * 6
    assert positive.sum(0).tolist() == [2, 2, 2]


def test_caption_rotation_covers_all_captions_deterministically():
    item = SimpleNamespace(
        pair_id="p0", images=torch.zeros(2, 3, 4, 4),
        captions=[f"caption-{index}" for index in range(5)],
        normalized_captions=[f"caption-{index}" for index in range(5)],
        change_status="changed", metadata={},
    )
    selected = []
    for epoch in range(3):
        batch = DeterministicRotatingCaptionCollator(2, seed=9, epoch=epoch)([item])
        selected.extend(batch["captions"])
        assert len(batch["captions"]) == 2
    assert set(selected) == set(item.captions)
    repeated = DeterministicRotatingCaptionCollator(2, seed=9, epoch=1)([item])
    assert repeated["captions"] == selected[2:4]


def test_lora_affects_only_explicit_final_attention_blocks():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4)
            self.k_proj = nn.Linear(4, 4)
            self.v_proj = nn.Linear(4, 4)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(4)])

    backbone = Backbone()
    prefixes = final_attention_block_prefixes(backbone, "layers", count=2)
    replaced = inject_lora_attention_projections(
        backbone,
        approved_block_prefixes=prefixes,
        projection_names=("q_proj", "k_proj", "v_proj"),
        rank=2,
    )
    assert prefixes == ["layers.2", "layers.3"]
    assert len(replaced) == 6
    trainable = [name for name, parameter in backbone.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith(("layers.2.", "layers.3.")) for name in trainable)
    assert all("lora_" in name for name in trainable)
    assert not any(name.startswith(("layers.0.", "layers.1.")) for name in trainable)


def test_r1_checkpoint_and_batch_metadata_contracts_are_explicit():
    root = Path(__file__).parents[1]
    training = (root / "scripts/train_qcpr_retrieval_repair_r1.py").read_text()
    launcher = (root / "cluster/ysu/submit_qcpr_retrieval_repair_r1.sh").read_text()
    assert 'baseline_payload.get("epoch") != 19' in training
    assert 'baseline_payload["optimizer"]' not in training
    assert '"fresh_optimizer": True' in training
    assert '"old_optimizer_loaded": False' in training
    assert '"logical_score_matrix": list(contract.score_matrix_shape)' in training
    assert '"gradient_accumulation": 1' in training
    assert '"gradcache_recomputation": True' in training
    assert '--logical-physical-batch 128' in launcher
    assert '--physical-micro-batch 16' in launcher
    assert 'automatic_promotion_to_r2' in launcher
