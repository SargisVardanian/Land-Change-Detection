from __future__ import annotations

import torch

from land_change_detection.models.retrieval_heads import (
    FalseNegativeSafeEmbeddingQueue,
    RetrievalProjectionHead,
    TextEmbeddingAdapter,
    build_caption_positive_mask,
    classify_caption_semantics,
    multi_positive_set_info_nce,
    multi_positive_symmetric_info_nce,
    normalize_caption_text,
    stable_caption_group_ids,
)


def test_stable_caption_groups_ignore_case_and_punctuation():
    groups = stable_caption_group_ids(
        ["No change has occurred.", "no change has occurred", "A building appeared"]
    )
    assert groups[0].item() == groups[1].item()
    assert groups[0].item() != groups[2].item()


def test_caption_normalization_handles_whitespace_unicode_and_is_deterministic():
    texts = ["No change has occurred.", "  NO   CHANGE has occurred!!! ", "Ｎｏ change has occurred"]
    normalized = [normalize_caption_text(text) for text in texts]
    assert normalized == ["no change has occurred"] * 3
    first = stable_caption_group_ids(texts)
    second = stable_caption_group_ids(texts)
    assert torch.equal(first, second)


def test_positive_mask_expands_normalized_duplicate_groups():
    mapping = torch.tensor([0, 1, 1])
    groups = stable_caption_group_ids(["same.", "Same", "different"])
    mask = build_caption_positive_mask(mapping, pair_count=2, caption_group_ids=groups)
    assert mask.tolist() == [
        [True, True, False],
        [True, True, True],
    ]


def test_set_mass_loss_removes_positive_count_floor():
    pair = torch.eye(2)
    text = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    mapping = torch.tensor([0, 0, 1, 1])
    groups = stable_caption_group_ids(["p0a", "p0b", "p1a", "p1b"])
    old = multi_positive_symmetric_info_nce(
        pair,
        text,
        mapping,
        groups,
        temperature=0.01,
    )
    new = multi_positive_set_info_nce(
        pair,
        text,
        mapping,
        groups,
        temperature=0.01,
        text_to_pair_weight=0.5,
        pair_to_text_weight=0.5,
    )
    assert new.item() < 1e-4
    assert old.item() > 0.3


def test_empty_positive_mask_is_rejected():
    with torch.no_grad():
        pair = torch.eye(2)
        text = torch.eye(2)
        mapping = torch.tensor([0, 3])
        groups = stable_caption_group_ids(["a", "b"])
        try:
            multi_positive_set_info_nce(pair, text, mapping, groups)
        except ValueError as exc:
            assert "out-of-range" in str(exc)
        else:
            raise AssertionError("Expected out-of-range positive mapping rejection")


def test_trainable_temperature_is_bounded_and_differentiable():
    head = RetrievalProjectionHead(
        4,
        trainable_temperature=True,
        initial_temperature=0.07,
        max_logit_scale=20.0,
    )
    scale = head.similarity_scale()
    assert scale.requires_grad
    assert 14.0 < scale.item() < 15.0
    head.logit_scale.data.fill_(100.0)
    assert head.similarity_scale().item() <= 20.0001
    loss = head.similarity_scale()
    loss.backward()
    assert head.logit_scale.grad is not None
    assert torch.isfinite(head.logit_scale.grad)


def test_fixed_temperature_keeps_legacy_projection_state_dict_shape():
    head = RetrievalProjectionHead(4, trainable_temperature=False)
    assert "logit_scale" not in head.state_dict()


def test_text_adapter_near_identity_shape_and_gradients():
    adapter = TextEmbeddingAdapter(4)
    x = torch.randn(3, 4)
    y = adapter(x)
    assert y.shape == x.shape
    assert torch.allclose(y, torch.nn.functional.normalize(x, dim=-1), atol=1e-6)
    loss = (y[:, 0] ** 2).sum()
    loss.backward()
    grads = [parameter.grad for parameter in adapter.parameters() if parameter.grad is not None]
    assert grads and any(torch.isfinite(grad).all() and bool((grad.abs() > 0).any()) for grad in grads)


def test_false_negative_safe_queue_excludes_same_caption_group():
    queue = FalseNegativeSafeEmbeddingQueue(max_size=4, dim=2)
    groups = stable_caption_group_ids(["same", "other"])
    queue.enqueue(torch.eye(2), groups)
    mask = queue.safe_negative_mask(stable_caption_group_ids(["SAME!", "new"]))
    assert mask.tolist() == [[False, True], [True, True]]


def test_caption_semantic_classifier_is_centralized():
    assert classify_caption_semantics("A new building appeared")["appeared"]
    assert classify_caption_semantics("The building disappeared")["disappeared"]
    assert classify_caption_semantics("No change has occurred.")["no_change"]
