from __future__ import annotations

import torch
import pytest

from land_change_detection.models.retrieval_heads import (
    FalseNegativeSafeEmbeddingQueue,
    RetrievalProjectionHead,
    TextEmbeddingAdapter,
    build_caption_positive_mask,
    caption_detail_score,
    classify_caption_semantics,
    multi_positive_set_info_nce,
    multi_positive_symmetric_info_nce,
    semantic_direction_contradicts,
    semantic_teacher_relevance_matrix,
    semantic_text_to_pair_set_loss,
    normalize_caption_text,
    stable_caption_group_ids,
    structured_fna_loss,
    structured_fna_relevance_matrix,
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


@pytest.mark.parametrize(
    "caption",
    [
        "no change",
        "no changes",
        "no significant change",
        "no significant changes",
        "no noticeable change",
        "no visible change",
        "no substantial change",
        "no major change",
        "no meaningful change",
        "without change",
        "without any change",
        "remained unchanged",
        "remains unchanged",
        "stayed unchanged",
        "stays unchanged",
        "remained the same",
        "remains the same",
        "stayed the same",
        "no difference",
        "no differences",
        "no changes occurred",
        "no change occurred",
        "No significant change occurred.",
    ],
)
def test_no_change_phrases_are_negation_aware(caption):
    semantics = classify_caption_semantics(caption)
    assert semantics["no_change"] is True
    assert semantics["changed"] is False


@pytest.mark.parametrize(
    ("caption", "key"),
    [
        ("no new buildings appeared", "appeared"),
        ("no buildings were constructed", "constructed"),
        ("no structures were demolished", "demolished"),
        ("buildings did not appear", "appeared"),
        ("the area did not expand", "expanded"),
        ("vegetation has not decreased", "decreased"),
    ],
)
def test_negated_direction_terms_are_false_and_no_change(caption, key):
    semantics = classify_caption_semantics(caption)
    assert semantics[key] is False
    assert semantics["no_change"] is True
    assert semantics["changed"] is False


@pytest.mark.parametrize(
    ("caption", "key"),
    [
        ("two buildings appeared", "appeared"),
        ("several structures were demolished", "demolished"),
        ("vegetation decreased", "decreased"),
    ],
)
def test_non_negated_direction_terms_remain_changed(caption, key):
    semantics = classify_caption_semantics(caption)
    assert semantics[key] is True
    assert semantics["changed"] is True
    assert semantics["no_change"] is False


def test_cluster_regression_no_significant_change_occurred():
    semantics = classify_caption_semantics("No significant change occurred.")
    assert semantics["no_change"] is True
    assert semantics["changed"] is False


def test_structures_demolished_southern_area_regression():
    semantics = classify_caption_semantics("Several structures were demolished in the southern area.")
    assert semantics["demolished"] is True
    assert semantics["changed"] is True
    assert semantics["has_object"] is True
    assert "structures" in semantics["object_terms"]
    assert semantics["has_location"] is True
    assert "southern" in semantics["location_terms"]


@pytest.mark.parametrize(
    "object_phrase",
    [
        "structure",
        "structures",
        "facility",
        "facilities",
        "settlement",
        "settlements",
        "parking",
        "parking lot",
        "parking_lot",
        "bridge",
        "bridges",
    ],
)
def test_expanded_object_terms_are_detected(object_phrase):
    semantics = classify_caption_semantics(f"A new {object_phrase} appeared.")
    assert semantics["has_object"] is True


@pytest.mark.parametrize(
    "location",
    [
        "northern",
        "southern",
        "eastern",
        "western",
        "northeastern",
        "northwestern",
        "southeastern",
        "southwestern",
    ],
)
def test_directional_location_adjectives_are_detected(location):
    semantics = classify_caption_semantics(f"A building appeared in the {location} area.")
    assert semantics["has_location"] is True
    assert location in semantics["location_terms"]


@pytest.mark.parametrize("caption", ["no change", "no significant changes", "remained unchanged"])
def test_no_change_implies_not_changed_invariant(caption):
    semantics = classify_caption_semantics(caption)
    assert not (semantics["no_change"] and semantics["changed"])


def test_semantic_direction_contradicts_no_change_and_changed_symmetrically():
    no_change = classify_caption_semantics("No significant change occurred.")
    changed = classify_caption_semantics("Two buildings appeared.")
    assert semantic_direction_contradicts(no_change, changed) is True
    assert semantic_direction_contradicts(changed, no_change) is True


def test_semantic_paraphrases_receive_soft_relevance_and_exact_groups_are_hard():
    teacher = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0]]),
        dim=-1,
    )
    captions = ["A new building appeared", "New buildings were constructed", "The road disappeared"]
    mapping = torch.tensor([0, 1, 2])
    groups = stable_caption_group_ids(captions)
    relevance = semantic_teacher_relevance_matrix(teacher, captions, mapping, groups, pair_count=3, top_k=2)
    assert relevance[0, 0].item() == 1.0
    assert relevance[0, 1].item() > 0.9
    assert relevance[0, 2].item() == 0.0


def test_exact_normalized_caption_groups_remain_semantic_hard_positives():
    teacher = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    captions = ["No change has occurred.", "no change has occurred"]
    relevance = semantic_teacher_relevance_matrix(
        teacher,
        captions,
        torch.tensor([0, 1]),
        stable_caption_group_ids(captions),
        pair_count=2,
        top_k=1,
    )
    assert relevance.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_opposite_directions_receive_zero_semantic_relevance():
    teacher = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.01]]), dim=-1)
    captions = ["A building appeared", "A building was demolished"]
    relevance = semantic_teacher_relevance_matrix(
        teacher,
        captions,
        torch.tensor([0, 1]),
        stable_caption_group_ids(captions),
        pair_count=2,
        top_k=2,
    )
    assert relevance[0, 1].item() == 0.0
    assert relevance[1, 0].item() == 0.0


def test_semantic_soft_target_loss_is_detached_and_finite():
    pair = torch.nn.functional.normalize(torch.eye(3), dim=-1).requires_grad_(True)
    text = torch.nn.functional.normalize(torch.eye(3), dim=-1)
    teacher = text.detach().clone().requires_grad_(True)
    captions = ["A new building appeared", "New buildings were constructed", "No change"]
    loss = semantic_text_to_pair_set_loss(
        pair,
        text,
        teacher,
        captions,
        torch.tensor([0, 1, 2]),
        stable_caption_group_ids(captions),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert pair.grad is not None
    assert teacher.grad is None


def test_caption_detail_score_prefers_direction_object_count_location_without_length_domination():
    detailed = "Two new buildings appeared in the upper left corner"
    long_vague = "change " * 80
    assert caption_detail_score(detailed) > caption_detail_score(long_vague)


def test_structured_fna_attracts_compatible_constraints_and_rejects_direction_conflicts():
    captions = [
        "A building appeared in the upper left",
        "New buildings were constructed in the upper left",
        "A building was demolished in the upper left",
        "A road appeared on the right",
    ]
    mapping = torch.arange(4)
    relevance = structured_fna_relevance_matrix(
        captions,
        mapping,
        stable_caption_group_ids(captions),
        pair_count=4,
    )

    assert relevance[0, 0].item() == 1.0
    assert relevance[0, 1].item() == 1.0
    assert relevance[0, 2].item() == 0.0
    assert relevance[0, 3].item() == 0.5


def test_structured_fna_loss_is_finite_and_improves_when_compatible_scores_increase():
    relevance = torch.tensor([[1.0, 0.8, 0.0]])
    weak = torch.tensor([[0.0, -1.0, 1.0]], requires_grad=True)
    strong = torch.tensor([[2.0, 1.5, -2.0]], requires_grad=True)
    weak_loss = structured_fna_loss(weak, relevance)
    strong_loss = structured_fna_loss(strong, relevance)

    assert torch.isfinite(weak_loss)
    assert strong_loss < weak_loss
    strong_loss.backward()
    assert strong.grad is not None


def test_semantic_loss_reports_structured_fna_diagnostics_when_enabled():
    pair = torch.nn.functional.normalize(torch.eye(3), dim=-1).requires_grad_(True)
    text = torch.nn.functional.normalize(torch.eye(3), dim=-1)
    captions = ["A building appeared", "New buildings were constructed", "A building was demolished"]
    loss, diagnostics = semantic_text_to_pair_set_loss(
        pair,
        text,
        text.detach(),
        captions,
        torch.arange(3),
        stable_caption_group_ids(captions),
        structured_fna_weight=0.25,
        return_diagnostics=True,
    )
    assert torch.isfinite(loss)
    assert diagnostics["structured_fna_weight"] == 0.25
    assert diagnostics["structured_fna_positive_pairs"] > 3
