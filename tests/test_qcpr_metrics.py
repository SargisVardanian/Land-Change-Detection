from land_change_detection.benchmark.qcpr_metrics import (
    aggregate_rankings,
    auprc,
    changeretcap_compat,
    effective_patch_count,
    energy_inside_mask,
    ndcg_at,
    normalized_entropy,
    pointing_game,
    soft_iou,
)


def test_exact_metrics_use_physical_relevance_sets():
    rows = [(["wrong", "pair-a", "pair-b"], {"pair-a"}), (["pair-b"], {"pair-b"})]
    result = aggregate_rankings(rows)
    assert result["queries"] == 2
    assert result["recall@1"] == 0.5
    assert result["recall@5"] == 1.0
    assert 0.0 < result["mrr"] < 1.0


def test_ndcg_supports_multi_positive_relevance():
    assert ndcg_at(["a", "b"], {"a", "b"}, 2) == 1.0


def test_changeretcap_is_explicit_top_k_view():
    result = changeretcap_compat(["x", "a", "b"], {"a", "b"}, k=2)
    assert result["recall"] == 1.0
    assert result["precision"] == 0.5


def test_soft_localization_metrics_are_deterministic():
    scores = [0.9, 0.1, 0.0, 0.0]
    mask = [True, False, False, False]
    assert energy_inside_mask(scores, mask) == 0.9
    assert soft_iou(scores, mask) == 1.0
    assert auprc(scores, mask) == 1.0
    assert pointing_game(scores, mask) == 1.0
    assert effective_patch_count(scores) > 1.0
    assert 0.0 <= normalized_entropy(scores) <= 1.0
