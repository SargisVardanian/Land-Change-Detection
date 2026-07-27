from land_change_detection.benchmark.qcpr_metrics import aggregate_rankings, changeretcap_compat, ndcg_at


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
