from evaluate_qcpr_data_only_comparison import (
    query_metrics,
    summarize_query_metrics,
)


def test_query_metrics_uses_all_graded_positive_items():
    result = query_metrics(
        [1, 0, 2],
        ["item-a", "item-b", "item-c"],
        {"item-a": 3, "item-c": 1},
    )
    assert result["mrr"] == 1 / 2
    assert result["recall_at_1"] == 0.0
    assert result["recall_at_5"] == 1.0
    assert result["positive_set_size"] == 2.0
    assert result["ndcg_at_10"] > 0.0


def test_per_source_summary_does_not_recurse():
    records = [
        {
            "mrr": 1.0,
            "recall_at_1": 1.0,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "average_precision": 1.0,
            "ndcg_at_10": 1.0,
            "positive_set_size": 1.0,
            "source": "source-a",
        },
        {
            "mrr": 0.5,
            "recall_at_1": 0.0,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "average_precision": 0.5,
            "ndcg_at_10": 0.6,
            "positive_set_size": 2.0,
            "source": "source-b",
        },
    ]
    result = summarize_query_metrics(records, 10)
    assert result["query_count"] == 2
    assert set(result["per_source"]) == {"source-a", "source-b"}
    assert result["metrics"]["mrr"] == 0.75
