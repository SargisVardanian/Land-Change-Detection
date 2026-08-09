from scripts.build_qcpr_forest_human_extension import forest_split_leakage, numeric_stats


def _item(item_id: str, split: str, group: str, a: str, b: str) -> dict:
    return {
        "item_id": item_id,
        "split": split,
        "physical_group_id": group,
        "frames": [{"sha256": a}, {"sha256": b}],
    }


def test_forest_split_leakage_passes_disjoint_items() -> None:
    result = forest_split_leakage(
        [
            _item("a", "train", "ga", "a1", "a2"),
            _item("b", "test", "gb", "b1", "b2"),
        ]
    )
    assert result["PASS"] is True
    assert all(result["checks"].values())


def test_forest_split_leakage_detects_reversed_pair() -> None:
    result = forest_split_leakage(
        [
            _item("a", "train", "ga", "same-a", "same-b"),
            _item("b", "test", "gb", "same-b", "same-a"),
        ]
    )
    assert result["PASS"] is False
    assert result["collision_counts"]["pair_frame_sha256"] == 1
    assert result["collision_counts"]["frame_sha256"] == 2


def test_numeric_stats_are_deterministic() -> None:
    assert numeric_stats([1, 2, 3, 4])["median"] == 2.5
    assert numeric_stats([1, 2, 3, 4])["p95"] == 4.0
