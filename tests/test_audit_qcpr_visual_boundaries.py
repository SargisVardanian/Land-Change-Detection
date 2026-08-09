from __future__ import annotations

from scripts.audit_qcpr_visual_boundaries import BKTree, _cross_count, _different_item_count


def test_boundary_counts_and_bktree() -> None:
    rows = [
        {"item_id": "a", "source": "x", "split": "train"},
        {"item_id": "b", "source": "x", "split": "test"},
        {"item_id": "c", "source": "y", "split": "test"},
    ]
    assert _different_item_count(rows, rows) == 3
    assert _cross_count(rows, rows, "source") == 2
    assert _cross_count(rows, rows, "split") == 2
    tree = BKTree()
    tree.add(0)
    tree.add(3)
    assert sorted(tree.query(1, 1)) == [(0, 1), (3, 1)]
