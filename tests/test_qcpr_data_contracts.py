from __future__ import annotations

from qcpr_data.contracts.schemas import FrameRecord, PhysicalItem, QueryRecord
from qcpr_data.contracts.validation import ValidationError, assert_mask_free, validate_query_record
from qcpr_data.identities.overlap import audit_split_leakage
from qcpr_data.queries.long_series import build_long_series_queries
from qcpr_data.queries.localized import build_localized_eval_queries
from qcpr_data.queries.semantic import build_semantic_eval_queries


def item(
    item_id: str = "x:1",
    *,
    split: str = "train",
    frame_sha: str = "a" * 64,
    second_frame_sha: str = "b" * 64,
) -> dict:
    return PhysicalItem(
        item_id=item_id,
        item_type="pair",
        source="x",
        source_revision="1",
        physical_group_id=item_id,
        scene_id=item_id,
        event_id=None,
        frames=(
            FrameRecord(f"{item_id}:0", "/tmp/a", frame_sha, "t1"),
            FrameRecord(f"{item_id}:1", "/tmp/b", second_frame_sha, "t2"),
        ),
        split=split,
        training_enabled=True,
        quality_status="READY",
    ).to_dict()


def test_nested_official_label_is_detected() -> None:
    try:
        assert_mask_free({"provenance": {"nested": {"official_label": "x"}}})
    except ValidationError as exc:
        assert "official_label" in str(exc)
    else:
        raise AssertionError("nested forbidden key was not detected")


def test_generated_caption_cannot_be_training_enabled() -> None:
    query = QueryRecord(
        query_id="q",
        text="generated",
        query_scope="exact",
        source_item_id="x",
        positive_item_ids=("x",),
        graded_relevance={"x": 3},
        temporal_direction="none",
        localized_relation=None,
        verification="generated_unverified",
        training_enabled=True,
        split="train",
    )
    assert validate_query_record(query) == ["unverified/generated/derived text cannot be training-enabled"]


def test_generated_verified_is_a_supported_but_nonpromoted_state() -> None:
    query = QueryRecord(
        query_id="q-verified-generated",
        text="candidate",
        query_scope="long_series",
        source_item_id="x",
        positive_item_ids=("x",),
        graded_relevance={"x": 2},
        temporal_direction="forward",
        localized_relation=None,
        verification="generated_verified",
        training_enabled=True,
        split="train",
    )
    assert validate_query_record(query) == ["unverified/generated/derived text cannot be training-enabled"]


def test_shared_image_across_splits_fails() -> None:
    result = audit_split_leakage(
        [
            item("x:train", frame_sha="a" * 64, second_frame_sha="b" * 64),
            item("x:test", split="test", frame_sha="a" * 64, second_frame_sha="c" * 64),
        ]
    )
    assert not result["passed"]
    assert result["shared_image_count"] == 1


def test_reversed_pair_across_splits_fails() -> None:
    forward = item("x:train", frame_sha="a" * 64, second_frame_sha="b" * 64)
    reverse = item("x:test", split="test", frame_sha="b" * 64, second_frame_sha="a" * 64)
    result = audit_split_leakage([forward, reverse])
    assert not result["passed"]
    assert result["reversed_pair_count"] == 1


def test_overlapping_sequence_windows_fail() -> None:
    first = item("sequence:train", frame_sha="a" * 64, second_frame_sha="b" * 64)
    second = item("sequence:test", split="test", frame_sha="b" * 64, second_frame_sha="c" * 64)
    first["item_type"] = "sequence"
    second["item_type"] = "sequence"
    from qcpr_data.splits.sequence_disjoint import validate_sequence_disjoint

    result = validate_sequence_disjoint([first, second])
    assert not result["passed"]
    assert result["shared_frame_count"] == 1


def test_event_like_semantic_group_is_not_promoted() -> None:
    rows = build_semantic_eval_queries(
        [{"canonical_pair_id": "x:1", "query_id": "q", "text": "change", "semantic_group_id": "event:1"}],
        {"x:1": item()},
    )
    assert rows == []


def test_semantic_queries_are_multi_positive_and_grouped_by_split() -> None:
    rows = build_semantic_eval_queries(
        [
            {"canonical_pair_id": "x:1", "query_id": "q1", "text": "new buildings appeared", "semantic_group_id": "appearance"},
            {"canonical_pair_id": "x:2", "query_id": "q2", "text": "buildings were added", "semantic_group_id": "appearance"},
        ],
        {"x:1": item("x:1"), "x:2": item("x:2")},
    )
    assert len(rows) == 2
    assert all(row["positive_item_ids"] == ["x:1", "x:2"] for row in rows)
    assert all(row["graded_relevance"] == {"x:1": 3, "x:2": 3} for row in rows)


def test_long_series_requires_explicit_temporal_annotation() -> None:
    sequence = item("sequence:1")
    sequence["item_type"] = "sequence"
    sequence["frames"].append(
        {"frame_id": "sequence:1:2", "path": "/tmp/c", "sha256": "c" * 64, "timestamp": "t3", "sensor": None, "gsd": None, "width": None, "height": None}
    )
    items = {"sequence:1": sequence}
    assert build_long_series_queries(
        [{"sequence_id": "sequence:1", "text": "change", "verification_status": "human"}], items
    ) == []
    rows = build_long_series_queries(
        [
            {
                "sequence_id": "sequence:1",
                "text": "urban expansion progressed",
                "verification_status": "human",
                "temporal_direction": "forward",
                "query_temporal_extent": {"start": "t1", "end": "t3"},
                "relevant_frame_range": {"start": 0, "end": 2},
            }
        ],
        items,
    )
    assert len(rows) == 1
    assert rows[0]["temporal_direction"] == "forward"


def test_item_aware_query_validation_rejects_singleton_semantic() -> None:
    query = QueryRecord(
        query_id="semantic-singleton",
        text="change",
        query_scope="semantic",
        source_item_id="x:1",
        positive_item_ids=("x:1",),
        graded_relevance={"x:1": 3},
        temporal_direction="none",
        localized_relation=None,
        verification="derived_eval",
        training_enabled=False,
        split="train",
    )
    assert validate_query_record(query, {"x:1"}, items={"x:1": item("x:1")}) == [
        "semantic queries must have at least two positive items"
    ]


def test_localized_builder_reuses_existing_evaluation_query_rows() -> None:
    rows, sidecars = build_localized_eval_queries(
        [
            {
                "query_id": "localized-q",
                "query_scope": "localized",
                "source_item_id": "x:1",
                "text": "change near the bottom center",
                "localized_relation": {"regions": ["lower", "center"], "evaluation_only": True},
                "provenance": {"source_caption_id": "caption-1", "source_dataset": "s2looking"},
                "verification": "derived_eval",
            }
        ],
        {"x:1": item("x:1")},
        {},
    )
    assert len(rows) == 1
    assert rows[0]["query_id"] == "localized-q"
    assert rows[0]["localized_relation"]["regions"] == ["lower", "center"]
    assert sidecars == []
