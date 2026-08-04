from __future__ import annotations

from qcpr_data.contracts.schemas import FrameRecord, PhysicalItem, QueryRecord
from qcpr_data.contracts.validation import ValidationError, assert_mask_free, validate_query_record
from qcpr_data.identities.overlap import audit_split_leakage
from qcpr_data.queries.semantic import build_semantic_eval_queries


def item(item_id: str = "x:1", *, split: str = "train", frame_sha: str = "a" * 64) -> dict:
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
            FrameRecord(f"{item_id}:1", "/tmp/b", "b" * 64, "t2"),
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


def test_shared_image_across_splits_fails() -> None:
    result = audit_split_leakage([item("x:train", frame_sha="a" * 64), item("x:test", split="test", frame_sha="a" * 64)])
    assert not result["passed"]
    assert result["shared_image_count"] == 1


def test_event_like_semantic_group_is_not_promoted() -> None:
    rows = build_semantic_eval_queries(
        [{"canonical_pair_id": "x:1", "query_id": "q", "text": "change", "semantic_group_id": "event:1"}],
        {"x:1": item()},
    )
    assert rows == []
