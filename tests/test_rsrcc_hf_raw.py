import json

from acquire_rsrcc_hf_raw import build_raw_entries


def test_build_raw_entries_uses_pinned_split_paths_and_deduplicates_references():
    metadata = {
        "train": [
            {"before": "images/000/a_before.png", "after": "images/000/a_after.png", "text": "t"},
            {"before": "images/000/a_before.png", "after": "images/000/a_after.png", "text": "t"},
        ],
        "val": [],
        "test": [],
    }
    entries, audit = build_raw_entries(metadata, revision="abc123")
    assert len(entries) == 2
    assert audit["metadata_row_count"] == 2
    assert all("/abc123/train/images/000/" in row["asset_url"] for row in entries)
    assert {row["reference_count"] for row in entries} == {2}
