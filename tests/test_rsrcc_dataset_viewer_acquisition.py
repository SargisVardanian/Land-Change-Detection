import json

from acquire_rsrcc_via_dataset_viewer import resolve_viewer_assets


def test_resolve_viewer_assets_maps_val_to_validation_and_checks_text():
    metadata = {
        "train": [{"before": "images/000/a_before.png", "after": "images/000/a_after.png", "text": "Train text"}],
        "val": [{"before": "images/001/b_before.png", "after": "images/001/b_after.png", "text": "Val text"}],
        "test": [{"before": "images/002/c_before.png", "after": "images/002/c_after.png", "text": "Test text"}],
    }
    payloads = {
        "train": {"rows": [{"row_idx": 0, "row": {"before": {"src": "u-ta", "width": 512, "height": 512}, "after": {"src": "u-tb"}, "text": "Train text"}}]},
        "validation": {"rows": [{"row_idx": 0, "row": {"before": {"src": "u-va"}, "after": {"src": "u-vb"}, "text": "Val text"}}]},
        "test": {"rows": [{"row_idx": 0, "row": {"before": {"src": "u-sa"}, "after": {"src": "u-sb"}, "text": "Test text"}}]},
    }
    def fetch(split, offset, length):
        assert offset == 0
        assert length == 100
        return payloads[split]
    entries, audit = resolve_viewer_assets(metadata, page_fetch=fetch)
    assert audit["alignment_ok"] is True
    assert {(x["local_split"], x["viewer_split"]) for x in entries} == {
        ("train", "train"), ("val", "validation"), ("test", "test")
    }
    assert {x["asset_url"] for x in entries} == {"u-ta", "u-tb", "u-va", "u-vb", "u-sa", "u-sb"}


def test_resolve_viewer_assets_rejects_text_alignment_failure():
    metadata = {
        "train": [{"before": "a", "after": "b", "text": "metadata text"}],
        "val": [],
        "test": [],
    }
    payloads = {"train": {"rows": [{"row_idx": 0, "row": {"before": {"src": "u-a"}, "after": {"src": "u-b"}, "text": "different"}}]}}
    entries, audit = resolve_viewer_assets(
        metadata,
        page_fetch=lambda split, offset, length: payloads.get(split, {"rows": []}),
    )
    assert len(entries) == 2
    assert audit["alignment_ok"] is False
    assert audit["text_mismatches"] == [{"local_split": "train", "viewer_split": "train", "row_idx": 0}]


def test_resolve_viewer_assets_uses_total_for_ordered_parallel_pages():
    metadata = {
        "train": [
            {"before": f"images/{i}/before.png", "after": f"images/{i}/after.png", "text": f"Text {i}"}
            for i in range(4)
        ],
        "val": [],
        "test": [],
    }
    calls = []

    def fetch(split, offset, length):
        calls.append((split, offset, length))
        if split != "train":
            return {"rows": []}
        rows = []
        for index in range(offset, min(offset + 2, 4)):
            rows.append({
                "row_idx": index,
                "row": {
                    "before": {"src": f"before-{index}"},
                    "after": {"src": f"after-{index}"},
                    "text": f"Text {index}",
                },
            })
        return {"num_rows_total": 4, "rows": rows}

    entries, audit = resolve_viewer_assets(
        metadata,
        page_fetch=fetch,
        page_length=2,
        page_workers=2,
    )
    assert audit["alignment_ok"] is True
    assert audit["pages"]["train"] == {"page_count": 2, "metadata_rows": 4, "viewer_rows": 4}
    assert {row["viewer_row_idx"] for row in entries} == {0, 1, 2, 3}
    assert {offset for split, offset, length in calls if split == "train"} == {0, 2}
