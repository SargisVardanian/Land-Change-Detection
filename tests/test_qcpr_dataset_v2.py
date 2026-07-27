from pathlib import Path

from PIL import Image

from land_change_detection.data.qcpr_dataset_v2 import (
    build_relevance,
    captions_for_pair,
    canonical_pair_id_for,
    exclude_cross_split_image_conflicts,
    leakage_audit,
    legacy_pair_to_v2,
    mask_free,
    normalize_text,
    validate_relevance,
)


def row(pair="p", split="train", caption="A building appeared.", change=1, dataset="LEVIR-MCI"):
    return {
        "pair_id": pair,
        "split": split,
        "dataset_name": dataset,
        "t1_path": "a.png",
        "t2_path": "b.png",
        "captions": [caption],
        "source_metadata": {"changeflag": change, "scene_id": pair},
    }


def test_canonical_ids_are_namespaced_by_source():
    assert canonical_pair_id_for(row(pair="1", dataset="LEVIR-MCI")) == "levir_mci:1"
    assert canonical_pair_id_for(row(pair="1", dataset="SECOND-CC")) == "second_cc:1"


def test_generic_no_change_never_has_exact_positive():
    cap = captions_for_pair(row(caption="There is no difference.", change=0))[0]
    assert cap["query_scope"] == "generic_no_change"
    relevance = build_relevance([cap])[0]
    assert relevance["positive_pair_ids"] == []
    assert relevance["ignored_pair_ids"] == []
    assert relevance["negative_policy"] == "semantic_group_only"


def test_vague_changed_caption_is_not_forced_to_exact_pair():
    cap = captions_for_pair(row(caption="A building appeared.", change=1))[0]
    assert cap["query_scope"] == "semantic_group"
    assert cap["identifiability_score"] < 0.6


def test_distinctive_caption_can_be_exact_pair():
    cap = captions_for_pair(
        row(caption="Two buildings appeared beside the diagonal road in the upper left corner.")
    )[0]
    assert cap["query_scope"] == "exact_pair"


def test_exact_collisions_are_ignored_not_negative():
    first = captions_for_pair(row("a", caption="Two buildings appeared beside the road in the upper left corner."))[0]
    second = captions_for_pair(row("b", caption=first["text"]))[0]
    relevance = build_relevance([first, second])
    assert relevance[0]["ignored_pair_ids"] == ["levir_mci:b"]


def test_semantic_group_collisions_are_multi_positive():
    first = captions_for_pair(row("a", caption="new road"))[0]
    second = captions_for_pair(row("b", caption="new road"))[0]
    relevance = build_relevance([first, second])
    assert set(relevance[0]["positive_pair_ids"]) == {"levir_mci:a", "levir_mci:b"}


def test_sequence_compatibility_views_are_preserved():
    pair = legacy_pair_to_v2(row())
    assert len(pair["frames"]) == 2 and pair["t1_path"] == "a.png"
    assert pair["ordered_pair_hash"]
    assert pair["order_invariant_pair_hash"]


def test_cross_split_source_scene_fails_audit():
    first = legacy_pair_to_v2(row("a", "train"))
    second = legacy_pair_to_v2(row("b", "development"))
    second["source_scene_group_id"] = first["source_scene_group_id"]
    assert not leakage_audit([first, second])["passed"]


def test_mask_keys_cannot_enter_nested_mask_free_rows():
    try:
        mask_free({"source_metadata": {"query_masks": {"q": "x.png"}}})
    except ValueError:
        pass
    else:
        raise AssertionError("nested dense-label path accepted")


def test_caption_normalization_is_stable():
    assert normalize_text("New, ROAD!") == "new road"


def test_cross_split_shared_image_excludes_training_not_development(tmp_path):
    image = tmp_path / "shared.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image)
    train = row("train", "train")
    development = row("development", "val")
    train["t1_path"] = development["t2_path"] = str(image)
    kept, report = exclude_cross_split_image_conflicts([train, development])
    assert [item["pair_id"] for item in kept] == ["development"]
    assert report["excluded_pair_ids_by_source_role"] == {"train": ["levir_mci:train"]}


def test_reversed_pair_is_detected_across_splits(tmp_path):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(left)
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(right)
    train = row("train", "train")
    dev = row("dev", "development")
    train["t1_path"], train["t2_path"] = str(left), str(right)
    dev["t1_path"], dev["t2_path"] = str(right), str(left)
    kept, report = exclude_cross_split_image_conflicts([train, dev])
    assert [item["pair_id"] for item in kept] == ["dev"]
    assert report["conflict_components_before"] == 1


def test_relevance_audit_checks_true_positive():
    source = row(caption="Two buildings appeared beside the road in the upper left corner.")
    pair = legacy_pair_to_v2(source)
    caption = captions_for_pair(source)[0]
    relevance = build_relevance([caption])
    assert validate_relevance([pair], [caption], relevance)["passed"]
