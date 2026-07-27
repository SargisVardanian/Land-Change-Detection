from pathlib import Path
from land_change_detection.data.qcpr_dataset_v2 import build_relevance, captions_for_pair, exclude_cross_split_image_conflicts, leakage_audit, legacy_pair_to_v2, mask_free, normalize_text

def row(pair="p", split="train", caption="A building appeared.", change=1):
    return {"pair_id":pair,"split":split,"dataset_name":"LEVIR-MCI","t1_path":"a.png","t2_path":"b.png","captions":[caption],"source_metadata":{"changeflag":change,"scene_id":pair}}

def test_generic_no_change_never_has_exact_positive():
    cap=captions_for_pair(row(caption="There is no difference.", change=0))[0]
    assert cap["query_scope"]=="generic_no_change"
    relevance=build_relevance([cap])[0]
    assert relevance["positive_pair_ids"]==[]
    assert relevance["ignored_pair_ids"]==[]
    assert relevance["valid_negative_policy"]=="semantic_group_only"

def test_exact_collisions_are_ignored_not_positive():
    caps=captions_for_pair(row("a",caption="new road"))+captions_for_pair(row("b",caption="new road"))
    assert build_relevance(caps)[0]["ignored_pair_ids"]==["b"]

def test_sequence_compatibility_views_are_preserved():
    pair=legacy_pair_to_v2(row())
    assert len(pair["frames"])==2 and pair["t1_path"]=="a.png"

def test_cross_split_source_scene_fails_audit():
    a=legacy_pair_to_v2(row("a","train")); b=legacy_pair_to_v2(row("b","development")); b["source_scene_group_id"]=a["source_scene_group_id"]
    assert not leakage_audit([a,b])["passed"]

def test_mask_keys_cannot_enter_mask_free_rows():
    try: mask_free({"mask_path":"x.png"})
    except ValueError: pass
    else: raise AssertionError("mask path accepted")

def test_caption_normalization_is_stable():
    assert normalize_text("New, ROAD!")=="new road"

def test_cross_split_shared_image_excludes_training_not_development(tmp_path):
    image=tmp_path/"shared.png"; image.write_bytes(b"same image")
    train=row("train","train"); development=row("development","val")
    train["t1_path"]=development["t2_path"]=str(image)
    kept,report=exclude_cross_split_image_conflicts([train,development])
    assert [item["pair_id"] for item in kept]==["development"]
    assert report["excluded_pair_ids_by_source_role"]=={"train":["train"]}
