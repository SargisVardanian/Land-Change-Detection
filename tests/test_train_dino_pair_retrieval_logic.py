from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_dino_pair_retrieval import (
    AlternatingTaskLoader,
    PositiveAwareBatchSampler,
    RetrievalSample,
    _build_subset_dataloader,
    compute_retrieval_metrics,
    validate_pair_id_split_integrity,
)


def test_positive_aware_batch_sampler_co_batches_positive_examples():
    samples = [
        RetrievalSample(
            sample_id="pair_a",
            pair_id="pair_a",
            split="unknown",
            before_path="a_before.png",
            after_path="a_after.png",
            caption="caption one",
            transition_label="urban_growth",
            dominant_transition="urban_growth",
            transition_histogram=None,
            source="levir",
        ),
        RetrievalSample(
            sample_id="pair_a#cap001",
            pair_id="pair_a",
            split="unknown",
            before_path="a_before.png",
            after_path="a_after.png",
            caption="caption two",
            transition_label="urban_growth",
            dominant_transition="urban_growth",
            transition_histogram=None,
            source="levir",
        ),
        RetrievalSample(
            sample_id="pair_b",
            pair_id="pair_b",
            split="unknown",
            before_path="b_before.png",
            after_path="b_after.png",
            caption=None,
            transition_label="1->2",
            dominant_transition="1->2",
            transition_histogram=[1.0, 0.0],
            source="SECOND-CC",
        ),
        RetrievalSample(
            sample_id="pair_c",
            pair_id="pair_c",
            split="unknown",
            before_path="c_before.png",
            after_path="c_after.png",
            caption=None,
            transition_label="1->2",
            dominant_transition="1->2",
            transition_histogram=[0.9, 0.1],
            source="SECOND-CC",
        ),
    ]
    sampler = PositiveAwareBatchSampler(samples, batch_size=2, shuffle=False, seed=7)

    batches = list(iter(sampler))

    assert batches[0] == [0, 1]
    assert batches[1] == [2, 3]


def test_compute_retrieval_metrics_checks_top_k_not_anywhere_in_ranking():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.1, 0.99],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = ["a", "b", "c"]
    dominant_transitions = ["forest->urban", "water->road", "forest->urban"]
    histograms = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ]

    metrics = compute_retrieval_metrics(
        embeddings,
        labels,
        dominant_transitions,
        histograms,
        None,
        [],
    )

    assert metrics["transition_recall@1"] == 0.0
    assert metrics["transition_recall@5"] == 1.0


def test_compute_retrieval_metrics_uses_pair_identity_for_text_queries():
    change_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    text_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    metrics = compute_retrieval_metrics(
        change_embeddings,
        ["pair_a", "pair_b", "pair_c"],
        ["urban", "urban", "water"],
        [None, None, None],
        text_embeddings,
        ["pair_a", "pair_c"],
    )

    assert metrics["recall@1"] == 1.0
    assert metrics["median_rank"] == 1.0


def test_compute_retrieval_metrics_measures_directionality_against_reversed_and_opposite():
    change_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    reversed_embeddings = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    text_embeddings = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    metrics = compute_retrieval_metrics(
        change_embeddings,
        ["pair_a", "pair_b"],
        ["urban_growth", "water_expansion"],
        [None, None],
        text_embeddings,
        ["pair_a"],
        ["urban_growth"],
        reversed_embeddings,
    )

    assert metrics["reversed_pair_sanity_accuracy"] == 1.0


def test_compute_retrieval_metrics_omits_transition_fields_for_pure_levir_caption_eval():
    change_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    text_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    metrics = compute_retrieval_metrics(
        change_embeddings,
        ["pair_a", "pair_b"],
        ["urban_growth", "water_expansion"],
        [None, None],
        text_embeddings,
        ["pair_a", "pair_b"],
    )

    assert "transition_recall@5" not in metrics
    assert "transition_top1_hit_rate" not in metrics
    assert "pair_sample_fraction" not in metrics


def test_validate_pair_id_split_integrity_rejects_leakage():
    samples = [
        RetrievalSample(
            sample_id="pair_a#cap000",
            pair_id="pair_a",
            split="train",
            before_path="a_before.png",
            after_path="a_after.png",
            caption="caption one",
            transition_label="urban_growth",
            dominant_transition="urban_growth",
            transition_histogram=None,
            source="levir",
        ),
        RetrievalSample(
            sample_id="pair_a#cap001",
            pair_id="pair_a",
            split="val",
            before_path="a_before.png",
            after_path="a_after.png",
            caption="caption two",
            transition_label="urban_growth",
            dominant_transition="urban_growth",
            transition_histogram=None,
            source="levir",
        ),
    ]

    try:
        validate_pair_id_split_integrity(samples)
    except ValueError as exc:
        assert "pair_id leakage" in str(exc)
    else:
        raise AssertionError("Expected pair_id leakage validation to fail.")


def test_alternating_task_loader_keeps_caption_and_transition_batches_separate():
    temp_root = Path(__file__).resolve().parent / ".tmp_train_logic"
    temp_root.mkdir(parents=True, exist_ok=True)
    for name in ("a_before.png", "a_after.png", "b_before.png", "b_after.png", "c_before.png", "c_after.png"):
        Image.new("RGB", (8, 8), color=(1, 2, 3)).save(temp_root / name)
    samples = [
        RetrievalSample(
            sample_id="pair_a",
            pair_id="pair_a",
            split="unknown",
            before_path=str(temp_root / "a_before.png"),
            after_path=str(temp_root / "a_after.png"),
            caption="caption one",
            transition_label="urban_growth",
            dominant_transition="urban_growth",
            transition_histogram=None,
            source="levir",
        ),
        RetrievalSample(
            sample_id="pair_a#cap001",
            pair_id="pair_a",
            split="unknown",
            before_path=str(temp_root / "a_before.png"),
            after_path=str(temp_root / "a_after.png"),
            caption="caption two",
            transition_label="urban_growth",
            dominant_transition="urban_growth",
            transition_histogram=None,
            source="levir",
        ),
        RetrievalSample(
            sample_id="pair_b",
            pair_id="pair_b",
            split="unknown",
            before_path=str(temp_root / "b_before.png"),
            after_path=str(temp_root / "b_after.png"),
            caption=None,
            transition_label="1->2",
            dominant_transition="1->2",
            transition_histogram=[1.0, 0.0],
            source="SECOND-CC",
        ),
        RetrievalSample(
            sample_id="pair_c",
            pair_id="pair_c",
            split="unknown",
            before_path=str(temp_root / "c_before.png"),
            after_path=str(temp_root / "c_after.png"),
            caption=None,
            transition_label="1->2",
            dominant_transition="1->2",
            transition_histogram=[0.9, 0.1],
            source="SECOND-CC",
        ),
    ]
    args = type("Args", (), {"image_size": 32, "batch_size": 2, "num_workers": 0, "seed": 7})()
    caption_loader = _build_subset_dataloader([sample for sample in samples if sample.caption], args, shuffle=False)
    transition_loader = _build_subset_dataloader([sample for sample in samples if sample.transition_histogram], args, shuffle=False)
    loader = AlternatingTaskLoader([caption_loader, transition_loader])

    batches = list(loader)

    assert all(caption is not None for caption in batches[0]["caption"])
    assert all(hist is not None for hist in batches[1]["transition_histogram"])
