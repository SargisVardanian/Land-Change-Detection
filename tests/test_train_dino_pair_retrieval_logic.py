from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_dino_pair_retrieval import (
    PositiveAwareBatchSampler,
    RetrievalSample,
    compute_retrieval_metrics,
)


def test_positive_aware_batch_sampler_co_batches_positive_examples():
    samples = [
        RetrievalSample(
            sample_id="pair_a",
            before_path="a_before.png",
            after_path="a_after.png",
            caption="caption one",
            transition_label="urban_growth",
            dominant_transition="urban_growth",
            transition_histogram=None,
            source="levir",
        ),
        RetrievalSample(
            sample_id="pair_a",
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
        text_batches=0,
    )

    assert metrics["transition_recall@1"] == 0.0
    assert metrics["transition_recall@5"] == 1.0
