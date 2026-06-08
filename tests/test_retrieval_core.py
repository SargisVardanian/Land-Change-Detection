from __future__ import annotations

import json

from land_change_detection.retrieval.contracts import (
    RetrievalItem,
    RetrievalMode,
    RetrievalQuery,
)
from land_change_detection.retrieval.evidence import build_retrieval_evidence
from land_change_detection.retrieval.indexing.manifests import RetrievalManifestItem
from land_change_detection.retrieval.indexing.vector_store import InMemoryVectorStore, cosine_similarity
from land_change_detection.retrieval.registry import (
    available_retrieval_backends,
    get_retrieval_backend,
)
from land_change_detection.retrieval.runtime import RetrievalRuntime


def test_retrieval_runtime_fake_backend_is_deterministic():
    runtime = RetrievalRuntime(backend_name="fake", model_dir=".", device="cpu")
    query = RetrievalQuery(
        mode=RetrievalMode.TEXT_BITEMPORAL,
        top_k=3,
        text="wetting",
        filters={"transition_hint": "dry_to_wet"},
    )

    artifact = runtime.run(query)

    assert artifact.mode is RetrievalMode.TEXT_BITEMPORAL
    assert [item.rank for item in artifact.items] == [1, 2, 3]
    assert [round(item.score, 3) for item in artifact.items] == [1.0, 0.9, 0.8]
    assert artifact.items[0].transition_hint == "dry_to_wet"
    assert artifact.items[0].item_id == "text_bitemporal:wetting:1"


def test_retrieval_contracts_are_json_serializable():
    item = RetrievalItem(
        item_id="sample-1",
        score=0.75,
        mode=RetrievalMode.STATIC_REGION,
        rank=1,
        metadata={"sensor": "sentinel-2"},
    )
    query = RetrievalQuery(mode=RetrievalMode.STATIC_REGION, image_path="/tmp/example.png")
    runtime = RetrievalRuntime(backend_name="noop", model_dir=".", device="cpu")
    artifact = runtime.run(query)
    evidence = build_retrieval_evidence(artifact)

    json.dumps(item.to_dict())
    json.dumps(query.to_dict())
    json.dumps(artifact.to_dict())
    json.dumps(evidence.to_dict())


def test_retrieval_registry_and_noop_backend():
    assert "fake" in available_retrieval_backends()
    assert "noop" in available_retrieval_backends()
    backend = get_retrieval_backend("noop", model_dir=".", device="cpu")
    artifact = backend.retrieve(RetrievalQuery(mode=RetrievalMode.TRAJECTORY, top_k=2))
    assert artifact.result.backend_name == "noop"
    assert artifact.items == []


def test_manifest_and_vector_store_helpers():
    store = InMemoryVectorStore(
        [
            RetrievalManifestItem(
                item_id="a",
                mode=RetrievalMode.STATIC_REGION,
                vector=[1.0, 0.0],
                metadata={"month": 5},
            ),
            RetrievalManifestItem(
                item_id="b",
                mode=RetrievalMode.STATIC_REGION,
                vector=[0.0, 1.0],
            ),
        ]
    )

    ranked = store.search([1.0, 0.0], top_k=2)

    assert ranked[0][0].item_id == "a"
    assert ranked[0][1] == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    roundtrip = RetrievalManifestItem.from_dict(store.items[0].to_dict())
    assert roundtrip.item_id == "a"
