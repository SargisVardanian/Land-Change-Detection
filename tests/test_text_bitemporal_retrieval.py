from __future__ import annotations

from land_change_detection.retrieval.backends.text_bitemporal_remoteclip import (
    TextBitemporalArchiveItem,
    TextBitemporalRemoteCLIPBackend,
    changed_region_pool,
)
from land_change_detection.retrieval.contracts import RetrievalMode, RetrievalQuery


def _build_backend() -> TextBitemporalRemoteCLIPBackend:
    archive_items = [
        TextBitemporalArchiveItem(
            item_id="pair-water-1",
            before_image_path="/tmp/before_water_1.tif",
            after_image_path="/tmp/after_water_1.tif",
            metadata={
                "transition_hint": "dry_to_wet",
                "sensor": "sentinel-2",
                "acquisition_date": "2025-05-01",
            },
            region_hint={"x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.6},
            thumbnail_path="/tmp/water_thumb.png",
            transition_hint="dry_to_wet",
        ),
        TextBitemporalArchiveItem(
            item_id="pair-urban-1",
            before_image_path="/tmp/before_urban_1.tif",
            after_image_path="/tmp/after_urban_1.tif",
            metadata={
                "transition_hint": "cropland_to_built_up",
                "sensor": "sentinel-2",
                "acquisition_date": "2025-06-01",
            },
            region_hint={"x0": 0.55, "y0": 0.2, "x1": 0.9, "y1": 0.7},
            thumbnail_path="/tmp/urban_thumb.png",
            transition_hint="cropland_to_built_up",
        ),
        TextBitemporalArchiveItem(
            item_id="pair-forest-1",
            before_image_path="/tmp/before_forest_1.tif",
            after_image_path="/tmp/after_forest_1.tif",
            metadata={
                "transition_hint": "forest_loss",
                "sensor": "landsat-8",
                "acquisition_date": "2025-07-01",
            },
            region_hint={"x0": 0.2, "y0": 0.55, "x1": 0.6, "y1": 0.95},
            thumbnail_path="/tmp/forest_thumb.png",
            transition_hint="forest_loss",
        ),
    ]
    return TextBitemporalRemoteCLIPBackend(
        model_dir=".",
        device="cpu",
        archive_items=archive_items,
    )


def test_text_bitemporal_backend_supports_free_text_queries():
    backend = _build_backend()
    query = RetrievalQuery(
        mode=RetrievalMode.TEXT_BITEMPORAL,
        text="wetting",
        top_k=2,
        region_hint={"x0": 0.05, "y0": 0.05, "x1": 0.45, "y1": 0.55},
    )

    artifact = backend.retrieve(query)

    assert artifact.result.backend_name == "text_bitemporal_remoteclip"
    assert len(artifact.items) == 2
    assert artifact.items[0].rank == 1
    assert artifact.items[0].metadata["query_text"] == "wetting"
    assert artifact.metadata["localized_query"] is True
    assert artifact.items[0].region is not None


def test_text_bitemporal_backend_supports_structured_transition_queries():
    backend = _build_backend()
    query = RetrievalQuery(
        mode=RetrievalMode.TRANSITION_CONDITIONED,
        top_k=3,
        filters={
            "from_class": "cropland",
            "to_class": "built_up",
            "transition_hint": "cropland_to_built_up",
        },
    )

    artifact = backend.retrieve(query)

    assert len(artifact.items) == 3
    assert artifact.result.metadata["query_text"] == "cropland to built_up"
    assert all(item.score <= 1.0 for item in artifact.items)
    assert all(item.transition_hint for item in artifact.items)


def test_changed_region_pool_is_deterministic_and_normalized():
    vector = [1.0, 2.0, 3.0]
    pooled = changed_region_pool(vector, {"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.5})

    assert len(pooled) == 3
    assert round(sum(value * value for value in pooled), 6) == 1.0
    assert pooled == changed_region_pool(vector, {"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.5})


def test_text_bitemporal_backend_requires_text_or_transition_filters():
    backend = _build_backend()
    query = RetrievalQuery(mode=RetrievalMode.TEXT_BITEMPORAL, top_k=1)

    try:
        backend.retrieve(query)
    except ValueError as exc:
        assert "free text or structured transition filters" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing query text.")
