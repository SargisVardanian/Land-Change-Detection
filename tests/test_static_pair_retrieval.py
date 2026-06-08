from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from land_change_detection.retrieval.backends.pair_analog_prithvi import PairAnalogPrithviBackend
from land_change_detection.retrieval.backends.static_region_prithvi import StaticRegionPrithviBackend
from land_change_detection.retrieval.contracts import RetrievalMode, RetrievalQuery
from land_change_detection.retrieval.registry import register_retrieval_backend
from land_change_detection.retrieval.runtime import RetrievalRuntime


def _load_script_module(script_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / script_name
    spec = spec_from_file_location(script_name.replace(".py", ""), module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_index = _load_script_module("build_retrieval_index.py").build_index
query_index = _load_script_module("query_retrieval_index.py").query_index


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows))


def test_static_region_index_build_and_query(tmp_path: Path):
    manifest_path = tmp_path / "static_manifest.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "item_id": "water-spring-s2",
                "image_path": "/tmp/water_1.png",
                "sensor": "sentinel-2",
                "region_id": "armenia-north",
                "transition_label": "water",
                "metadata": {"month": 5, "geography": "armenia"},
            },
            {
                "item_id": "urban-summer-s2",
                "image_path": "/tmp/urban_1.png",
                "sensor": "sentinel-2",
                "region_id": "armenia-south",
                "transition_label": "built_up",
                "metadata": {"month": 8, "geography": "armenia"},
            },
        ],
    )
    index_path = tmp_path / "static_region_index.json"
    build_index("static_region", manifest_path, index_path, embedding_backend_name="fake", prefer_faiss=True)

    register_retrieval_backend("static_region_prithvi_test", StaticRegionPrithviBackend)
    runtime = RetrievalRuntime(backend_name="static_region_prithvi_test", model_dir=index_path, device="cpu")
    artifact = runtime.run(
        RetrievalQuery(
            mode=RetrievalMode.STATIC_REGION,
            top_k=2,
            text="water",
            image_path="/tmp/query_water.png",
            filters={"month": 5, "sensor": "sentinel-2", "region_id": "armenia-north", "transition_label": "water"},
        )
    )

    assert [item.item_id for item in artifact.items] == ["water-spring-s2", "urban-summer-s2"]
    assert artifact.items[0].metadata["vector_store"] == "numpy"
    assert artifact.items[0].metadata["embedding_backend"] == "numpy"
    assert artifact.items[0].transition_hint == "water"


def test_pair_analog_index_build_and_query(tmp_path: Path):
    manifest_path = tmp_path / "pair_manifest.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "item_id": "wetting-may-s2",
                "before_image_path": "/tmp/before_wet_a.png",
                "after_image_path": "/tmp/after_wet_a.png",
                "sensor": "sentinel-2",
                "source": "oscd",
                "region_id": "armenia-lake",
                "transition_label": "dry_to_wet",
                "metadata": {"month": 5, "geography": "armenia"},
            },
            {
                "item_id": "urban-july-s1",
                "before_image_path": "/tmp/before_urban_b.png",
                "after_image_path": "/tmp/after_urban_b.png",
                "sensor": "sentinel-1",
                "source": "custom",
                "region_id": "armenia-yerevan",
                "transition_label": "cropland_to_built_up",
                "metadata": {"month": 7, "geography": "armenia"},
            },
        ],
    )
    index_path = tmp_path / "pair_analog_index.json"
    build_index("pair_analog", manifest_path, index_path, embedding_backend_name="fake", prefer_faiss=False)

    register_retrieval_backend("pair_analog_prithvi_test", PairAnalogPrithviBackend)
    runtime = RetrievalRuntime(backend_name="pair_analog_prithvi_test", model_dir=index_path, device="cpu")
    artifact = runtime.run(
        RetrievalQuery(
            mode=RetrievalMode.PAIR_ANALOG,
            top_k=2,
            text="wetting",
            before_image_path="/tmp/query_before.png",
            after_image_path="/tmp/query_after.png",
            filters={"month": 5, "sensor": "sentinel-2", "region_id": "armenia-lake", "transition_label": "dry_to_wet"},
        )
    )

    assert artifact.items[0].item_id == "wetting-may-s2"
    assert artifact.items[0].before_image_path == "/tmp/before_wet_a.png"
    assert artifact.items[0].after_image_path == "/tmp/after_wet_a.png"
    assert artifact.items[0].transition_hint == "dry_to_wet"


def test_query_script_returns_ranked_json(tmp_path: Path):
    manifest_path = tmp_path / "static_query_manifest.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "item_id": "water-one",
                "image_path": "/tmp/static_a.png",
                "sensor": "sentinel-2",
                "metadata": {"month": 4, "geography": "armenia"},
            }
        ],
    )
    index_path = tmp_path / "static_region_index.json"
    build_index("static_region", manifest_path, index_path, embedding_backend_name="numpy", prefer_faiss=False)

    result = query_index(
        type(
            "Args",
            (),
            {
                "mode": "static_region",
                "index": index_path,
                "top_k": 1,
                "embedding_backend": "numpy",
                "prefer_faiss": False,
                "image_path": "/tmp/static_query.png",
                "before_image_path": None,
                "after_image_path": None,
                "item_id": None,
                "text": "water",
                "month": 4,
                "season": None,
                "sensor": "sentinel-2",
                "geography": "armenia",
                "region_id": None,
                "transition_label": None,
            },
        )()
    )

    assert result["result"]["items"][0]["item_id"] == "water-one"
    assert result["result"]["metadata"]["vector_store"] == "numpy"
