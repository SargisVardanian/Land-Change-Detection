"""Global physical identity graph used for leakage-safe split checks."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable, Mapping


def _frame_nodes(frame: Mapping[str, Any]) -> set[str]:
    nodes = set()
    digest = str(frame.get("sha256") or "").strip()
    path = str(frame.get("path") or "").strip()
    if digest:
        nodes.add(f"image:{digest}")
    if path:
        nodes.add(f"file:{path}")
    return nodes


def _scene_node(item: Mapping[str, Any]) -> str:
    source = str(item.get("source") or "")
    scene = str(item.get("scene_id") or item.get("physical_group_id") or item.get("item_id"))
    return f"scene:{source}:{scene}"


def _identity_nodes(item: Mapping[str, Any]) -> set[str]:
    """Add only explicit provenance identities; namespace local IDs by source."""

    source = str(item.get("source") or "")
    provenance = item.get("provenance") if isinstance(item.get("provenance"), Mapping) else {}
    nodes = {_scene_node(item)}
    group_id = str(item.get("physical_group_id") or "").strip()
    if group_id:
        nodes.add(f"physical_group:{source}:{group_id}")
    local_keys = (
        "aoi_id",
        "parent_scene_id",
        "parent_image_id",
        "source_scene_group_id",
        "source_parent_id",
        "sequence_id",
        "temporal_overlap_id",
        "crop_id",
        "event_id",
    )
    for key in local_keys:
        value = str(provenance.get(key) or "").strip()
        if value:
            nodes.add(f"provenance:{key}:{source}:{value}")
    # These keys are reserved for a source-neutral identity supplied by the
    # source audit.  They are deliberately not inferred from ordinary names.
    for key in ("global_aoi_id", "global_parent_id", "global_scene_id", "global_event_id"):
        value = str(provenance.get(key) or "").strip()
        if value:
            nodes.add(f"global:{key}:{value}")
    return nodes


def build_identity_graph(items: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for item in items:
        nodes = _identity_nodes(item)
        for frame in item.get("frames", []):
            nodes.update(_frame_nodes(frame))
        for node in nodes:
            graph.setdefault(node, set())
        for left in nodes:
            graph[left].update(node for node in nodes if node != left)
    return dict(graph)


def connected_components(graph: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    unseen = set(graph)
    components: list[dict[str, Any]] = []
    while unseen:
        root = min(unseen)
        queue: deque[str] = deque([root])
        component: set[str] = set()
        unseen.remove(root)
        while queue:
            node = queue.popleft()
            component.add(node)
            for neighbor in graph.get(node, set()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append({"component_id": f"component:{len(components):06d}", "nodes": sorted(component)})
    return components
