"""Global physical identity graph used for leakage-safe split checks."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable, Mapping


def _frame_node(frame: Mapping[str, Any]) -> str:
    digest = str(frame.get("sha256") or "")
    if digest:
        return f"image:{digest}"
    return f"path:{frame.get('path', '')}"


def _scene_node(item: Mapping[str, Any]) -> str:
    source = str(item.get("source") or "")
    scene = str(item.get("scene_id") or item.get("physical_group_id") or item.get("item_id"))
    return f"scene:{source}:{scene}"


def build_identity_graph(items: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for item in items:
        nodes = [_scene_node(item)]
        nodes.extend(_frame_node(frame) for frame in item.get("frames", []))
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
