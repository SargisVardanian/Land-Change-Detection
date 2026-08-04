"""Event-disjoint split validator."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def validate_event_disjoint(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    events: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item.get("event_id") is not None:
            events[str(item["event_id"])].add(str(item.get("split")))
    leaks = sorted(event for event, splits in events.items() if len(splits) > 1)
    return {"event_count": len(events), "cross_split_events": leaks, "passed": not leaks}
