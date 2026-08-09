"""Build a compact, deterministic Dataset-v2 decision package."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def summarize(
    *,
    code_sha: str,
    branch: str,
    release_path: Path,
    source_registry: Iterable[Mapping[str, Any]],
    items: Iterable[Mapping[str, Any]],
    queries: Iterable[Mapping[str, Any]],
    integrity: Mapping[str, Any],
    statuses: Mapping[str, Any],
) -> dict[str, Any]:
    """Return JSON-compatible evidence without changing release policy."""

    item_rows = list(items)
    query_rows = list(queries)
    source_rows = list(source_registry)
    return {
        "schema_version": "qcpr-dataset-v2-decision-package-v1",
        "branch": branch,
        "code_sha": code_sha,
        "release_path": str(release_path),
        "integrity_passed": bool(integrity.get("passed")),
        "physical_item_count": len(item_rows),
        "frame_count": sum(len(row.get("frames") or []) for row in item_rows),
        "query_count": len(query_rows),
        "physical_items_by_source": dict(
            sorted(Counter(str(row.get("source")) for row in item_rows).items())
        ),
        "queries_by_scope": dict(
            sorted(Counter(str(row.get("query_scope")) for row in query_rows).items())
        ),
        "training_query_count": sum(bool(row.get("training_enabled")) for row in query_rows),
        "source_registry": source_rows,
        "view_status": dict(statuses),
        "integrity": dict(integrity),
    }


def write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    """Write a human-readable projection of the JSON decision package."""

    lines = [
        "# QCPR Dataset-v2 decision package",
        "",
        f"- Release: `{summary.get('release_path')}`",
        f"- Code: `{summary.get('code_sha')}`",
        f"- Integrity passed: `{bool(summary.get('integrity_passed'))}`",
        f"- Physical items: `{summary.get('physical_item_count', 0)}`",
        f"- Frames: `{summary.get('frame_count', 0)}`",
        f"- Queries: `{summary.get('query_count', 0)}`",
        f"- Training queries: `{summary.get('training_query_count', 0)}`",
        "",
        "## View status",
        "",
        "```json",
        json.dumps(summary.get("view_status") or {}, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

