from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def safe_git_commit(cwd: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def path_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    if path.is_file():
        return {
            "path": str(path),
            "kind": "file",
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        rel = str(item.relative_to(path)).encode("utf-8")
        digest.update(rel)
        size = item.stat().st_size
        total_bytes += size
        digest.update(str(size).encode("utf-8"))
    return {
        "path": str(path),
        "kind": "directory",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "bytes": total_bytes,
    }


def jsonl_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "num_rows": len(rows),
    }


def locate_storage_inventory(project_root: Path | None) -> str | None:
    if project_root is None:
        return None
    candidate = project_root / "reports" / "download_inventory.json"
    return str(candidate) if candidate.exists() else None


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
