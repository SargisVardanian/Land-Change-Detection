from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GB = 1_000_000_000
HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class DatasetRegistryEntry:
    canonical_name: str
    phase: str
    source_type: str
    source_identifier: str
    expected_download_bytes: int | None
    expected_extracted_bytes: int | None
    size_status: str
    automatic_download_supported: bool
    manual_download_required: bool
    license: str
    redistribution_allowed: bool | None
    expected_directory: str
    required_files_or_directories: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    download_handler: str = "none"
    disabled_by_default: bool = False
    category: str = "dataset"
    fallback_directories: tuple[str, ...] = ()

    def expected_path(self, project_root: Path) -> Path:
        return project_root / self.expected_directory

    def fallback_paths(self, project_root: Path) -> list[Path]:
        return [project_root / rel for rel in self.fallback_directories]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dataset_registry() -> tuple[DatasetRegistryEntry, ...]:
    return (
        DatasetRegistryEntry(
            canonical_name="LEVIR-CC",
            phase="baseline",
            source_type="huggingface",
            source_identifier="lcybuaa/LEVIR-CC",
            expected_download_bytes=int(2.68 * GB),
            expected_extracted_bytes=int(5.00 * GB),
            size_status="verified",
            automatic_download_supported=True,
            manual_download_required=False,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="datasets/raw/LEVIR-CC",
            notes=("clean first-stage text-to-pair retrieval benchmark",),
            download_handler="hf_dataset",
        ),
        DatasetRegistryEntry(
            canonical_name="LEVIR-MCI",
            phase="baseline",
            source_type="huggingface",
            source_identifier="lcybuaa/LEVIR-MCI",
            expected_download_bytes=int(2.77 * GB),
            expected_extracted_bytes=int(5.00 * GB),
            size_status="verified",
            automatic_download_supported=True,
            manual_download_required=False,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="datasets/raw/LEVIR-MCI-unpacked/LEVIR-MCI-dataset",
            required_files_or_directories=("images",),
            notes=("grounded retrieval dataset with binary masks",),
            download_handler="hf_dataset",
            fallback_directories=("datasets/raw/LEVIR-MCI",),
        ),
        DatasetRegistryEntry(
            canonical_name="SECOND-CC",
            phase="baseline",
            source_type="zenodo",
            source_identifier="10.5281/zenodo.16937571",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=True,
            manual_download_required=False,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="datasets/raw/SECOND-CC",
            notes=("transition-aware bridge dataset; size must be discovered first",),
            download_handler="zenodo_dataset",
        ),
        DatasetRegistryEntry(
            canonical_name="DINOv2-B-14",
            phase="baseline",
            source_type="huggingface",
            source_identifier="facebook/dinov2-base",
            expected_download_bytes=int(0.45 * GB),
            expected_extracted_bytes=int(0.60 * GB),
            size_status="verified",
            automatic_download_supported=True,
            manual_download_required=False,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="models/dinov2-base",
            required_files_or_directories=("config.json",),
            notes=("first foundation visual backbone target",),
            download_handler="hf_model",
            category="model",
        ),
        DatasetRegistryEntry(
            canonical_name="RemoteCLIP-RN50",
            phase="baseline",
            source_type="huggingface",
            source_identifier="chendelong/RemoteCLIP",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=True,
            manual_download_required=False,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="models/remoteclip-rn50",
            required_files_or_directories=("config.json",),
            notes=("single optional RemoteCLIP checkpoint; block until size is known",),
            download_handler="hf_model",
            category="model",
        ),
        DatasetRegistryEntry(
            canonical_name="RSICRC",
            phase="baseline",
            source_type="git",
            source_identifier="https://github.com/Chen-Yang-Liu/RSICC.git",
            expected_download_bytes=int(0.05 * GB),
            expected_extracted_bytes=int(0.10 * GB),
            size_status="verified",
            automatic_download_supported=True,
            manual_download_required=False,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="code/reference/RSICRC",
            notes=("reference captioning/retrieval repo",),
            download_handler="git_repo",
            category="reference_repo",
        ),
        DatasetRegistryEntry(
            canonical_name="SecondCC-MModalCC",
            phase="baseline",
            source_type="git",
            source_identifier="https://github.com/ChangeCapsInRS/SecondCC.git",
            expected_download_bytes=int(0.05 * GB),
            expected_extracted_bytes=int(0.10 * GB),
            size_status="verified",
            automatic_download_supported=True,
            manual_download_required=False,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="code/reference/SecondCC",
            notes=("SecondCC/MModalCC reference repo",),
            download_handler="git_repo",
            category="reference_repo",
        ),
        DatasetRegistryEntry(
            canonical_name="Change-Agent",
            phase="baseline",
            source_type="git",
            source_identifier="https://github.com/hanlinwu/ChangeChat.git",
            expected_download_bytes=int(0.05 * GB),
            expected_extracted_bytes=int(0.10 * GB),
            size_status="verified",
            automatic_download_supported=True,
            manual_download_required=False,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="code/reference/Change-Agent",
            notes=("Change-Agent style reference repo for grounded change captioning",),
            download_handler="git_repo",
            category="reference_repo",
        ),
        DatasetRegistryEntry(
            canonical_name="VisTA",
            phase="baseline",
            source_type="git",
            source_identifier="https://github.com/like413/VisTA.git",
            expected_download_bytes=int(0.05 * GB),
            expected_extracted_bytes=int(0.10 * GB),
            size_status="verified",
            automatic_download_supported=True,
            manual_download_required=False,
            license="research-only",
            redistribution_allowed=False,
            expected_directory="code/reference/VisTA",
            notes=("repo only; QAG-360K remains manual",),
            download_handler="git_repo",
            category="reference_repo",
        ),
        DatasetRegistryEntry(
            canonical_name="TERRA-CD-Repo",
            phase="baseline",
            source_type="git",
            source_identifier="https://github.com/omkarsoak/TERRA-CD.git",
            expected_download_bytes=int(0.05 * GB),
            expected_extracted_bytes=int(0.10 * GB),
            size_status="verified",
            automatic_download_supported=True,
            manual_download_required=False,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="code/reference/TERRA-CD",
            notes=("repo only; dataset archive remains manual",),
            download_handler="git_repo",
            category="reference_repo",
        ),
        DatasetRegistryEntry(
            canonical_name="Hi-UCD-S",
            phase="baseline",
            source_type="git",
            source_identifier="https://github.com/Daisy-7/Hi-UCD-S.git",
            expected_download_bytes=int(0.05 * GB),
            expected_extracted_bytes=int(0.10 * GB),
            size_status="verified",
            automatic_download_supported=True,
            manual_download_required=False,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="code/reference/Hi-UCD-S",
            notes=("repo only; dataset remains manual",),
            download_handler="git_repo",
            category="reference_repo",
        ),
        DatasetRegistryEntry(
            canonical_name="Hi-UCD",
            phase="manual",
            source_type="manual",
            source_identifier="manual-request",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=False,
            manual_download_required=True,
            license="research-only",
            redistribution_allowed=False,
            expected_directory="datasets/manual/Hi-UCD",
            notes=("manual OneDrive/Baidu/request workflow",),
        ),
        DatasetRegistryEntry(
            canonical_name="QAG-360K",
            phase="manual",
            source_type="manual",
            source_identifier="manual-google-drive",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=False,
            manual_download_required=True,
            license="research-only",
            redistribution_allowed=False,
            expected_directory="datasets/manual/QAG-360K",
            notes=("manual access; test split often requires author contact",),
        ),
        DatasetRegistryEntry(
            canonical_name="TERRA-CD-Data",
            phase="manual",
            source_type="manual",
            source_identifier="manual-download-required",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=False,
            manual_download_required=True,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="datasets/manual/TERRA-CD",
            notes=("manual dataset archive placement",),
        ),
        DatasetRegistryEntry(
            canonical_name="SECOND-CC-Manual",
            phase="manual",
            source_type="manual",
            source_identifier="manual-fallback",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=False,
            manual_download_required=True,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="datasets/manual/SECOND-CC",
            notes=("manual fallback when Zenodo automation is unavailable",),
        ),
        DatasetRegistryEntry(
            canonical_name="DynamicEarthNet",
            phase="deferred",
            source_type="manual",
            source_identifier="https://mediatum.ub.tum.de/1650201",
            expected_download_bytes=int(524.0 * GB),
            expected_extracted_bytes=int(560.0 * GB),
            size_status="verified",
            automatic_download_supported=False,
            manual_download_required=True,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="datasets/raw/DynamicEarthNet",
            notes=("never download in baseline mode",),
            disabled_by_default=True,
        ),
        DatasetRegistryEntry(
            canonical_name="SpaceNet7",
            phase="deferred",
            source_type="manual",
            source_identifier="manual-download-required",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=False,
            manual_download_required=True,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="datasets/raw/SpaceNet7",
            disabled_by_default=True,
        ),
        DatasetRegistryEntry(
            canonical_name="SSL4EO-S12",
            phase="deferred",
            source_type="manual",
            source_identifier="manual-download-required",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=False,
            manual_download_required=True,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="datasets/raw/SSL4EO-S12",
            disabled_by_default=True,
        ),
        DatasetRegistryEntry(
            canonical_name="RS5M",
            phase="deferred",
            source_type="manual",
            source_identifier="manual-download-required",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=False,
            manual_download_required=True,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="datasets/raw/RS5M",
            disabled_by_default=True,
        ),
        DatasetRegistryEntry(
            canonical_name="BigEarthNet",
            phase="deferred",
            source_type="manual",
            source_identifier="manual-download-required",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=False,
            manual_download_required=True,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="datasets/raw/BigEarthNet",
            disabled_by_default=True,
        ),
        DatasetRegistryEntry(
            canonical_name="TinyRS-R1",
            phase="deferred",
            source_type="manual",
            source_identifier="manual-download-required",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=False,
            manual_download_required=True,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="models/tinyrs-r1",
            disabled_by_default=True,
            category="model",
        ),
        DatasetRegistryEntry(
            canonical_name="RSUniVLM",
            phase="deferred",
            source_type="manual",
            source_identifier="manual-download-required",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=False,
            manual_download_required=True,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="models/rsunivlm",
            disabled_by_default=True,
            category="model",
        ),
        DatasetRegistryEntry(
            canonical_name="DeltaVLM",
            phase="deferred",
            source_type="manual",
            source_identifier="manual-download-required",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=False,
            manual_download_required=True,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="models/deltavlm",
            disabled_by_default=True,
            category="model",
        ),
        DatasetRegistryEntry(
            canonical_name="Prithvi-EO-2.0-300M-TL",
            phase="deferred",
            source_type="huggingface",
            source_identifier="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=True,
            manual_download_required=False,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="models/Prithvi-EO-2.0-300M-TL",
            disabled_by_default=True,
            category="model",
            download_handler="hf_model",
        ),
        DatasetRegistryEntry(
            canonical_name="Prithvi-EO-2.0-600M-TL",
            phase="deferred",
            source_type="huggingface",
            source_identifier="ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
            expected_download_bytes=None,
            expected_extracted_bytes=None,
            size_status="unknown",
            automatic_download_supported=True,
            manual_download_required=False,
            license="unknown",
            redistribution_allowed=None,
            expected_directory="models/Prithvi-EO-2.0-600M-TL",
            disabled_by_default=True,
            category="model",
            download_handler="hf_model",
        ),
    )


def registry_entry_by_name(name: str) -> DatasetRegistryEntry:
    for entry in dataset_registry():
        if entry.canonical_name == name:
            return entry
    raise KeyError(name)


def collect_phase_entries(phase: str) -> list[DatasetRegistryEntry]:
    registry = list(dataset_registry())
    if phase == "baseline":
        return [entry for entry in registry if entry.phase == "baseline"]
    if phase == "manual":
        return [entry for entry in registry if entry.phase == "manual"]
    if phase == "deferred":
        return [entry for entry in registry if entry.phase == "deferred"]
    if phase == "all":
        return registry
    raise ValueError(f"Unsupported phase: {phase}")


def _path_has_required_content(path: Path, required: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    if not required:
        return any(path.iterdir()) if path.is_dir() else True
    return all((path / rel).exists() for rel in required)


def detect_existing_entry(entry: DatasetRegistryEntry, project_root: Path) -> tuple[bool, Path | None]:
    primary = entry.expected_path(project_root)
    if _path_has_required_content(primary, entry.required_files_or_directories):
        return True, primary
    for candidate in entry.fallback_paths(project_root):
        if _path_has_required_content(candidate, entry.required_files_or_directories):
            return True, candidate
    return False, None


def current_hf_home(project_root: Path) -> Path:
    return project_root / "cache" / "huggingface"


def _hf_cache_path(project_root: Path, entry: DatasetRegistryEntry) -> Path | None:
    if entry.source_type != "huggingface":
        return None
    hub_root = current_hf_home(project_root) / "hub"
    prefix = "models" if entry.category == "model" else "datasets"
    repo_slug = entry.source_identifier.replace("/", "--")
    return hub_root / f"{prefix}--{repo_slug}"


def _has_hf_cache(project_root: Path, entry: DatasetRegistryEntry) -> tuple[bool, str | None]:
    cache_path = _hf_cache_path(project_root, entry)
    if cache_path is None:
        return False, None
    return cache_path.exists(), str(cache_path) if cache_path.exists() else None


def _needs_unknown_size_block(entry: DatasetRegistryEntry) -> bool:
    return entry.automatic_download_supported and entry.expected_download_bytes is None


def _entry_plan(
    entry: DatasetRegistryEntry,
    project_root: Path,
    *,
    allow_unknown_size: bool,
    max_download_bytes: int,
    baseline_mode: bool,
) -> dict[str, Any]:
    exists, existing_path = detect_existing_entry(entry, project_root)
    cached, cached_path = _has_hf_cache(project_root, entry)
    blocked_reasons: list[str] = []
    if baseline_mode and entry.canonical_name == "DynamicEarthNet":
        blocked_reasons.append("dynamicearthnet_disabled_in_baseline_phase")
    if entry.disabled_by_default:
        blocked_reasons.append("disabled_by_default")
    if entry.manual_download_required:
        blocked_reasons.append("manual_download_required")
    if _needs_unknown_size_block(entry) and not allow_unknown_size:
        blocked_reasons.append("unknown_size_requires_allow_unknown_size")
    if entry.expected_download_bytes is not None and entry.expected_download_bytes > max_download_bytes:
        blocked_reasons.append("exceeds_max_download_cap")
    return {
        "canonical_name": entry.canonical_name,
        "phase": entry.phase,
        "source_type": entry.source_type,
        "source_identifier": entry.source_identifier,
        "expected_directory": entry.expected_directory,
        "expected_download_bytes": entry.expected_download_bytes,
        "expected_extracted_bytes": entry.expected_extracted_bytes,
        "size_status": entry.size_status,
        "automatic_download_supported": entry.automatic_download_supported,
        "manual_download_required": entry.manual_download_required,
        "download_handler": entry.download_handler,
        "exists": exists,
        "existing_path": str(existing_path) if existing_path else None,
        "cached": cached,
        "cached_path": cached_path,
        "planned_download_bytes": 0 if exists else entry.expected_download_bytes,
        "planned_extracted_bytes": 0 if exists else entry.expected_extracted_bytes,
        "blocked_reasons": blocked_reasons,
        "allowed": not blocked_reasons and entry.automatic_download_supported and not exists,
        "notes": list(entry.notes),
        "category": entry.category,
    }


def build_download_plan(
    project_root: Path,
    *,
    phase: str,
    reserve_gb: float,
    max_download_gb: float,
    allow_unknown_size: bool = False,
) -> dict[str, Any]:
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "datasets" / "raw").mkdir(parents=True, exist_ok=True)
    hf_home = current_hf_home(project_root)
    hf_home.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(project_root).free
    reserve_bytes = int(reserve_gb * GB)
    max_download_bytes = int(max_download_gb * GB)
    plans = [
        _entry_plan(
            entry,
            project_root,
            allow_unknown_size=allow_unknown_size,
            max_download_bytes=max_download_bytes,
            baseline_mode=(phase == "baseline"),
        )
        for entry in collect_phase_entries(phase)
    ]
    totals = {
        "planned_download_bytes": sum(int(row["planned_download_bytes"] or 0) for row in plans if row["allowed"]),
        "planned_extracted_bytes": sum(int(row["planned_extracted_bytes"] or 0) for row in plans if row["allowed"]),
    }
    projected_free_after_download = free_bytes - totals["planned_download_bytes"]
    projected_free_after_extract = projected_free_after_download - totals["planned_extracted_bytes"]
    policy_violations: list[str] = []
    if totals["planned_download_bytes"] > max_download_bytes:
        policy_violations.append("planned_download_bytes_exceed_cap")
    if projected_free_after_extract < reserve_bytes:
        policy_violations.append("reserve_space_violated")
    return {
        "phase": phase,
        "project_root": str(project_root),
        "hf_home": str(hf_home),
        "free_bytes": free_bytes,
        "reserve_bytes": reserve_bytes,
        "max_download_bytes": max_download_bytes,
        "allow_unknown_size": allow_unknown_size,
        "entries": plans,
        "totals": totals,
        "projected_free_after_download_bytes": projected_free_after_download,
        "projected_free_after_extract_bytes": projected_free_after_extract,
        "policy_violations": policy_violations,
    }


def render_human_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"Phase: {plan['phase']}",
        f"Project root: {plan['project_root']}",
        f"HF_HOME: {plan['hf_home']}",
        f"Free space: {plan['free_bytes'] / GB:.2f} GB",
        f"Reserve target: {plan['reserve_bytes'] / GB:.2f} GB",
        f"Download cap: {plan['max_download_bytes'] / GB:.2f} GB",
        "",
        "Entries:",
    ]
    for row in plan["entries"]:
        status = "ALREADY_PRESENT" if row["exists"] else "ALLOW" if row["allowed"] else "BLOCK"
        lines.append(
            f"- {row['canonical_name']} [{status}] "
            f"download={((row['planned_download_bytes'] or 0) / GB):.2f} GB "
            f"extract={((row['planned_extracted_bytes'] or 0) / GB):.2f} GB"
        )
        if row["blocked_reasons"]:
            lines.append(f"  reasons: {', '.join(row['blocked_reasons'])}")
        if row["existing_path"]:
            lines.append(f"  existing: {row['existing_path']}")
        if row["cached_path"]:
            lines.append(f"  cache: {row['cached_path']}")
    lines.extend(
        [
            "",
            f"Planned download total: {plan['totals']['planned_download_bytes'] / GB:.2f} GB",
            f"Planned extracted total: {plan['totals']['planned_extracted_bytes'] / GB:.2f} GB",
            f"Projected free after download: {plan['projected_free_after_download_bytes'] / GB:.2f} GB",
            f"Projected free after extract: {plan['projected_free_after_extract_bytes'] / GB:.2f} GB",
        ]
    )
    if plan["policy_violations"]:
        lines.append(f"Policy violations: {', '.join(plan['policy_violations'])}")
    else:
        lines.append("Policy violations: none")
    return "\n".join(lines) + "\n"


def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_download_checksums(project_root: Path, plan: dict[str, Any]) -> Path:
    reports_root = project_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"entries": []}
    for row in plan["entries"]:
        entry = registry_entry_by_name(str(row["canonical_name"]))
        files: list[dict[str, Any]] = []
        target_root = entry.expected_path(project_root)
        for path in _iter_files(target_root):
            files.append(
                {
                    "path": str(path.relative_to(project_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        payload["entries"].append({"canonical_name": entry.canonical_name, "files": files})
    path = reports_root / "download_checksums.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_download_inventory(project_root: Path, plan: dict[str, Any]) -> Path:
    reports_root = project_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    path = reports_root / "download_inventory.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return path
