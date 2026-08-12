#!/usr/bin/env python3
"""Validate and, only after human review, promote the QCPR r20 exact sample.

The r20 review package is deliberately a decision package rather than a
training set.  This script is the only promotion path for the calibrated
exact sample.  It refuses to infer labels from model output, source
attributes, candidate-neighbour selection, or the empty review templates.

The exact gate is explicit: callers must provide ``--min-exact-precision``.
The default gate statistic is the lower bound of the two-sided Wilson 95%
confidence interval.  A candidate manifest is materialized only when all
1,500 rows have two complete independent reviews and a complete adjudication,
and that lower bound meets the supplied threshold.

The promoted manifest is intentionally conservative.  It contains only the
reviewed exact-stratum rows adjudicated as EXACT (including a verified human
rewrite when the adjudicator supplied one); unreviewed r19g rows are never
silently extrapolated into r20.  The immutable r19g manifests are read-only
inputs and are never edited in place.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRATA = (
    "exact_discriminative",
    "semantic_multi_positive",
    "generic_no_change",
    "stable_scene_specific",
    "localized_direction",
)
DECISIONS = (
    "EXACT",
    "SEMANTIC_ONLY",
    "AMBIGUOUS_IGNORE",
    "REWRITE_REQUIRED",
    "REJECT",
)
QUESTION_VALUES = {"yes", "no", "uncertain"}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != "artifact_sha256"}
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_no}")
            rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def identity_key(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def parse_timestamp(value: Any) -> bool:
    if not nonempty(value):
        return False
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def attested(value: Any) -> bool:
    if value is True:
        return True
    if not nonempty(value):
        return False
    return str(value).strip().casefold() not in {"no", "false", "0", "not independent"}


def wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt((p * (1.0 - p) / total) + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def proportion(successes: int, total: int) -> dict[str, Any]:
    interval = wilson(successes, total)
    return {
        "successes": successes,
        "total": total,
        "value": successes / total if total else None,
        "ci95": interval,
    }


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if not left or len(left) != len(right):
        return None
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / total
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[label] / total) * (right_counts[label] / total)
        for label in set(left_counts) | set(right_counts)
    )
    if math.isclose(1.0 - expected, 0.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise ValueError(f"required file is missing: {path}")


def load_packets(review_package: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    packets: dict[str, dict[str, dict[str, Any]]] = {}
    for reviewer in ("reviewer_a", "reviewer_b"):
        path = review_package / f"{reviewer}_packet.jsonl"
        require_file(path)
        rows = read_jsonl(path)
        if len(rows) != 1500:
            raise ValueError(f"{path.name} must contain 1500 rows, got {len(rows)}")
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            sample_id = str(row.get("review_sample_id") or "")
            if not sample_id or sample_id in by_id:
                raise ValueError(f"{path.name} has missing or duplicate review_sample_id")
            if row.get("model_scores_included") is not False:
                raise ValueError(f"{path.name} exposes model scores or omits the false flag")
            if row.get("rankings_included") is not False or row.get("masks_included") is not False:
                raise ValueError(f"{path.name} is not an independent text/image review packet")
            if row.get("event_or_source_metadata_included") is not False:
                raise ValueError(f"{path.name} exposes event/source metadata")
            if row.get("paths_are_blind_aliases") is not True or row.get("visible_internal_ids") is not False:
                raise ValueError(f"{path.name}:{sample_id} is not a blind review packet")
            stratum = str(row.get("stratum") or "")
            if stratum not in STRATA:
                raise ValueError(f"{path.name} contains unknown stratum {stratum!r}")
            if not nonempty(row.get("caption")):
                raise ValueError(f"{path.name}:{sample_id} has no caption")
            true_pair = row.get("true_pair") or {}
            if not nonempty(true_pair.get("t1_path")) or not nonempty(true_pair.get("t2_path")):
                raise ValueError(f"{path.name}:{sample_id} has no blind intended pair assets")
            forbidden_visible_keys = {
                "query_id",
                "source_pair_id",
                "source_dataset",
                "source_event_id",
                "event_id",
            }
            if forbidden_visible_keys.intersection(row):
                raise ValueError(f"{path.name}:{sample_id} exposes canonical/source identifiers")
            for neighbour in row.get("candidate_neighbours") or []:
                if forbidden_visible_keys.intersection(neighbour) or "candidate_id" in neighbour:
                    raise ValueError(f"{path.name}:{sample_id} exposes a candidate identifier")
            by_id[sample_id] = row
        packets[reviewer] = by_id
    if set(packets["reviewer_a"]) != set(packets["reviewer_b"]):
        raise ValueError("reviewer packets do not contain the same 1,500 sample IDs")
    for sample_id in packets["reviewer_a"]:
        left = packets["reviewer_a"][sample_id]
        right = packets["reviewer_b"][sample_id]
        for key in ("caption", "stratum"):
            if left.get(key) != right.get(key):
                raise ValueError(f"reviewer packets disagree for {sample_id}: {key}")
    return packets["reviewer_a"], packets["reviewer_b"]


def load_internal_ledger(review_package: Path) -> dict[str, dict[str, Any]]:
    """Load the producer-only join ledger kept outside reviewer materials."""
    path = review_package.parent / "internal_review_join_ledger.jsonl"
    require_file(path)
    rows = read_jsonl(path)
    if len(rows) != 1500:
        raise ValueError(f"{path.name} must contain 1500 rows, got {len(rows)}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("review_sample_id") or "")
        if not sample_id or sample_id in by_id:
            raise ValueError(f"{path.name} has missing or duplicate review_sample_id")
        if not nonempty(row.get("query_id")) or not nonempty(row.get("source_pair_id")):
            raise ValueError(f"{path.name}:{sample_id} has incomplete canonical join fields")
        if not nonempty(row.get("true_native_t1_path")) or not nonempty(row.get("true_native_t2_path")):
            raise ValueError(f"{path.name}:{sample_id} has incomplete native pair paths")
        by_id[sample_id] = row
    return by_id


def validate_blind_assets(
    review_package: Path,
    packet_a: dict[str, dict[str, Any]],
    packet_b: dict[str, dict[str, Any]],
    audit: dict[str, Any],
) -> str:
    """Verify the anonymous image asset manifest and every referenced file."""
    path = review_package / "blind_assets_manifest.jsonl"
    require_file(path)
    rows = read_jsonl(path)
    expected_count = audit.get("blind_asset_count")
    if expected_count is not None and int(expected_count) != len(rows):
        raise ValueError(f"blind asset manifest count mismatch: {len(rows)} != {expected_count}")
    seen: set[str] = set()
    for row in rows:
        relative = str(row.get("relative_path") or "")
        if not relative or relative in seen or not relative.startswith("blind_assets/"):
            raise ValueError("blind asset manifest has invalid or duplicate relative paths")
        seen.add(relative)
        asset = review_package / relative
        require_file(asset)
        if int(row.get("bytes", -1)) != asset.stat().st_size or row.get("sha256") != file_sha(asset):
            raise ValueError(f"blind asset integrity mismatch: {relative}")
    referenced: set[str] = set()
    for packet in (packet_a, packet_b):
        for row in packet.values():
            true_pair = row.get("true_pair") or {}
            referenced.update(str(true_pair.get(key)) for key in ("t1_path", "t2_path"))
            for neighbour in row.get("candidate_neighbours") or []:
                referenced.update(str(neighbour.get(key)) for key in ("t1_path", "t2_path"))
    if None in referenced or not referenced.issubset(seen):
        raise ValueError("blind packets reference assets absent from the manifest")
    return file_sha(path)


def load_reviewer_decisions(
    review_package: Path,
    reviewer: str,
    packet: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], str]:
    path = review_package / f"{reviewer}_decision_template.jsonl"
    require_file(path)
    rows = read_jsonl(path)
    if len(rows) != 1500:
        raise ValueError(f"{path.name} must contain 1500 rows, got {len(rows)}")
    by_id: dict[str, dict[str, Any]] = {}
    identities: set[str] = set()
    for row in rows:
        sample_id = str(row.get("review_sample_id") or "")
        if not sample_id or sample_id in by_id or sample_id not in packet:
            raise ValueError(f"{path.name} has missing, duplicate, or unknown sample ID {sample_id!r}")
        if str(row.get("reviewer_id") or "") != reviewer:
            raise ValueError(f"{path.name}:{sample_id} has the wrong reviewer_id")
        identity = identity_key(row.get("reviewer_identity"))
        if not identity:
            raise ValueError(f"{path.name}:{sample_id} has no human reviewer identity")
        identities.add(identity)
        if not parse_timestamp(row.get("reviewed_at")):
            raise ValueError(f"{path.name}:{sample_id} has no valid reviewed_at timestamp")
        if not attested(row.get("independence_attestation")):
            raise ValueError(f"{path.name}:{sample_id} lacks an independence attestation")
        for field in ("caption_accurate", "identifiable_against_neighbours", "nonexact_candidate_consistent"):
            value = str(row.get(field) or "").casefold()
            if value not in QUESTION_VALUES:
                raise ValueError(f"{path.name}:{sample_id} has invalid {field}: {value!r}")
        decision = str(row.get("final_decision") or "")
        if decision not in DECISIONS:
            raise ValueError(f"{path.name}:{sample_id} has invalid final_decision: {decision!r}")
        confidence = row.get("confidence")
        if confidence is not None:
            try:
                if not 0.0 <= float(confidence) <= 1.0:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}:{sample_id} has invalid confidence") from exc
        if decision == "REWRITE_REQUIRED" and not nonempty(row.get("rewritten_caption")):
            raise ValueError(f"{path.name}:{sample_id} marks REWRITE_REQUIRED without a rewrite")
        by_id[sample_id] = row
    if set(by_id) != set(packet):
        raise ValueError(f"{path.name} does not cover exactly the packet samples")
    if len(identities) != 1:
        raise ValueError(f"{path.name} must represent one human identity, got {len(identities)}")
    return by_id, next(iter(identities))


def load_adjudication(
    review_package: Path,
    packet_a: dict[str, dict[str, Any]],
    reviewer_a: dict[str, dict[str, Any]],
    reviewer_b: dict[str, dict[str, Any]],
    reviewer_identities: tuple[str, str],
) -> tuple[dict[str, dict[str, Any]], str]:
    path = review_package / "adjudication_template.jsonl"
    require_file(path)
    rows = read_jsonl(path)
    if len(rows) != 1500:
        raise ValueError(f"{path.name} must contain 1500 rows, got {len(rows)}")
    by_id: dict[str, dict[str, Any]] = {}
    adjudicator_identities: set[str] = set()
    for row in rows:
        sample_id = str(row.get("review_sample_id") or "")
        if not sample_id or sample_id in by_id or sample_id not in packet_a:
            raise ValueError(f"{path.name} has missing, duplicate, or unknown sample ID {sample_id!r}")
        decision_a = reviewer_a[sample_id].get("final_decision")
        decision_b = reviewer_b[sample_id].get("final_decision")
        if row.get("reviewer_a_decision") != decision_a or row.get("reviewer_b_decision") != decision_b:
            raise ValueError(f"{path.name}:{sample_id} does not faithfully record both reviewer decisions")
        final_decision = str(row.get("final_decision") or "")
        if final_decision not in DECISIONS:
            raise ValueError(f"{path.name}:{sample_id} has invalid final_decision: {final_decision!r}")
        identity = identity_key(row.get("adjudicator_identity"))
        if not identity:
            raise ValueError(f"{path.name}:{sample_id} has no adjudicator identity")
        adjudicator_identities.add(identity)
        if not parse_timestamp(row.get("adjudicated_at")):
            raise ValueError(f"{path.name}:{sample_id} has no valid adjudicated_at timestamp")
        if (
            final_decision == "EXACT"
            and (decision_a == "REWRITE_REQUIRED" or decision_b == "REWRITE_REQUIRED")
            and not nonempty(row.get("rewritten_caption"))
        ):
            raise ValueError(f"{path.name}:{sample_id} needs a verified rewrite before EXACT promotion")
        by_id[sample_id] = row
    if set(by_id) != set(packet_a):
        raise ValueError(f"{path.name} does not cover exactly the 1,500 review samples")
    if len(adjudicator_identities) != 1:
        raise ValueError(f"{path.name} must represent one adjudicator identity")
    adjudicator = next(iter(adjudicator_identities))
    if adjudicator in set(reviewer_identities):
        raise ValueError("adjudicator_identity must be distinct from both independent reviewers")
    return by_id, adjudicator


def source_dataset(row: dict[str, Any] | None) -> str:
    if not row:
        return "unknown"
    provenance = row.get("provenance") or {}
    caption_provenance = row.get("caption_provenance") or {}
    return str(
        provenance.get("source_dataset")
        or caption_provenance.get("source_dataset")
        or row.get("source_dataset")
        or "unknown"
    )


def norm_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def calibration_metrics(
    packet: dict[str, dict[str, Any]],
    ledger: dict[str, dict[str, Any]],
    decisions_a: dict[str, dict[str, Any]],
    decisions_b: dict[str, dict[str, Any]],
    adjudication: dict[str, dict[str, Any]],
    train_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_stratum: dict[str, Counter[str]] = {stratum: Counter() for stratum in STRATA}
    reviewer_counts = {
        "reviewer_a": {stratum: Counter() for stratum in STRATA},
        "reviewer_b": {stratum: Counter() for stratum in STRATA},
    }
    final_rows: list[dict[str, Any]] = []
    for sample_id, packet_row in packet.items():
        if sample_id not in ledger:
            raise ValueError(f"review packet sample is absent from internal join ledger: {sample_id}")
        stratum = str(packet_row["stratum"])
        final = str(adjudication[sample_id]["final_decision"])
        by_stratum[stratum][final] += 1
        reviewer_counts["reviewer_a"][stratum][str(decisions_a[sample_id]["final_decision"])] += 1
        reviewer_counts["reviewer_b"][stratum][str(decisions_b[sample_id]["final_decision"])] += 1
        final_rows.append({
            "sample_id": sample_id,
            "stratum": stratum,
            "final_decision": final,
            "source_dataset": str(
                ledger[sample_id].get("source_dataset")
                or source_dataset(train_rows.get(str(ledger[sample_id].get("query_id")) or {}) or {})
            ),
            "query_id": ledger[sample_id].get("query_id"),
        })
    exact = [row for row in final_rows if row["stratum"] == "exact_discriminative"]
    exact_n = len(exact)
    exact_success = sum(row["final_decision"] == "EXACT" for row in exact)
    weak = sum(row["final_decision"] in {"REWRITE_REQUIRED", "REJECT"} for row in exact)
    semantic_alternative = sum(row["final_decision"] in {"SEMANTIC_ONLY", "AMBIGUOUS_IGNORE"} for row in exact)
    source_conditioned: dict[str, Any] = {}
    for source in ("levir_mci", "second_cc"):
        source_rows = [row for row in exact if row["source_dataset"] == source]
        source_success = sum(row["final_decision"] == "EXACT" for row in source_rows)
        source_conditioned[source] = {
            "exact_identifiable": proportion(source_success, len(source_rows)),
            "weak_caption_rate": proportion(
                sum(row["final_decision"] in {"REWRITE_REQUIRED", "REJECT"} for row in source_rows),
                len(source_rows),
            ),
            "semantic_alternative_rate": proportion(
                sum(row["final_decision"] in {"SEMANTIC_ONLY", "AMBIGUOUS_IGNORE"} for row in source_rows),
                len(source_rows),
            ),
        }
    all_ids = sorted(packet)
    kappa_all = cohen_kappa(
        [str(decisions_a[sample_id]["final_decision"]) for sample_id in all_ids],
        [str(decisions_b[sample_id]["final_decision"]) for sample_id in all_ids],
    )
    exact_ids = sorted(sample_id for sample_id in all_ids if packet[sample_id]["stratum"] == "exact_discriminative")
    kappa_exact = cohen_kappa(
        [str(decisions_a[sample_id]["final_decision"]) for sample_id in exact_ids],
        [str(decisions_b[sample_id]["final_decision"]) for sample_id in exact_ids],
    )
    generic = [row for row in final_rows if row["stratum"] == "generic_no_change"]
    semantic = [row for row in final_rows if row["stratum"] == "semantic_multi_positive"]
    return {
        "exact_identifiable_fraction": proportion(exact_success, exact_n),
        "exact_scope_precision": exact_success / exact_n if exact_n else None,
        "exact_scope_precision_ci95": wilson(exact_success, exact_n),
        "weak_caption_fraction": proportion(weak, exact_n),
        "weak_caption_rate": weak / exact_n if exact_n else None,
        "semantic_alternative_fraction": proportion(semantic_alternative, exact_n),
        "semantic_alternative_rate": semantic_alternative / exact_n if exact_n else None,
        "false_exact_rate": (exact_n - exact_success) / exact_n if exact_n else None,
        "source_conditioned": source_conditioned,
        "decision_counts_by_stratum": {stratum: dict(sorted(counts.items())) for stratum, counts in by_stratum.items()},
        "reviewer_decision_counts_by_stratum": {
            reviewer: {stratum: dict(sorted(counts.items())) for stratum, counts in values.items()}
            for reviewer, values in reviewer_counts.items()
        },
        "generic_no_change_contamination": proportion(
            sum(row["final_decision"] == "EXACT" for row in generic), len(generic)
        ),
        "semantic_stratum_exact_rate": proportion(
            sum(row["final_decision"] == "EXACT" for row in semantic), len(semantic)
        ),
        "reviewer_agreement": {
            "cohen_kappa_all_1500": kappa_all,
            "cohen_kappa_exact_stratum": kappa_exact,
            "exact_stratum_observed_agreement": (
                sum(decisions_a[sample_id]["final_decision"] == decisions_b[sample_id]["final_decision"] for sample_id in exact_ids)
                / len(exact_ids)
                if exact_ids
                else None
            ),
        },
    }


def make_manifest_rows(
    train_rows: list[dict[str, Any]],
    packet_by_id: dict[str, dict[str, Any]],
    ledger: dict[str, dict[str, Any]],
    adjudication: dict[str, dict[str, Any]],
    review_package: Path,
    dataset_release_sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = {
        str(ledger[sample_id].get("query_id")): (sample_id, adjudication[sample_id])
        for sample_id in packet_by_id
        if packet_by_id[sample_id].get("stratum") == "exact_discriminative"
        and adjudication[sample_id].get("final_decision") == "EXACT"
    }
    manifest: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    packet_audit_sha = file_sha(review_package / "review_package_audit.json")
    for original in train_rows:
        original_query_id = str(original.get("query_id") or "")
        if original_query_id not in selected:
            continue
        sample_id, decision = selected[original_query_id]
        row = copy.deepcopy(original)
        rewrite = decision.get("rewritten_caption")
        rewrite_used = nonempty(rewrite)
        text = str(rewrite if rewrite_used else row.get("text") or "").strip()
        if not text:
            raise ValueError(f"EXACT adjudication has no usable text: {sample_id}")
        new_query_id = f"r20_calibrated:{original_query_id}"
        row["query_id"] = new_query_id
        row["canonical_query_id"] = new_query_id
        row["text"] = text
        row["query_scope"] = "exact"
        row["query_classification"] = "exact_discriminative"
        row["view_scope"] = "exact"
        row["view_status"] = "READY_EXACT_R20_CALIBRATED"
        row["training_enabled"] = True
        row["candidate_training_enabled"] = True
        row["training_gate"] = "EXACT_R20_HUMAN_CALIBRATED"
        row["release_lineage"] = {
            **(row.get("release_lineage") or {}),
            "release": "qcpr_r20_candidate_exact_supervision_20260812",
            "source_release": dataset_release_sha,
            "projection_only": False,
            "r20_calibrated_from_query_id": original_query_id,
        }
        provenance = dict(row.get("provenance") or {})
        provenance["release_lineage"] = "qcpr_r20_candidate_exact_supervision_20260812"
        provenance["normalized_text"] = norm_text(text)
        provenance["filter_policy"] = "r20_human_exact_calibration"
        provenance["identifiability_status"] = "human_adjudicated_exact"
        provenance.pop("identifiability_score", None)
        provenance["r20_calibration"] = {
            "review_sample_id": sample_id,
            "source_query_id": original_query_id,
            "adjudication_decision": "EXACT",
            "rewrite_used": rewrite_used,
            "review_package_audit_sha256": packet_audit_sha,
        }
        row["provenance"] = provenance
        caption_provenance = dict(row.get("caption_provenance") or {})
        caption_provenance.update(
            {
                "human_authored": True,
                "mask_free": True,
                "review_status": "r20_adjudicated_exact",
                "source_revision": "r20_human_rewrite" if rewrite_used else "r20_human_verified",
                "type": "human_rewrite" if rewrite_used else caption_provenance.get("type", "source_authored"),
            }
        )
        row["caption_provenance"] = caption_provenance
        row["r20_calibration"] = {
            "review_sample_id": sample_id,
            "source_query_id": original_query_id,
            "adjudication_decision": "EXACT",
            "rewrite_used": rewrite_used,
            "review_package_audit_sha256": packet_audit_sha,
        }
        manifest.append(row)
        audit_rows.append(
            {
                "r20_query_id": new_query_id,
                "source_query_id": original_query_id,
                "review_sample_id": sample_id,
                "source_pair_id": row.get("source_pair_id") or row.get("source_item_id"),
                "source_dataset": source_dataset(row),
                "final_decision": "EXACT",
                "rewrite_used": rewrite_used,
                "review_package_audit_sha256": packet_audit_sha,
            }
        )
    if len({str(row.get("query_id")) for row in manifest}) != len(manifest):
        raise ValueError("r20 promoted manifest contains duplicate query IDs")
    if len({str(row.get("source_pair_id") or row.get("source_item_id")) for row in manifest}) != len(manifest):
        raise ValueError("r20 exact calibration must contain at most one reviewed query per physical pair")
    return manifest, audit_rows


def report_markdown(report: dict[str, Any]) -> str:
    calibration = report["human_calibration"]
    metrics = calibration.get("metrics") or {}
    gate = report.get("exact_gate") or {}
    candidate = report.get("r20_candidate") or {}
    status = report.get("status")
    return f"""# QCPR r20 exact-supervision report

## Decision

`{status}`

`R19G_IMMUTABLE = {report.get('r19g_immutable')}`  
`R20_MATCHED_EXACT_TRAINING_READY = {report.get('R20_MATCHED_EXACT_TRAINING_READY')}`  
`DATASET_AGENT_RECOMMENDATION = {report.get('recommendation')}`

The exact gate is explicit and was supplied as `{gate.get('minimum_exact_precision')}`.
The gate statistic is the lower bound of the Wilson 95% interval. The point
estimate and interval are reported separately; no gate is inferred from agent
diagnostics.

## Human calibration

- completed decisions: `{calibration.get('completed_decisions')}/1500`
- exact-identifiable fraction: `{metrics.get('exact_scope_precision')}`
- exact-identifiable 95% CI: `{metrics.get('exact_scope_precision_ci95')}`
- weak-caption rate: `{metrics.get('weak_caption_rate')}`
- semantic-alternative rate: `{metrics.get('semantic_alternative_rate')}`
- source-conditioned values: `{json.dumps(metrics.get('source_conditioned'), ensure_ascii=False, sort_keys=True)}`
- reviewer agreement: `{json.dumps(metrics.get('reviewer_agreement'), ensure_ascii=False, sort_keys=True)}`

Decision counts by stratum are stored in the JSON report. Generic no-change
rows remain diagnostic-only and are never included in exact training.

## Exact gate

- status: `{gate.get('status')}`
- criterion: `{gate.get('criterion')}`
- passes: `{gate.get('passes')}`
- reason: `{gate.get('reason')}`

## r20 candidate

- query count: `{candidate.get('train_query_count')}`
- physical-pair count: `{candidate.get('train_pair_count')}`
- manifest SHA256: `{candidate.get('exact_train_manifest_sha256')}`
- unreviewed r19g rows extrapolated: `0`
- legacy r19g development/test: unchanged and not rewritten

No RSCC, S2Looking text, TAMMs, or Forest text was added in this cycle.
"""


def build_report(
    *,
    args: argparse.Namespace,
    audit: dict[str, Any],
    source_lineage: dict[str, Any],
    metrics: dict[str, Any],
    completed: int,
    identities: tuple[str, str, str],
    gate: dict[str, Any],
    manifest: list[dict[str, Any]] | None,
    manifest_sha: str | None,
    audit_sha: str | None,
    package_hashes: dict[str, str],
) -> dict[str, Any]:
    train_path = args.release_dir / "exact_core_train.jsonl"
    dev_path = args.release_dir / "exact_core_development.jsonl"
    test_path = args.release_dir / "exact_core_test.jsonl"
    ready = bool(gate.get("passes") and manifest is not None and manifest_sha)
    report: dict[str, Any] = {
        "schema_version": "qcpr-r20-exact-supervision-report-v2",
        "status": "READY_R20_EXACT_HUMAN_CALIBRATED" if ready else gate.get("status", "HOLD_HUMAN_REVIEW_REQUIRED"),
        "producer_agent": "DATASET_AGENT",
        "producer_git_sha": args.producer_git_sha,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset_release_name": "qcpr_bitemporal_v2_train_20260808_final_r19g",
        "dataset_release_sha": args.dataset_release_sha,
        "r19g_immutable": True,
        "r19g_legacy_exact_eval_unchanged": True,
        "r19g_manifest_sha256": {
            "exact_core_train.jsonl": file_sha(train_path),
            "exact_core_development.jsonl": file_sha(dev_path),
            "exact_core_test.jsonl": file_sha(test_path),
        },
        "human_calibration": {
            "required_decisions": 1500,
            "completed_decisions": completed,
            "adjudicated_rows": completed if completed == 1500 else 0,
            "review_package_status": audit.get("status"),
            "reviewer_identities": {"reviewer_a": identities[0], "reviewer_b": identities[1], "adjudicator": identities[2]},
            "metrics": metrics if completed == 1500 else None,
            "exact_review_source_counts": audit.get("exact_source_counts"),
            "exact_review_source_registry_sha256": audit.get("exact_source_registry_sha256"),
            "no_decisions_fabricated": True,
        },
        "exact_gate": gate,
        "r20_candidate": {
            "lineage_status": "READY_CALIBRATED_EXACT_SAMPLE" if ready else "CANDIDATE_HOLD",
            "exact_train_manifest_status": "READY" if ready else "NOT_MATERIALIZED_GATE_HOLD",
            "exact_train_manifest_sha256": manifest_sha,
            "train_query_count": len(manifest) if manifest is not None else None,
            "train_pair_count": len({str(row.get('source_pair_id') or row.get('source_item_id')) for row in manifest}) if manifest is not None else None,
            "calibrated_development_manifest_status": "NOT_MATERIALIZED_R20_EXACT_SAMPLE_IS_TRAIN_ONLY",
            "r20_matched_exact_training_ready": ready,
            "promoted_manifest": "r20_exact_train_calibrated.jsonl" if ready else None,
            "promoted_manifest_audit": "r20_exact_train_calibrated_audit.jsonl" if ready else None,
            "promoted_manifest_audit_sha256": audit_sha,
        },
        "review_package": {
            "path": str(args.review_package),
            "source_lineage_artifact_sha256": source_lineage.get("artifact_sha256"),
            "files_sha256_current": package_hashes,
        },
        "source_policy": {
            "new_sources_added": False,
            "rscc_added": False,
            "s2looking_text_added": False,
            "tamms_added": False,
            "forest_added_to_primary_training": False,
        },
        "recommendation": "RUN_ONE_DATA_ONLY_ABLATION" if ready else "DO_NOT_TRAIN_YET",
        "training_authorization": {
            "main_training_allowed": ready,
            "r20_matched_exact_training_ready": ready,
            "model_agent_action": "REQUEST_ONE_MATCHED_DATA_ONLY_ABLATION" if ready else "DO_NOT_REQUEST_TRAINING",
        },
        "artifact_hash_definition": "SHA256 of UTF-8 bytes of json.dumps(value excluding artifact_sha256, ensure_ascii=False, sort_keys=True, separators=(',', ':')) followed by one newline",
    }
    report["artifact_sha256"] = canonical_sha(report)
    return report


def build_pending_report(
    args: argparse.Namespace,
    audit: dict[str, Any],
    source_lineage: dict[str, Any],
    packet_a: dict[str, dict[str, Any]],
    packet_b: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Create a truthful HOLD report for the untouched review package."""
    del packet_a, packet_b
    package_hashes = {
        name: file_sha(args.review_package / name)
        for name in (
            "review_package_audit.json",
            "reviewer_a_packet.jsonl",
            "reviewer_b_packet.jsonl",
            "reviewer_a_decision_template.jsonl",
            "reviewer_b_decision_template.jsonl",
            "adjudication_template.jsonl",
            "blind_assets_manifest.jsonl",
        )
    }
    lineage_path = args.review_package.parent / "source_lineage.json"
    if not lineage_path.exists():
        lineage_path = args.review_package / "source_lineage.json"
    package_hashes["source_lineage.json"] = file_sha(lineage_path)
    ledger_path = args.review_package.parent / "internal_review_join_ledger.jsonl"
    if ledger_path.exists():
        package_hashes["internal_review_join_ledger.jsonl"] = file_sha(ledger_path)
    metrics = {
        "exact_scope_precision": None,
        "exact_scope_precision_ci95": None,
        "weak_caption_rate": None,
        "semantic_alternative_rate": None,
        "source_conditioned": None,
    }
    gate = {
        "status": "HOLD_HUMAN_REVIEW_REQUIRED",
        "passes": False,
        "minimum_exact_precision": args.min_exact_precision,
        "statistic": "wilson_ci95_lower_bound",
        "criterion": f"Wilson 95% CI lower bound >= {args.min_exact_precision:g}",
        "observed_exact_scope_precision": None,
        "observed_exact_scope_precision_ci95": None,
        "completed_decisions": 0,
        "required_decisions": 1500,
        "reason": "all reviewer decision and adjudication fields are untouched",
    }
    report = build_report(
        args=args,
        audit=audit,
        source_lineage=source_lineage,
        metrics=metrics,
        completed=0,
        identities=("PENDING_REVIEWER_A", "PENDING_REVIEWER_B", "PENDING_ADJUDICATOR"),
        gate=gate,
        manifest=None,
        manifest_sha=None,
        audit_sha=None,
        package_hashes=package_hashes,
    )
    report["R20_MATCHED_EXACT_TRAINING_READY"] = False
    report["main_training_allowed"] = False
    report["artifact_sha256"] = canonical_sha(report)
    return report


def build_handoff(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    candidate = report["r20_candidate"]
    calibration = report["human_calibration"]
    gate = report["exact_gate"]
    ready = bool(report.get("R20_MATCHED_EXACT_TRAINING_READY"))
    handoff = {
        "schema_version": "qcpr-dataset-final-to-model-r20-v2",
        "status": report["status"],
        "producer_agent": "DATASET_AGENT",
        "producer_git_sha": args.producer_git_sha,
        "created_at": report["created_at"],
        "dataset_release_name": "qcpr_r20_candidate_exact_supervision_20260812",
        "dataset_release_sha": report["dataset_release_sha"],
        "parent_release_name": "qcpr_bitemporal_v2_train_20260808_final_r19g",
        "parent_dataset_release_sha": report["dataset_release_sha"],
        "R19G_IMMUTABLE": True,
        "R20_MATCHED_EXACT_TRAINING_READY": ready,
        "main_training_allowed": ready,
        "exact_train_manifest": {
            "path": "r20_exact_train_calibrated.jsonl" if ready else None,
            "sha256": candidate.get("exact_train_manifest_sha256"),
            "query_count": candidate.get("train_query_count"),
            "pair_count": candidate.get("train_pair_count"),
        },
        "exact_train_manifest_audit": {
            "path": "r20_exact_train_calibrated_audit.jsonl" if ready else None,
            "sha256": candidate.get("promoted_manifest_audit_sha256"),
        },
        "calibrated_development_manifest": None,
        "human_calibration": calibration,
        "exact_gate": gate,
        "generic_no_change_training_enabled": False,
        "masks_in_evaluation_sidecars_only": True,
        "new_sources_added": False,
        "recommendation": report["recommendation"],
        "artifact_hash_definition": report["artifact_hash_definition"],
    }
    handoff["artifact_sha256"] = canonical_sha(handoff)
    return handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--review-package", type=Path, required=True)
    parser.add_argument("--dataset-release-sha", required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument(
        "--min-exact-precision",
        type=float,
        required=True,
        help="Explicit exact-scope gate threshold; the Wilson 95%% CI lower bound must meet it.",
    )
    parser.add_argument("--allow-hold", action="store_true", help="Return success for a truthful pending/failed gate.")
    args = parser.parse_args()
    if not 0.0 <= args.min_exact_precision <= 1.0:
        raise SystemExit("--min-exact-precision must be between 0 and 1")
    try:
        for name in (
            "review_package_audit.json",
            "reviewer_a_packet.jsonl",
            "reviewer_b_packet.jsonl",
            "reviewer_a_decision_template.jsonl",
            "reviewer_b_decision_template.jsonl",
            "adjudication_template.jsonl",
            "blind_assets_manifest.jsonl",
        ):
            require_file(args.review_package / name)
        audit = read_json(args.review_package / "review_package_audit.json")
        lineage_path = args.review_package.parent / "source_lineage.json"
        if not lineage_path.exists():
            lineage_path = args.review_package / "source_lineage.json"
        require_file(lineage_path)
        source_lineage = read_json(lineage_path)
        if audit.get("dataset_release_sha") != args.dataset_release_sha:
            raise ValueError("review package and requested dataset release SHA do not match")
        if audit.get("r19_immutable_required") is not True:
            raise ValueError("review package does not enforce r19g immutability")
        packet_a, packet_b = load_packets(args.review_package)
        blind_asset_manifest_sha256 = validate_blind_assets(args.review_package, packet_a, packet_b, audit)
        # An untouched review package is a valid, truthful HOLD state.  Do
        # not make the user fill 4,500 cells merely to ask for a status.  As
        # soon as any decision material exists, validate the entire contract
        # and fail closed on partial or malformed review data.
        decision_a_rows = read_jsonl(args.review_package / "reviewer_a_decision_template.jsonl")
        decision_b_rows = read_jsonl(args.review_package / "reviewer_b_decision_template.jsonl")
        adjudication_rows = read_jsonl(args.review_package / "adjudication_template.jsonl")
        populated = any(
            row.get(key) not in (None, "")
            for rows in (decision_a_rows, decision_b_rows, adjudication_rows)
            for row in rows
            for key in (
                "reviewer_identity",
                "reviewed_at",
                "independence_attestation",
                "caption_accurate",
                "identifiable_against_neighbours",
                "nonexact_candidate_consistent",
                "final_decision",
                "adjudicator_identity",
                "adjudicated_at",
            )
        )
        if not populated:
            report = build_pending_report(args, audit, source_lineage, packet_a, packet_b)
            candidate_report = args.candidate_dir / "qcpr_r20_exact_supervision_report.json"
            write_json(candidate_report, report)
            (args.candidate_dir / "qcpr_r20_exact_supervision_report.md").write_text(
                report_markdown(report), encoding="utf-8"
            )
            handoff = build_handoff(report, args)
            handoff_path = args.candidate_dir / "handoff" / "dataset_final_to_model_r20.json"
            write_json(handoff_path, handoff)
            status = {
                "schema_version": "qcpr-r20-exact-supervision-promotion-status-v1",
                "status": report["status"],
                "dataset_release_sha": args.dataset_release_sha,
                "producer_git_sha": args.producer_git_sha,
                "created_at": report["created_at"],
                "required_decisions": 1500,
                "completed_decisions": 0,
                "minimum_exact_precision": args.min_exact_precision,
                "exact_gate": report["exact_gate"],
                "R19G_IMMUTABLE": True,
                "R20_MATCHED_EXACT_TRAINING_READY": False,
                "exact_train_manifest_sha256": None,
                "report_artifact_sha256": report["artifact_sha256"],
                "handoff_artifact_sha256": handoff["artifact_sha256"],
            }
            status["artifact_sha256"] = canonical_sha(status)
            write_json(args.candidate_dir / "promotion_gate_status.json", status)
            print(json.dumps(status, ensure_ascii=False, sort_keys=True))
            return 0 if args.allow_hold else 2
        ledger = load_internal_ledger(args.review_package)
        if set(ledger) != set(packet_a):
            raise ValueError("internal join ledger and blind packets do not contain the same 1,500 sample IDs")
        for sample_id, packet_row in packet_a.items():
            aliases = ledger[sample_id].get("true_blind_t1_path"), ledger[sample_id].get("true_blind_t2_path")
            visible = packet_row.get("true_pair") or {}
            if aliases != (visible.get("t1_path"), visible.get("t2_path")):
                raise ValueError(f"blind packet and internal ledger disagree for {sample_id}")
        decisions_a, identity_a = load_reviewer_decisions(args.review_package, "reviewer_a", packet_a)
        decisions_b, identity_b = load_reviewer_decisions(args.review_package, "reviewer_b", packet_b)
        if identity_a == identity_b:
            raise ValueError("reviewer_a and reviewer_b must have distinct human identities")
        adjudication, adjudicator = load_adjudication(
            args.review_package,
            packet_a,
            decisions_a,
            decisions_b,
            (identity_a, identity_b),
        )
        train_rows = read_jsonl(args.release_dir / "exact_core_train.jsonl")
        train_by_query = {str(row.get("query_id")): row for row in train_rows}
        if len(train_by_query) != len(train_rows):
            raise ValueError("r19g exact train has duplicate/missing query IDs")
        for sample_id, packet_row in packet_a.items():
            if packet_row["stratum"] == "exact_discriminative":
                query_id = str(ledger[sample_id].get("query_id") or "")
                if query_id not in train_by_query:
                    raise ValueError(f"exact review sample is not in r19g train: {query_id}")
                if train_by_query[query_id].get("training_enabled") is not True:
                    raise ValueError(f"exact review sample is not training-enabled in r19g: {query_id}")
        metrics = calibration_metrics(packet_a, ledger, decisions_a, decisions_b, adjudication, train_by_query)
        exact_metric = metrics["exact_scope_precision_ci95"]
        ci_lower = exact_metric[0] if exact_metric else None
        completed = len(adjudication)
        gate_passes = bool(
            completed == 1500
            and ci_lower is not None
            and ci_lower >= args.min_exact_precision
        )
        gate = {
            "status": "PASS_EXACT_SCOPE_PRECISION_GATE" if gate_passes else "HOLD_EXACT_SCOPE_PRECISION_GATE",
            "passes": gate_passes,
            "minimum_exact_precision": args.min_exact_precision,
            "statistic": "wilson_ci95_lower_bound",
            "criterion": f"Wilson 95% CI lower bound >= {args.min_exact_precision:g}",
            "observed_exact_scope_precision": metrics.get("exact_scope_precision"),
            "observed_exact_scope_precision_ci95": metrics.get("exact_scope_precision_ci95"),
            "completed_decisions": completed,
            "required_decisions": 1500,
            "reason": (
                "human calibration is complete and the explicit gate passes"
                if gate_passes
                else (
                    f"human calibration decisions incomplete: {completed}/1500"
                    if completed != 1500
                    else "Wilson 95% CI lower bound does not meet the supplied threshold"
                )
            ),
        }
        package_hashes = {
            name: file_sha(args.review_package / name)
            for name in (
                "review_package_audit.json",
                "reviewer_a_packet.jsonl",
                "reviewer_b_packet.jsonl",
                "reviewer_a_decision_template.jsonl",
                "reviewer_b_decision_template.jsonl",
                "adjudication_template.jsonl",
                "blind_assets_manifest.jsonl",
            )
        }
        package_hashes["source_lineage.json"] = file_sha(lineage_path)
        package_hashes["internal_review_join_ledger.jsonl"] = file_sha(
            args.review_package.parent / "internal_review_join_ledger.jsonl"
        )
        package_hashes["blind_assets_manifest.jsonl"] = blind_asset_manifest_sha256
        manifest: list[dict[str, Any]] | None = None
        manifest_audit: list[dict[str, Any]] | None = None
        manifest_sha: str | None = None
        manifest_audit_sha: str | None = None
        if gate_passes:
            output_manifest = args.candidate_dir / "r20_exact_train_calibrated.jsonl"
            output_audit = args.candidate_dir / "r20_exact_train_calibrated_audit.jsonl"
            if output_manifest.exists() or output_audit.exists():
                raise ValueError("refusing to overwrite an existing promoted r20 manifest")
            manifest, manifest_audit = make_manifest_rows(
                train_rows,
                packet_a,
                ledger,
                adjudication,
                args.review_package,
                args.dataset_release_sha,
            )
            if not manifest:
                raise ValueError("gate passed but no adjudicated EXACT rows were available")
            write_jsonl(output_manifest, manifest)
            write_jsonl(output_audit, manifest_audit)
            manifest_sha = file_sha(output_manifest)
            manifest_audit_sha = file_sha(output_audit)
        report = build_report(
            args=args,
            audit=audit,
            source_lineage=source_lineage,
            metrics=metrics,
            completed=completed,
            identities=(identity_a, identity_b, adjudicator),
            gate=gate,
            manifest=manifest,
            manifest_sha=manifest_sha,
            audit_sha=manifest_audit_sha,
            package_hashes=package_hashes,
        )
        report["R20_MATCHED_EXACT_TRAINING_READY"] = bool(gate_passes and manifest)
        report["main_training_allowed"] = bool(gate_passes and manifest)
        report["artifact_sha256"] = canonical_sha(report)
        candidate_report = args.candidate_dir / "qcpr_r20_exact_supervision_report.json"
        candidate_report.parent.mkdir(parents=True, exist_ok=True)
        if completed == 1500:
            write_json(candidate_report, report)
            (args.candidate_dir / "qcpr_r20_exact_supervision_report.md").write_text(
                report_markdown(report), encoding="utf-8"
            )
        handoff = build_handoff(report, args)
        handoff_path = args.candidate_dir / "handoff" / "dataset_final_to_model_r20.json"
        if completed == 1500:
            write_json(handoff_path, handoff)
        status = {
            "schema_version": "qcpr-r20-exact-supervision-promotion-status-v1",
            "status": report["status"],
            "dataset_release_sha": args.dataset_release_sha,
            "producer_git_sha": args.producer_git_sha,
            "created_at": report["created_at"],
            "required_decisions": 1500,
            "completed_decisions": completed,
            "minimum_exact_precision": args.min_exact_precision,
            "exact_gate": gate,
            "R19G_IMMUTABLE": True,
            "R20_MATCHED_EXACT_TRAINING_READY": bool(gate_passes and manifest),
            "exact_train_manifest_sha256": manifest_sha,
            "report_artifact_sha256": report["artifact_sha256"],
            "handoff_artifact_sha256": handoff["artifact_sha256"],
        }
        status["artifact_sha256"] = canonical_sha(status)
        write_json(args.candidate_dir / "promotion_gate_status.json", status)
        print(json.dumps(status, ensure_ascii=False, sort_keys=True))
        if not gate_passes and not args.allow_hold:
            return 2
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"PROMOTION_GATE_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
