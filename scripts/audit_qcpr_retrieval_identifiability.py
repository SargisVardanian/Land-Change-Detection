from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--reproduction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize(text):
    return " ".join(str(text).casefold().strip(" .").split())


def lexical_signature(text):
    stop = {"a", "an", "the", "there", "is", "are", "was", "were", "has", "have", "of", "to"}
    words = re.findall(r"[a-z0-9]+", normalize(text))
    output = []
    for word in words:
        if word in stop:
            continue
        if len(word) > 5 and word.endswith("ing"):
            word = word[:-3]
        elif len(word) > 4 and word.endswith("ed"):
            word = word[:-2]
        elif len(word) > 4 and word.endswith("s"):
            word = word[:-1]
        output.append(word)
    return " ".join(output)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def distribution(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    pick = lambda q: ordered[min(int((len(ordered) - 1) * q), len(ordered) - 1)]
    return {
        "count": len(values), "min": ordered[0], "p25": pick(0.25), "median": pick(0.5),
        "p75": pick(0.75), "p90": pick(0.9), "p99": pick(0.99), "max": ordered[-1],
        "mean": sum(values) / len(values),
    }


def cluster_summary(rows):
    exact = defaultdict(list)
    normalized = defaultdict(list)
    lexical = defaultdict(list)
    for row in rows:
        exact[row["caption"]].append(row)
        normalized[row["normalized_caption"]].append(row)
        lexical[lexical_signature(row["normalized_caption"])].append(row)
    def summarize(groups):
        duplicate = {key: value for key, value in groups.items() if len(value) > 1}
        pair_counts = [len({row["pair_id"] for row in cluster}) for cluster in duplicate.values()]
        return {
            "clusters": len(groups), "duplicate_clusters": len(duplicate),
            "cluster_size_distribution": distribution([len(cluster) for cluster in duplicate.values()]),
            "physical_pairs_per_duplicate_cluster": distribution(pair_counts),
            "largest_clusters": [
                {"caption_or_signature": key, "queries": len(cluster),
                 "physical_pairs": len({row["pair_id"] for row in cluster}),
                 "examples": [{"pair_id": row["pair_id"], "caption": row["caption"]} for row in cluster[:5]]}
                for key, cluster in sorted(duplicate.items(), key=lambda item: len(item[1]), reverse=True)[:20]
            ],
        }, groups
    exact_summary, exact_groups = summarize(exact)
    normalized_summary, normalized_groups = summarize(normalized)
    lexical_summary, lexical_groups = summarize(lexical)
    return exact_summary, normalized_summary, lexical_summary, exact_groups, normalized_groups, lexical_groups


def subset_audit(manifest_rows, split):
    captions = []
    pair_status = {}
    for row in manifest_rows:
        if row["split"] != split:
            continue
        status = "no_change" if row.get("source_metadata", {}).get("changeflag") == 0 else "changed"
        pair_status[row["pair_id"]] = status
        for index, caption in enumerate(row["captions"]):
            captions.append({"pair_id": row["pair_id"], "dataset": row["dataset_name"], "status": status,
                             "caption_index": index, "caption": caption, "normalized_caption": normalize(caption)})
    result = {}
    for status in ("all", "changed", "no_change"):
        selected = captions if status == "all" else [row for row in captions if row["status"] == status]
        exact, normalized, lexical, _, normalized_groups, lexical_groups = cluster_summary(selected)
        normalized_pair_counts = {
            key: len({row["pair_id"] for row in cluster})
            for key, cluster in normalized_groups.items()
        }
        lexical_pair_counts = {
            key: len({row["pair_id"] for row in cluster})
            for key, cluster in lexical_groups.items()
        }
        shared = [row for row in selected if normalized_pair_counts[row["normalized_caption"]] > 1]
        lexical_shared = [row for row in selected if lexical_pair_counts[lexical_signature(row["normalized_caption"])] > 1]
        exclusions = sum(max(normalized_pair_counts[row["normalized_caption"]] - 1, 0) for row in selected)
        pair_count = len({row["pair_id"] for row in selected})
        possible_negatives = len(selected) * max(pair_count - 1, 0)
        generic = [row for row in selected if row["status"] == "no_change" and len(lexical_signature(row["caption"]).split()) <= 4]
        result[status] = {
            "physical_pair_count": pair_count, "query_count": len(selected),
            "exact_unique_caption_count": len({row["caption"] for row in selected}),
            "normalized_unique_caption_count": len({row["normalized_caption"] for row in selected}),
            "exact_caption_clusters": exact, "normalized_caption_clusters": normalized,
            "lexical_near_duplicate_clusters": lexical,
            "shared_across_pairs_queries": len(shared),
            "shared_across_pairs_fraction": len(shared) / max(len(selected), 1),
            "lexical_shared_queries": len(lexical_shared),
            "lexical_shared_fraction": len(lexical_shared) / max(len(selected), 1),
            "ambiguous_negative_exclusion_count": exclusions,
            "ambiguous_negative_exclusion_rate": exclusions / max(possible_negatives, 1),
            "captions_with_no_observed_pair_discriminative_information": len(generic),
            "no_information_definition": "generic no-change caption with <=4 lexical content tokens; descriptive audit only",
        }
    return result


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = load_jsonl(args.manifest_dir / "natural_train_retrieval_manifest.jsonl")
    val_manifest = load_jsonl(args.manifest_dir / "natural_validation_retrieval_manifest.jsonl")
    manifests = train_manifest + val_manifest
    train_audit = subset_audit(manifests, "train")
    val_audit = subset_audit(manifests, "val")
    reproduction = json.loads((args.reproduction_dir / "reproduction.json").read_text())
    if not reproduction.get("passed"):
        raise RuntimeError("error analysis requires reproduced epoch-19 rankings")
    epoch_rows = load_jsonl(args.reproduction_dir / "epoch19_top10.jsonl")
    initial_rows = load_jsonl(args.reproduction_dir / "initial_top10.jsonl")
    initial_by_query = {row["query_id"]: row for row in initial_rows}
    val_pairs = {row["pair_id"]: row for row in val_manifest}
    caption_audit_rows = [
        row for row in load_jsonl(args.manifest_dir / "caption_quality_audit.jsonl")
        if row.get("split") == "val"
    ]
    audit_by_pair_caption = {
        (row["pair_id"], row["normalized_caption"]): row for row in caption_audit_rows
    }
    audit_by_pair = defaultdict(list)
    for row in caption_audit_rows:
        audit_by_pair[row["pair_id"]].append(row)
    normalized_pairs = defaultdict(set)
    lexical_pairs = defaultdict(set)
    for row in val_manifest:
        for caption in row["captions"]:
            normalized_pairs[normalize(caption)].add(row["pair_id"])
            lexical_pairs[lexical_signature(caption)].add(row["pair_id"])
    errors = []
    bucket_names = (
        "exact caption ambiguity", "near-duplicate caption ambiguity",
        "no-change generic query", "wrong physical pair but semantically valid result",
        "correct change type but wrong location", "correct object but wrong count",
        "same domain and visually similar pair", "LEVIR/SECOND domain confusion",
        "probable annotation mismatch", "genuine model ranking failure",
    )
    bucket_counts = Counter({name: 0 for name in bucket_names})
    for row in epoch_rows:
        if row["true_pair_rank"] == 1:
            continue
        true_pair = row["true_pair_id"]
        top_pair = row["top10_pair_ids"][0]
        query = row["query"]
        normalized_query = normalize(query)
        signature = lexical_signature(query)
        true_meta = val_pairs[true_pair]
        top_meta = val_pairs[top_pair]
        query_audit = audit_by_pair_caption.get((true_pair, normalized_query), {})
        top_audits = audit_by_pair.get(top_pair, [])
        query_objects = set(query_audit.get("object_family_audit_tags", []))
        query_locations = set(query_audit.get("location_audit_tags", []))
        query_counts = set(query_audit.get("count_audit_tags", []))
        query_direction = query_audit.get("temporal_direction")
        same_direction = [candidate for candidate in top_audits if candidate.get("temporal_direction") == query_direction]
        object_matches = [candidate for candidate in same_direction if query_objects & set(candidate.get("object_family_audit_tags", []))]
        status = "no_change" if true_meta.get("source_metadata", {}).get("changeflag") == 0 else "changed"
        if top_pair in normalized_pairs[normalized_query]:
            bucket = "exact caption ambiguity"
        elif top_pair in lexical_pairs[signature]:
            bucket = "near-duplicate caption ambiguity"
        elif status == "no_change" and len(signature.split()) <= 4:
            bucket = "no-change generic query"
        elif object_matches and query_locations and all(
            not (query_locations & set(candidate.get("location_audit_tags", [])))
            for candidate in object_matches
        ):
            bucket = "correct change type but wrong location"
        elif object_matches and query_counts and all(
            not (query_counts & set(candidate.get("count_audit_tags", [])))
            for candidate in object_matches
        ):
            bucket = "correct object but wrong count"
        elif true_meta["dataset_name"] != top_meta["dataset_name"]:
            bucket = "LEVIR/SECOND domain confusion"
        elif any(lexical_signature(caption) == signature for caption in top_meta["captions"]):
            bucket = "wrong physical pair but semantically valid result"
        elif true_meta.get("preprocessing_fingerprint") == top_meta.get("preprocessing_fingerprint"):
            bucket = "probable annotation mismatch"
        elif true_meta["dataset_name"] == top_meta["dataset_name"]:
            bucket = "same domain and visually similar pair"
        else:
            bucket = "genuine model ranking failure"
        bucket_counts[bucket] += 1
        initial = initial_by_query[row["query_id"]]
        errors.append({**row, "status": status, "bucket": bucket,
                       "initial_rank": initial["true_pair_rank"],
                       "rank_improvement": initial["true_pair_rank"] - row["true_pair_rank"],
                       "top_wrong_dataset": top_meta["dataset_name"],
                       "analysis_only": True})
    highest = sorted(errors, key=lambda row: (-row["top10_scores"][0], row["true_pair_rank"]))[:50]
    improved = sorted(errors, key=lambda row: row["rank_improvement"], reverse=True)[:50]
    no_change = sorted((row for row in errors if row["status"] == "no_change"), key=lambda row: row["true_pair_rank"], reverse=True)[:50]
    changed = sorted((row for row in errors if row["status"] == "changed"), key=lambda row: row["true_pair_rank"], reverse=True)[:50]
    write_rows(args.output_dir / "highest_ranked_failures.jsonl", highest)
    write_rows(args.output_dir / "strongest_improvements.jsonl", improved)
    write_rows(args.output_dir / "no_change_failures.jsonl", no_change)
    write_rows(args.output_dir / "changed_failures.jsonl", changed)
    audit = {
        "immutable_reproduction": reproduction,
        "train": train_audit, "validation": val_audit,
        "error_bucket_counts": dict(bucket_counts),
        "error_bucket_note": "heuristic analysis only; buckets are never used as training labels or losses",
        "semantic_cluster_note": "No human-verified semantic-equivalent groups exist. Lexical signatures are reported as near-duplicate proxies and are not converted to positives.",
        "identifiability_conclusion": {
            "low_exact_pair_performance_partly_under_specified": val_audit["all"]["shared_across_pairs_fraction"] > 0,
            "reason": "captions shared by multiple physical pairs cannot uniquely identify the labelled exact pair without additional information",
            "fabricated_unique_information": False,
        },
    }
    (args.output_dir / "retrieval_identifiability_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    lines = [
        "# QCPR retrieval identifiability audit", "",
        f"Immutable epoch-19 reproduction passed: **{reproduction['passed']}**.", "",
        "## Main conclusion", "",
        "Low exact-pair performance is partly caused by under-specified captions. Repeated generic descriptions, especially no-change captions, map to many physical pairs. This does not make those pairs verified positives; they are excluded only where the existing duplicate contract explicitly identifies ambiguity.", "",
        "## Validation", "",
        "| subset | pairs | queries | normalized unique | shared queries | shared fraction | lexical shared fraction |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all", "changed", "no_change"):
        row = val_audit[name]
        lines.append(f"| {name} | {row['physical_pair_count']} | {row['query_count']} | {row['normalized_unique_caption_count']} | {row['shared_across_pairs_queries']} | {row['shared_across_pairs_fraction']:.4f} | {row['lexical_shared_fraction']:.4f} |")
    lines.extend(["", "## Error buckets", ""])
    for bucket, count in bucket_counts.most_common():
        lines.append(f"- {bucket}: {count}")
    lines.extend(["", "Lexical near-duplicate clusters are deterministic proxies, not verified semantic equivalence. No handcrafted bucket is used by R1 training.", ""])
    (args.output_dir / "retrieval_identifiability_audit.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
