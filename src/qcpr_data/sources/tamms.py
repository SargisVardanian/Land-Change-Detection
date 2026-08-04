"""TAMMs temporal sequence adapter."""

from .common import iter_jsonl, normalize_sequence_row


def iter_items(path, *, split_resolver, source_revision="pilot", training_enabled=False, quality_status="PHYSICAL_ONLY"):
    for row in iter_jsonl(path):
        yield normalize_sequence_row(
            row,
            source="TAMMs",
            source_revision=source_revision,
            split=split_resolver(row),
            training_enabled=training_enabled,
            quality_status=quality_status,
            item_id_value=str(row.get("sequence_id")),
            provenance={"adapter": "TAMMs", "source_split": row.get("split"), "text_status": row.get("verification_status")},
        )
