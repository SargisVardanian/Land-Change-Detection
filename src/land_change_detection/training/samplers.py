from __future__ import annotations

from dataclasses import dataclass

from .retrieval_datasets import ManifestRecord, ManifestRetrievalDataset


@dataclass(frozen=True)
class TripletSample:
    anchor: ManifestRecord
    positive: ManifestRecord
    negative: ManifestRecord


def build_triplets(dataset: ManifestRetrievalDataset) -> list[TripletSample]:
    triplets: list[TripletSample] = []
    fallback_negative = dataset.records[-1] if dataset.records else None
    for record in dataset.records:
        if not record.positives:
            continue
        positive = dataset.by_id.get(record.positives[0])
        if positive is None:
            continue
        negative_id = record.negatives[0] if record.negatives else None
        negative = dataset.by_id.get(negative_id) if negative_id is not None else fallback_negative
        if negative is None or negative.item_id == record.item_id:
            continue
        triplets.append(TripletSample(anchor=record, positive=positive, negative=negative))
    return triplets
