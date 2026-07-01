from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DatasetRole(StrEnum):
    JOINT_TEXT_MASK = "joint_text_mask"
    CHANGE_CAPTION = "change_caption"
    TEMPORAL_SEGMENTATION = "temporal_segmentation"
    TEMPORAL_SSL = "temporal_ssl"
    EO_IMAGE_TEXT = "eo_image_text"


@dataclass(frozen=True)
class TemporalDatasetSpec:
    name: str
    role: DatasetRole
    priority: int
    sample_count: int | None
    temporal_depth: str
    image_size: str
    has_text: bool
    has_binary_masks: bool
    has_semantic_masks: bool
    source: str
    notes: str


DATASETS: tuple[TemporalDatasetSpec, ...] = (
    TemporalDatasetSpec(
        name="LEVIR-MCI",
        role=DatasetRole.JOINT_TEXT_MASK,
        priority=1,
        sample_count=10_077,
        temporal_depth="2",
        image_size="256x256",
        has_text=True,
        has_binary_masks=True,
        has_semantic_masks=False,
        source="https://github.com/Chen-Yang-Liu/Change-Agent",
        notes="Five captions per pair; road/building change masks; primary joint retrieval-caption-mask dataset.",
    ),
    TemporalDatasetSpec(
        name="SECOND-CC",
        role=DatasetRole.JOINT_TEXT_MASK,
        priority=1,
        sample_count=6_041,
        temporal_depth="2",
        image_size="256x256",
        has_text=True,
        has_binary_masks=True,
        has_semantic_masks=True,
        source="https://github.com/ChangeCapsInRS/SecondCC",
        notes="Five captions per pair; semantic maps and 30 land-cover transitions; critical for text-specific grounding.",
    ),
    TemporalDatasetSpec(
        name="RSCC",
        role=DatasetRole.CHANGE_CAPTION,
        priority=1,
        sample_count=62_315,
        temporal_depth="2",
        image_size="varied",
        has_text=True,
        has_binary_masks=False,
        has_semantic_masks=False,
        source="https://github.com/Bili-Sakura/RSCC",
        notes="Large disaster-oriented pre/post caption corpus; verify released splits and license before ingestion.",
    ),
    TemporalDatasetSpec(
        name="ChangeChat-87k",
        role=DatasetRole.CHANGE_CAPTION,
        priority=2,
        sample_count=87_000,
        temporal_depth="2",
        image_size="varied",
        has_text=True,
        has_binary_masks=False,
        has_semantic_masks=False,
        source="https://github.com/hanlinwu/ChangeChat",
        notes="Instruction-tuning data generated with rule-based and GPT-assisted procedures; keep provenance flags.",
    ),
    TemporalDatasetSpec(
        name="CC-Foundation",
        role=DatasetRole.CHANGE_CAPTION,
        priority=2,
        sample_count=200_000,
        temporal_depth="2",
        image_size="varied",
        has_text=True,
        has_binary_masks=False,
        has_semantic_masks=False,
        source="https://github.com/Meize0729/CCExpert",
        notes="Paper reports 1.2M captions; only ingest files that are actually public and document their provenance.",
    ),
    TemporalDatasetSpec(
        name="DynamicEarthNet",
        role=DatasetRole.TEMPORAL_SEGMENTATION,
        priority=1,
        sample_count=75,
        temporal_depth="daily sequence over two years",
        image_size="1024x1024 AOI observations",
        has_text=False,
        has_binary_masks=True,
        has_semantic_masks=True,
        source="https://mediatum.ub.tum.de/1650201",
        notes="Monthly pixel labels for seven land-cover classes; suitable for long-sequence temporal adaptation.",
    ),
    TemporalDatasetSpec(
        name="SpaceNet 7",
        role=DatasetRole.TEMPORAL_SEGMENTATION,
        priority=1,
        sample_count=101,
        temporal_depth="24 monthly observations",
        image_size="AOI mosaics",
        has_text=False,
        has_binary_masks=True,
        has_semantic_masks=False,
        source="https://spacenet.ai/sn7-challenge/",
        notes="Tracked building polygons and construction/demolition events across more than 100 geographies.",
    ),
    TemporalDatasetSpec(
        name="Hi-UCD",
        role=DatasetRole.TEMPORAL_SEGMENTATION,
        priority=1,
        sample_count=None,
        temporal_depth="3",
        image_size="0.1 m aerial imagery",
        has_text=False,
        has_binary_masks=True,
        has_semantic_masks=True,
        source="https://github.com/Daisy-7/Hi-UCD-S",
        notes="Three temporal phases and nine land-cover classes for detailed urban semantic change.",
    ),
    TemporalDatasetSpec(
        name="S2Looking",
        role=DatasetRole.TEMPORAL_SEGMENTATION,
        priority=2,
        sample_count=5_000,
        temporal_depth="2",
        image_size="1024x1024",
        has_text=False,
        has_binary_masks=True,
        has_semantic_masks=False,
        source="https://github.com/S2Looking/S2Looking",
        notes="Side-looking global building-change benchmark with strong viewpoint and illumination variation.",
    ),
    TemporalDatasetSpec(
        name="xBD",
        role=DatasetRole.TEMPORAL_SEGMENTATION,
        priority=2,
        sample_count=None,
        temporal_depth="2",
        image_size="high-resolution disaster tiles",
        has_text=False,
        has_binary_masks=True,
        has_semantic_masks=True,
        source="https://xview2.org/dataset",
        notes="Pre/post imagery with building polygons and ordinal damage classes; useful for disaster transfer.",
    ),
    TemporalDatasetSpec(
        name="TERRA-CD",
        role=DatasetRole.TEMPORAL_SEGMENTATION,
        priority=2,
        sample_count=5_221,
        temporal_depth="2",
        image_size="Sentinel-2 city patches",
        has_text=False,
        has_binary_masks=True,
        has_semantic_masks=True,
        source="https://github.com/omkarsoak/TERRA-CD",
        notes="2019/2024 pairs with land-cover, vegetation-change and semantic-transition masks.",
    ),
    TemporalDatasetSpec(
        name="SSL4EO-S12",
        role=DatasetRole.TEMPORAL_SSL,
        priority=1,
        sample_count=251_079,
        temporal_depth="4 seasons x S1/S2 modalities",
        image_size="264x264 per sensor patch",
        has_text=False,
        has_binary_masks=False,
        has_semantic_masks=False,
        source="https://github.com/zhu-xlab/SSL4EO-S12",
        notes="About 3M images; best first source for DINO/JEPA-style temporal adaptation.",
    ),
    TemporalDatasetSpec(
        name="RS5M",
        role=DatasetRole.EO_IMAGE_TEXT,
        priority=1,
        sample_count=5_000_000,
        temporal_depth="1",
        image_size="varied",
        has_text=True,
        has_binary_masks=False,
        has_semantic_masks=False,
        source="https://github.com/om-ai-lab/RS5M",
        notes="General EO image-text alignment; not a substitute for temporal change supervision.",
    ),
    TemporalDatasetSpec(
        name="SkyScript",
        role=DatasetRole.EO_IMAGE_TEXT,
        priority=2,
        sample_count=2_600_000,
        temporal_depth="1",
        image_size="varied",
        has_text=True,
        has_binary_masks=False,
        has_semantic_masks=False,
        source="https://github.com/wangzhecheng/SkyScript",
        notes="OSM-linked remote-sensing image-text pairs for broad semantic pretraining.",
    ),
    TemporalDatasetSpec(
        name="BigEarthNet.txt",
        role=DatasetRole.EO_IMAGE_TEXT,
        priority=2,
        sample_count=464_044,
        temporal_depth="1 co-registered S1/S2 observation",
        image_size="Sentinel patches",
        has_text=True,
        has_binary_masks=False,
        has_semantic_masks=False,
        source="https://arxiv.org/abs/2603.29630",
        notes="Reported 9.6M captions, VQA items and referring expressions; verify public release before use.",
    ),
)


def datasets_for_role(role: DatasetRole, *, max_priority: int | None = None) -> tuple[TemporalDatasetSpec, ...]:
    selected = [item for item in DATASETS if item.role == role]
    if max_priority is not None:
        selected = [item for item in selected if item.priority <= max_priority]
    return tuple(sorted(selected, key=lambda item: (item.priority, item.name.lower())))


def joint_text_mask_datasets() -> tuple[TemporalDatasetSpec, ...]:
    return datasets_for_role(DatasetRole.JOINT_TEXT_MASK)
