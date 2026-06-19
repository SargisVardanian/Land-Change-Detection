from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PairRetrievalBaselinePreset:
    name: str
    description: str
    visual_backbone: str
    text_backbone: str
    pair_feature_mode: str
    supported: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BASELINE_PRESETS: tuple[PairRetrievalBaselinePreset, ...] = (
    PairRetrievalBaselinePreset(
        name="simple_patch_smoke",
        description="Offline smoke baseline with simple patch vision and simple text features.",
        visual_backbone="simple_patch",
        text_backbone="simple_text",
        pair_feature_mode="change_fusion",
        notes=("unit tests", "sanity checks", "no external weights required"),
    ),
    PairRetrievalBaselinePreset(
        name="dinov2_t2_only",
        description="DINOv2 baseline that uses only T2 visual tokens.",
        visual_backbone="dinov2",
        text_backbone="remoteclip",
        pair_feature_mode="t2_only",
        notes=("requires DINOv2 weights", "requires RemoteCLIP text weights"),
    ),
    PairRetrievalBaselinePreset(
        name="dinov2_signed_delta",
        description="DINOv2 baseline with signed delta and absolute delta features.",
        visual_backbone="dinov2",
        text_backbone="remoteclip",
        pair_feature_mode="signed_delta",
        notes=("requires DINOv2 weights", "requires RemoteCLIP text weights"),
    ),
    PairRetrievalBaselinePreset(
        name="dinov2_change_fusion",
        description="DINOv2 baseline with full change-fusion features.",
        visual_backbone="dinov2",
        text_backbone="remoteclip",
        pair_feature_mode="change_fusion",
        notes=("requires DINOv2 weights", "requires RemoteCLIP text weights"),
    ),
    PairRetrievalBaselinePreset(
        name="remoteclip_pair_fusion",
        description="RemoteCLIP image+text pair-fusion baseline.",
        visual_backbone="remoteclip",
        text_backbone="remoteclip",
        pair_feature_mode="change_fusion",
        supported=False,
        notes=("preset reserved", "RemoteCLIP vision encoder path not implemented yet"),
    ),
    PairRetrievalBaselinePreset(
        name="rsicrc_reference",
        description="RSICRC-style reference baseline.",
        visual_backbone="remoteclip",
        text_backbone="remoteclip",
        pair_feature_mode="change_fusion",
        supported=False,
        notes=("preset reserved", "RSICRC architecture reproduction not implemented yet"),
    ),
    PairRetrievalBaselinePreset(
        name="transition_histogram_oracle",
        description="Oracle baseline for transition-histogram similarity after semantic datasets are ready.",
        visual_backbone="oracle",
        text_backbone="simple_text",
        pair_feature_mode="oracle",
        supported=False,
        notes=("preset reserved", "semantic histogram oracle not implemented yet"),
    ),
)


def preset_names() -> tuple[str, ...]:
    return tuple(preset.name for preset in BASELINE_PRESETS)


def preset_by_name(name: str) -> PairRetrievalBaselinePreset:
    for preset in BASELINE_PRESETS:
        if preset.name == name:
            return preset
    raise KeyError(name)
