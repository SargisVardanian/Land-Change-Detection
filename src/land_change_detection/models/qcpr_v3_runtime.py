from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from land_change_detection.models.qcpr_v3 import QCPRV3Config, QCPRV3GenericGrounding
from land_change_detection.models.qcpr_v3_factory import QCPRV3BackboneConfig, build_clean_v3_model
from land_change_detection.models.qcpr_v3_teacher import FrozenV1Teacher
from land_change_detection.models.unichange_v3_retrieval import UniChangeV3RetrievalModel


IMMUTABLE_V1 = Path("/mnt/weka/svardanyan/rs_change_project/runs/qcpr_e0_20260711-015629/pilot/best_retrieval.pt")


@dataclass(frozen=True)
class V3InitializationAudit:
    initialization_mode: str
    checkpoint: str | None
    checkpoint_sha256: str | None
    checkpoint_stage: str
    strict_teacher_load: bool
    copied_student_modules: tuple[str, ...]
    excluded_v2_modules: tuple[str, ...]


def build_clean_v3(
    config: QCPRV3BackboneConfig,
    *,
    device: torch.device,
    enable_region_slots: bool = False,
) -> tuple[UniChangeV3RetrievalModel, None, V3InitializationAudit]:
    grounding_config = (
        QCPRV3Config(
            visual_source_dim=768,
            text_dim=768,
            global_text_dim=512,
            enable_region_slots=enable_region_slots,
        )
        if config.grounding_backbone_kind == "siglip2"
        else QCPRV3Config(
            visual_source_dim=768,
            enable_region_slots=enable_region_slots,
        )
    )
    student = build_clean_v3_model(
        config,
        device=device,
        grounding_config=grounding_config,
    )
    audit = V3InitializationAudit(
        initialization_mode="clean_pretrained",
        checkpoint=None,
        checkpoint_sha256=None,
        checkpoint_stage="pretrained_backbones_plus_random_v3_heads",
        strict_teacher_load=False,
        copied_student_modules=(),
        excluded_v2_modules=(
            "unichange_v2_model_instance",
            "patch_reranker",
            "object_scores",
            "direction_scores",
            "location_scores",
            "count_scores",
            "relation_scores",
        ),
    )
    return student, None, audit


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_config(checkpoint: str | Path = IMMUTABLE_V1) -> SimpleNamespace:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return SimpleNamespace(**payload["config"])


def build_v3_and_teacher(
    checkpoint: str | Path = IMMUTABLE_V1,
    *,
    device: torch.device,
    build_legacy_model,
    enable_region_slots: bool = False,
) -> tuple[UniChangeV3RetrievalModel, FrozenV1Teacher, V3InitializationAudit]:
    """Strict teacher plus v1-compatible student initialized only from immutable v1."""
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = SimpleNamespace(**payload["config"])
    teacher_legacy = build_legacy_model(config, device)
    teacher = FrozenV1Teacher(teacher_legacy, checkpoint).to(device)

    # Build a second, independent v1 instance and strictly load before extracting
    # only semantically compatible global modules. The old patch reranker is not
    # attached to v3 and cannot enter its optimizer or scoring path.
    student_legacy = build_legacy_model(config, device)
    student_legacy.load_state_dict(payload["model"], strict=True)
    grounder = QCPRV3GenericGrounding(QCPRV3Config(
        visual_source_dim=768,
        enable_region_slots=enable_region_slots,
    )).to(device)
    student = UniChangeV3RetrievalModel(
        student_legacy.visual_encoder,
        student_legacy.temporal_encoder,
        student_legacy.text_encoder,
        student_legacy.retrieval_head,
        grounder,
        text_adapter=student_legacy.text_adapter,
    ).to(device)
    audit = V3InitializationAudit(
        initialization_mode="historical_e0",
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        checkpoint_stage=str(payload.get("stage", payload.get("stage1_next", "unknown"))),
        strict_teacher_load=True,
        copied_student_modules=("visual_encoder", "temporal_encoder", "text_encoder", "retrieval_head", "text_adapter"),
        excluded_v2_modules=("patch_reranker", "object_scores", "direction_scores", "location_scores", "count_scores", "relation_scores"),
    )
    return student, teacher, audit


def assert_teacher_not_in_optimizer(teacher: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    teacher_ids = {id(parameter) for parameter in teacher.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    overlap = teacher_ids & optimizer_ids
    if overlap:
        raise RuntimeError(f"immutable v1 teacher leaked into optimizer ({len(overlap)} parameters)")
