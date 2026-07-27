from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    stage: str
    rel_path: str
    role: str
    default_bootstrap: bool = False
    notes: tuple[str, ...] = ()

    def path_for(self, project_root: Path) -> Path:
        return project_root / "datasets" / "raw" / Path(self.rel_path)

    def to_dict(self, project_root: Path | None = None) -> dict[str, object]:
        payload = asdict(self)
        if project_root is not None:
            payload["path"] = str(self.path_for(project_root))
        return payload


DATASET_CURRICULUM: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="LEVIR-CC",
        stage="stage_1_text_to_pair",
        rel_path="LEVIR-CC",
        role="clean first-stage text-to-pair retrieval benchmark",
        default_bootstrap=False,
        notes=("caption-only pretraining before grounded retrieval",),
    ),
    DatasetSpec(
        name="LEVIR-MCI",
        stage="stage_2_grounded",
        rel_path="LEVIR-MCI-unpacked/LEVIR-MCI-dataset",
        role="grounded retrieval with binary change masks and first real experiment path",
        default_bootstrap=True,
        notes=("validate parser", "render sample grid", "run overfit-100"),
    ),
    DatasetSpec(
        name="SECOND-CC",
        stage="stage_3_transition_aware",
        rel_path="SECOND-CC",
        role="transition-aware and direction-aware pair retrieval",
        default_bootstrap=False,
        notes=("semantic before/after labels", "pair-to-pair retrieval bridge"),
    ),
    DatasetSpec(
        name="Hi-UCD",
        stage="stage_3_transition_aware",
        rel_path="Hi-UCD",
        role="multi-phase semantic retrieval follow-up after SECOND-CC",
        default_bootstrap=False,
        notes=("urban change across multiple phases",),
    ),
    DatasetSpec(
        name="TERRA-CD",
        stage="stage_3_transition_aware",
        rel_path="TERRA-CD",
        role="semantic transition supervision candidate",
    ),
    DatasetSpec(
        name="ChangeNet",
        stage="stage_3_transition_aware",
        rel_path="ChangeNet",
        role="pair-to-pair retrieval supervision candidate",
    ),
    DatasetSpec(
        name="HRSCD",
        stage="stage_3_transition_aware",
        rel_path="HRSCD",
        role="semantic change follow-up benchmark",
    ),
    DatasetSpec(
        name="HZNU-FCD",
        stage="stage_3_transition_aware",
        rel_path="HZNU-FCD",
        role="fine-grained change benchmark candidate",
    ),
    DatasetSpec(
        name="S2Looking",
        stage="stage_3_transition_aware",
        rel_path="S2Looking",
        role="change localization follow-up benchmark",
    ),
    DatasetSpec(
        name="xBD",
        stage="stage_3_transition_aware",
        rel_path="xBD",
        role="disaster-oriented change retrieval follow-up benchmark",
    ),
    DatasetSpec(
        name="RSCC",
        stage="stage_4_scale_up",
        rel_path="RSCC",
        role="larger caption-rich pre/post disaster dataset for scale-up",
    ),
    DatasetSpec(
        name="RSRCC",
        stage="stage_4_scale_up",
        rel_path="RSRCC",
        role="reasoning-oriented change questions for later evaluation",
    ),
    DatasetSpec(
        name="ChangeIMTI",
        stage="stage_4_scale_up",
        rel_path="ChangeIMTI",
        role="multi-task instruction-style change data",
    ),
    DatasetSpec(
        name="CC-Foundation",
        stage="stage_4_scale_up",
        rel_path="CC-Foundation",
        role="large-scale caption expansion after initial benchmarks",
    ),
    DatasetSpec(
        name="UCCD",
        stage="stage_4_scale_up",
        rel_path="UCCD",
        role="urban construction captioning scale-up benchmark",
    ),
    DatasetSpec(
        name="DynamicEarthNet",
        stage="stage_5_temporal_later",
        rel_path="DynamicEarthNet",
        role="later temporal trend retrieval and prediction",
    ),
    DatasetSpec(
        name="SpaceNet7",
        stage="stage_5_temporal_later",
        rel_path="SpaceNet7",
        role="later trajectory retrieval and temporal prediction",
    ),
)


def datasets_by_stage() -> dict[str, list[DatasetSpec]]:
    grouped: dict[str, list[DatasetSpec]] = {}
    for spec in DATASET_CURRICULUM:
        grouped.setdefault(spec.stage, []).append(spec)
    return grouped


def curriculum_summary(project_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in DATASET_CURRICULUM:
        path = spec.path_for(project_root)
        rows.append(
            {
                "name": spec.name,
                "stage": spec.stage,
                "role": spec.role,
                "path": str(path),
                "exists": path.exists(),
                "default_bootstrap": spec.default_bootstrap,
                "notes": list(spec.notes),
            }
        )
    return rows
