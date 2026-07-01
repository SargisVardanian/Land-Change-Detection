from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskRoute:
    name: str
    objective: str


class CyclicTaskSampler:
    def __init__(self, routes: list[TaskRoute | str]):
        if not routes:
            raise ValueError("routes must not be empty")
        self.routes = [route if isinstance(route, TaskRoute) else TaskRoute(str(route), str(route)) for route in routes]
        self.index = 0

    def next(self) -> TaskRoute:
        route = self.routes[self.index % len(self.routes)]
        self.index += 1
        return route

    def state_dict(self) -> dict[str, int]:
        return {"index": self.index}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.index = int(state.get("index", 0))


def retrieval_stage_sampler() -> CyclicTaskSampler:
    return CyclicTaskSampler(
        [
            TaskRoute("text_to_pair", "multi_positive_symmetric_info_nce"),
            TaskRoute("text_to_pair", "multi_positive_symmetric_info_nce"),
            TaskRoute("text_to_pair", "multi_positive_symmetric_info_nce"),
            TaskRoute("pair_to_pair", "supervised_contrastive"),
        ]
    )


def segmentation_stage_sampler() -> CyclicTaskSampler:
    return CyclicTaskSampler(
        [
            TaskRoute("binary_segmentation", "bce_dice"),
            TaskRoute("semantic_segmentation", "class_weighted_cross_entropy"),
            TaskRoute("binary_segmentation", "bce_dice"),
            TaskRoute("semantic_segmentation", "class_weighted_cross_entropy"),
        ]
    )
