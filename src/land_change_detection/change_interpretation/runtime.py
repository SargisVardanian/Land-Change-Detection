from __future__ import annotations

from typing import Protocol

from ..contracts import CellPack
from .contracts import CellInterpretation, SceneInterpretation


class CellInterpreterBackend(Protocol):
    def interpret_cell(self, pack: CellPack) -> CellInterpretation:
        ...


class ChangeInterpreterRuntime:
    def __init__(self, backend: CellInterpreterBackend):
        self.backend = backend

    def interpret_cell(self, pack: CellPack) -> CellInterpretation:
        return self.backend.interpret_cell(pack)

    def interpret_scene(self, packs: list[CellPack]) -> SceneInterpretation:
        reports = [self.interpret_cell(pack) for pack in packs]
        return SceneInterpretation(
            scene_overview="",
            cell_reports=reports,
            global_uncertainty="medium",
            metadata={"backend": type(self.backend).__name__},
        )
