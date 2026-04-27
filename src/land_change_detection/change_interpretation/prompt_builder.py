from __future__ import annotations

from ..contracts import CellPack


def build_cell_prompt(pack: CellPack) -> str:
    legend_lines = [f"{class_id}: {label}" for class_id, label in sorted(pack.legend.items())]
    return "\n".join(
        [
            "You are analyzing one grid cell from aligned remote-sensing before/after imagery.",
            "Inputs include BEFORE crop, AFTER crop, BEFORE semantic overlay, and AFTER semantic overlay.",
            "Use the raw imagery as primary evidence. Use semantic overlays as secondary support only.",
            "Do not assume buildings, roads, vegetation change, or construction unless visually supported.",
            "Return strict JSON with: cell_id, before_observations, after_observations, stable_elements, changed_elements, primary_transition, alternative_hypotheses, uncertainty, final_summary.",
            f"Cell id: {pack.cell.cell_id}",
            "Legend:",
            *legend_lines,
        ]
    )
