from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TeacherOutput:
    pair_embedding: Tensor
    text_embedding: Tensor
    similarity: Tensor


class FrozenV1Teacher(nn.Module):
    """Separate immutable teacher, never registered as part of the student."""

    def __init__(self, model: nn.Module, checkpoint_path: str | Path, *, state_key: str = "model"):
        super().__init__()
        self.model = model
        self.checkpoint_path = Path(checkpoint_path)
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = payload[state_key] if state_key in payload else payload
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        digest = hashlib.sha256()
        with self.checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.checkpoint_sha256 = digest.hexdigest()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, images: Tensor, captions: list[str], caption_to_pair: Tensor, temporal_valid_mask: Tensor | None = None) -> TeacherOutput:
        output = self.model(images, captions, caption_to_pair, temporal_valid_mask)
        pair = F.normalize(output.pair_embedding, dim=-1)
        text = F.normalize(output.text_embedding, dim=-1)
        return TeacherOutput(pair, text, text @ pair.T)


def teacher_preservation_losses(
    student_pair: Tensor,
    student_text: Tensor,
    student_similarity: Tensor,
    teacher: TeacherOutput,
    *,
    temperature: float = 0.07,
) -> dict[str, Tensor]:
    pair = 1.0 - F.cosine_similarity(student_pair, teacher.pair_embedding, dim=-1).mean()
    text = 1.0 - F.cosine_similarity(student_text, teacher.text_embedding, dim=-1).mean()
    teacher_distribution = F.softmax(teacher.similarity / temperature, dim=-1)
    similarity = F.kl_div(
        F.log_softmax(student_similarity / temperature, dim=-1),
        teacher_distribution,
        reduction="batchmean",
    ) * temperature**2
    return {"pair_preservation": pair, "text_preservation": text, "similarity_distillation": similarity}
