from __future__ import annotations

import torch
import pytest

from land_change_detection.models.qcpr_v3_teacher import FrozenV1Teacher, teacher_preservation_losses
from land_change_detection.models.qcpr_v3_runtime import IMMUTABLE_V1, checkpoint_config


class TinyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 3, bias=False)

    def forward(self, images, captions, caption_to_pair, temporal_valid_mask=None):
        pair = self.encoder(images)
        text = self.encoder(torch.ones(len(captions), 3))
        return type("Output", (), {"pair_embedding": pair, "text_embedding": text})()


def test_teacher_is_strict_frozen_separate_and_absent_from_student_optimizer(tmp_path) -> None:
    source = TinyTeacher()
    path = tmp_path / "v1.pt"
    torch.save({"model": source.state_dict()}, path)
    teacher = FrozenV1Teacher(TinyTeacher(), path)
    student = TinyTeacher()
    optimizer = torch.optim.AdamW(student.parameters())
    teacher_ids = {id(parameter) for parameter in teacher.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert teacher_ids.isdisjoint(optimizer_ids)
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    output = teacher(torch.randn(2, 3), ["a", "b"], torch.arange(2))
    losses = teacher_preservation_losses(output.pair_embedding, output.text_embedding, output.similarity, output)
    assert all(torch.isfinite(value) for value in losses.values())


@pytest.mark.skipif(not IMMUTABLE_V1.exists(), reason="cluster immutable v1 checkpoint unavailable")
def test_real_v1_checkpoint_strict_loading() -> None:
    import ucv2_cluster_common as common

    model = common.build_model(checkpoint_config(IMMUTABLE_V1), torch.device("cpu"))
    payload = torch.load(IMMUTABLE_V1, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
