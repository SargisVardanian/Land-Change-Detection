from argparse import Namespace
from pathlib import Path

from scripts.train_qcpr_emergent_epochs import build_training_command


def _args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "output_dir": Path("/tmp/out"), "manifest_dir": Path("/tmp/manifests"),
        "batch_size": 32, "num_workers": 8, "seed": 7, "track": "B",
        "initialization_checkpoint": Path("/tmp/a0.pt"),
        "learning_rate": None, "text_adapter_learning_rate": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_b_warm_start_never_passes_a0_as_resume_checkpoint() -> None:
    command = build_training_command(
        _args(), epoch=1, total_steps=10, phase="late_interaction", resume_checkpoint=None,
    )
    assert command[command.index("--v3-checkpoint") + 1] == "/tmp/a0.pt"
    assert "--resume-checkpoint" not in command


def test_b_resume_keeps_a0_model_initialization_separate_from_b_state() -> None:
    command = build_training_command(
        _args(), epoch=2, total_steps=20, phase="late_interaction",
        resume_checkpoint=Path("/tmp/b-smoke-last.pt"),
    )
    assert command[command.index("--v3-checkpoint") + 1] == "/tmp/a0.pt"
    assert command[command.index("--resume-checkpoint") + 1] == "/tmp/b-smoke-last.pt"
