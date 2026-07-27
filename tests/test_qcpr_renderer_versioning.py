from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pytest

from render_unichange_v2_retrieval import _assert_no_nonfinite, render_temporal_channel_panels, temporal_channel_render_status


def test_qcpr_v1_renderer_writes_grey_unavailable_panels(tmp_path) -> None:
    status = temporal_channel_render_status({}, "v1")
    figure, axes = plt.subplots(1, 3)
    render_temporal_channel_panels(
        axes,
        {name: np.zeros((8, 8)) for name in ("changed", "appeared", "disappeared")},
        status,
    )
    output = tmp_path / "v1_temporal_channels.png"
    figure.savefig(output)
    plt.close(figure)
    assert output.stat().st_size > 0
    assert status["temporal_channels_available"] is False
    assert status["temporal_channels_trained"] is False
    assert all(not axis.images for axis in axes)
    assert all("Unavailable for QCPR v1" in axis.texts[0].get_text() for axis in axes)


def test_untrained_v2_channels_are_not_presented_as_explanations() -> None:
    status = temporal_channel_render_status({}, "v2")
    figure, axes = plt.subplots(1, 3)
    render_temporal_channel_panels(
        axes,
        {name: np.ones((8, 8)) for name in ("changed", "appeared", "disappeared")},
        status,
    )
    assert status["temporal_channels_available"] is True
    assert status["temporal_channels_trained"] is False
    assert all(not axis.images for axis in axes)
    plt.close(figure)


def test_trained_v2_channels_record_supervision_and_gradient_provenance() -> None:
    checkpoint = {
        "temporal_channels_trained": True,
        "temporal_channel_provenance": {
            "supervision": {"appeared": "S2Looking label1", "disappeared": "S2Looking label2"},
            "gradients": {"temporal_channel_head": "finite_nonzero"},
        },
    }
    status = temporal_channel_render_status(checkpoint, "v2")
    assert status["temporal_channels_trained"] is True
    assert status["supervision_provenance"]["appeared"] == "S2Looking label1"
    assert status["gradient_provenance"]["temporal_channel_head"] == "finite_nonzero"


def test_evaluation_artifacts_reject_nan_and_inf() -> None:
    _assert_no_nonfinite({"finite": [0.0, 1.0]})
    with pytest.raises(ValueError, match="Non-finite"):
        _assert_no_nonfinite({"bad": float("nan")})
