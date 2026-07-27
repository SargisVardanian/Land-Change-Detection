from __future__ import annotations


def test_inspect_terratorch_module_contract():
    from land_change_detection.training.prithvi_experimental import inspect_terratorch_module

    payload = inspect_terratorch_module()
    assert "available" in payload
    assert "candidate_submodules" in payload
