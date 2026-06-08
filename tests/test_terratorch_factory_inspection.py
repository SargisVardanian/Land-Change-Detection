from __future__ import annotations


def test_inspect_terratorch_factories_contract():
    from land_change_detection.training.prithvi_experimental import inspect_terratorch_factories

    payload = inspect_terratorch_factories()
    assert "available" in payload
    assert "factory_candidates" in payload
