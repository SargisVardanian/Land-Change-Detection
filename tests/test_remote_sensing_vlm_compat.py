import os

import pytest

transformers = pytest.importorskip("transformers")
LlamaTokenizer = pytest.importorskip("transformers").LlamaTokenizer

from land_change_detection.remote_sensing_vlm import EARTHDIAL_RGB_DIR, _ensure_earthdial_import_path, _patch_transformers_compat


def test_earthdial_env_compat_sets_protobuf_mode() -> None:
    _ensure_earthdial_import_path()
    assert os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") == "python"


def test_transformers_compat_patches_earthdial_requirements() -> None:
    _patch_transformers_compat()
    assert hasattr(transformers, "EncoderDecoderCache")
    assert hasattr(transformers.utils, "is_flash_attn_greater_or_equal_2_10")


def test_earthdial_slow_tokenizer_loads_when_local_model_exists() -> None:
    if not EARTHDIAL_RGB_DIR.exists():
        return
    _ensure_earthdial_import_path()
    tokenizer = LlamaTokenizer.from_pretrained(EARTHDIAL_RGB_DIR, trust_remote_code=True, use_fast=False)
    assert type(tokenizer).__name__ == "LlamaTokenizer"
