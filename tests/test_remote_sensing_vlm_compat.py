import os

import transformers

from land_change_detection.remote_sensing_vlm import _ensure_earthdial_import_path, _patch_transformers_compat


def test_earthdial_env_compat_sets_protobuf_mode() -> None:
    _ensure_earthdial_import_path()
    assert os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") == "python"


def test_transformers_compat_patches_earthdial_requirements() -> None:
    _patch_transformers_compat()
    assert hasattr(transformers, "EncoderDecoderCache")
    assert hasattr(transformers.utils, "is_flash_attn_greater_or_equal_2_10")
