from __future__ import annotations

from types import SimpleNamespace

import pytest

from qcpr_siglip2.backbones.siglip2 import (
    synchronize_tokenizer_config,
    validate_tokenizer_contract,
)


def test_tokenizer_contract_accepts_valid_ids_and_vocab() -> None:
    tokenizer = SimpleNamespace(
        vocab_size=256000,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    contract = validate_tokenizer_contract(tokenizer, 256000)
    assert contract["special_token_ids"] == {
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
    }
    assert contract["special_token_ids_in_range"] is True


def test_tokenizer_contract_rejects_out_of_range_ids() -> None:
    tokenizer = SimpleNamespace(
        vocab_size=32000,
        bos_token_id=49406,
        eos_token_id=49407,
        pad_token_id=1,
    )
    with pytest.raises(ValueError, match="exceed"):
        validate_tokenizer_contract(tokenizer, 32000)


def test_runtime_text_config_is_synchronized_without_weight_changes() -> None:
    text_config = SimpleNamespace(
        bos_token_id=49406,
        eos_token_id=49407,
        pad_token_id=1,
    )
    model_config = SimpleNamespace(
        text_config=SimpleNamespace(
            bos_token_id=49406,
            eos_token_id=49407,
            pad_token_id=1,
        )
    )
    model = SimpleNamespace(text_model=SimpleNamespace(config=text_config), config=model_config)
    contract = {
        "special_token_ids": {
            "bos_token_id": 2,
            "eos_token_id": 1,
            "pad_token_id": 0,
        }
    }
    result = synchronize_tokenizer_config(model, contract)
    assert result["config_ids_before_sync"] == {
        "bos_token_id": 49406,
        "eos_token_id": 49407,
        "pad_token_id": 1,
    }
    assert result["config_ids_after_sync"] == {
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
    }
    assert model.text_model.config.bos_token_id == 2
    assert model.config.text_config.pad_token_id == 0
