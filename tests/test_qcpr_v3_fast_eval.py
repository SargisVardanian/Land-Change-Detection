import torch

import evaluate_qcpr_v3_fast as fast_eval


def test_pad_token_batches_preserves_variable_length_tokens_and_attention():
    first_tokens = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    second_tokens = torch.arange(12, 24, dtype=torch.float32).reshape(1, 4, 3)
    first_attention = torch.tensor([[True, True], [True, False]])
    second_attention = torch.tensor([[True, True, True, False]])

    tokens, attention = fast_eval.pad_token_batches(
        [first_tokens, second_tokens], [first_attention, second_attention]
    )

    assert tokens.shape == (3, 4, 3)
    assert attention.shape == (3, 4)
    torch.testing.assert_close(tokens[0, :2], first_tokens[0])
    torch.testing.assert_close(tokens[2], second_tokens[0])
    assert not attention[0, 2:].any()
    assert attention.dtype == torch.bool
