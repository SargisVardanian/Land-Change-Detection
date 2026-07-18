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


def test_write_progress_is_atomic_and_has_eta(tmp_path, monkeypatch):
    ticks = iter((15.0,))
    monkeypatch.setattr(fast_eval.time, "monotonic", lambda: next(ticks))
    path = tmp_path / "progress.json"
    progress = fast_eval.write_progress(
        path, stage="scoring", completed_chunks=2, total_chunks=4,
        started_monotonic=10.0, query_count=100, candidate_count=20,
    )
    assert path.exists() and not path.with_suffix(".json.tmp").exists()
    assert progress["completed_fraction"] == 0.5
    assert progress["eta_seconds"] == 5.0
    assert not progress["complete"]
    assert __import__("json").loads(path.read_text())["total_chunks"] == 4


def test_score_uses_only_global_top_n_candidates():
    from types import SimpleNamespace
    class Model:
        def __init__(self): self.candidate_counts = []
        def score_encoded(
            self, pairs, per_time, text, tokens, attention, content, *,
            decode_mask=True,
        ):
            self.candidate_counts.append(pairs.shape[0])
            assert decode_mask is False
            shape = (text.shape[0], pairs.shape[0])
            return SimpleNamespace(local_score=torch.ones(shape), token_patch_score=torch.full(shape, 2.0), reranked_score=torch.full(shape, 3.0), decoded_mask_logits=torch.zeros(*shape, 1, 1))
    model = Model()
    corpus = {
        "text": torch.tensor([[1.0], [-1.0]]), "pairs": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        "per_time": torch.zeros(4, 2, 1, 1), "tokens": torch.zeros(2, 1, 1),
        "attention": torch.ones(2, 1, dtype=torch.bool), "content": torch.ones(2, 1, dtype=torch.bool),
        "mapping": torch.tensor([0, 3]), "segmentation": torch.zeros(4, dtype=torch.bool), "masks": torch.zeros(4, 1, 1),
    }
    global_scores, local, token, reranked, _, _ = fast_eval.score(model, corpus, torch.device("cpu"), 2, 1, False, rerank_top_n=1)
    assert model.candidate_counts == [2]
    selected = global_scores.topk(1, dim=1).indices
    assert torch.isfinite(token.gather(1, selected)).all()
    assert (token == float("-inf")).sum() == 6
    assert torch.equal(reranked.gather(1, selected), torch.full((2, 1), 3.0))
