from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from land_change_detection.models.qcpr_v3 import QCPRV3ScoreOutput


@dataclass(frozen=True)
class CanonicalV3Inputs:
    global_query_embeddings: Tensor
    text_token_embeddings: Tensor
    text_attention_mask: Tensor
    pair_embeddings: Tensor
    per_time_tokens: Tensor
    text_content_mask: Tensor | None = None

    def as_kwargs(self) -> dict[str, Tensor]:
        return {
            "global_query_embeddings": self.global_query_embeddings,
            "text_token_embeddings": self.text_token_embeddings,
            "text_attention_mask": self.text_attention_mask,
            "text_content_mask": self.text_content_mask,
            "pair_embeddings": self.pair_embeddings,
            "per_time_tokens": self.per_time_tokens,
        }


def canonical_score(model, inputs: CanonicalV3Inputs) -> QCPRV3ScoreOutput:
    """The only trainer/evaluator/renderer scoring adapter."""
    grounder = getattr(model, "grounder", model)
    return grounder.score_query_pair_chunks(
        inputs.global_query_embeddings,
        inputs.text_token_embeddings,
        inputs.text_attention_mask,
        inputs.pair_embeddings,
        inputs.per_time_tokens,
        text_content_mask=inputs.text_content_mask,
    )


def trainer_score(model, inputs: CanonicalV3Inputs) -> QCPRV3ScoreOutput:
    return canonical_score(model, inputs)


def evaluator_score(model, inputs: CanonicalV3Inputs) -> QCPRV3ScoreOutput:
    return canonical_score(model, inputs)


def renderer_score(model, inputs: CanonicalV3Inputs) -> QCPRV3ScoreOutput:
    return canonical_score(model, inputs)


def faithful_mask_score(model, inputs: CanonicalV3Inputs) -> Tensor:
    """Recompute the local score through the canonical mask/pooling path only."""
    return canonical_score(model, inputs).local_score
