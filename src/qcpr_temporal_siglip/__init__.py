"""Direct CLIP-like temporal retrieval with a post-retrieval soft map."""

from .backbone import TemporalSigLIPBackbone
from .config import TemporalSigLIPConfig
from .evaluator import direct_retrieval_metrics, rank_pairs
from .localization import TemporalSoftChangeMap, SoftChangeMapOutput
from .model import TemporalSigLIP, TemporalSigLIPOutput
from .objective import (
    RelevanceMasks,
    build_relevance_masks,
    positive_weights_from_grades,
    symmetric_mult_positive_clip_loss,
)

__all__ = [
    "RelevanceMasks",
    "SoftChangeMapOutput",
    "TemporalSigLIP",
    "TemporalSigLIPBackbone",
    "TemporalSigLIPConfig",
    "TemporalSigLIPOutput",
    "TemporalSoftChangeMap",
    "build_relevance_masks",
    "direct_retrieval_metrics",
    "positive_weights_from_grades",
    "rank_pairs",
    "symmetric_mult_positive_clip_loss",
]
